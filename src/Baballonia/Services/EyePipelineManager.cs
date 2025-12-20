using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.Filters;
using Baballonia.Services.Inference.Models;
using Baballonia.Services.Inference.VideoSources;
using Microsoft.Extensions.Logging;
using OpenCvSharp;

namespace Baballonia.Services;

/// <summary>
/// This class should be the only place where direct Pipeline modifications happen
/// </summary>
public class EyePipelineManager
{
    private readonly ILogger<EyePipelineManager> _logger;
    private readonly EyeProcessingPipeline _pipeline;
    private readonly ILocalSettingsService _localSettings;
    private readonly InferenceFactory _inferenceFactory;
    private readonly SingleCameraSourceFactory _singleCameraSourceFactory;

    private string? _currentLeftAddress;
    private string? _currentRightAddress;

    public EyePipelineManager(ILogger<EyePipelineManager> logger, EyeProcessingPipeline pipeline,
        ILocalSettingsService localSettings, InferenceFactory inferenceFactory,
        SingleCameraSourceFactory singleCameraSourceFactory)
    {
        _logger = logger;
        _pipeline = pipeline;
        _localSettings = localSettings;
        _inferenceFactory = inferenceFactory;
        _singleCameraSourceFactory = singleCameraSourceFactory;

        InitializePipeline();
    }


    public void InitializePipeline()
    {
        _pipeline.ImageConverter = new MatToFloatTensorConverter();
        var dualTransformer = new DualImageTransformer();
        dualTransformer.LeftTransformer.TargetSize = new Size(128, 128);
        dualTransformer.RightTransformer.TargetSize = new Size(128, 128);
        _pipeline.ImageTransformer = dualTransformer;
        // Read generalized eyelid mode setting so mirroring can be applied even when enhancer is not used
        var modeStr = _localSettings.ReadSetting<string>("EyeHome_EyelidMode", "Both");
        _pipeline.EyelidModeSetting = modeStr switch
        {
            "LeftOnly" => EyeProcessingPipeline.EyelidMode.LeftOnly,
            "RightOnly" => EyeProcessingPipeline.EyelidMode.RightOnly,
            _ => EyeProcessingPipeline.EyelidMode.Both
        };

        _ = LoadInferenceAsync();
        LoadFilter();
        LoadEyeStabilization();
        LoadEyelidEnhancer();
    }

    public async Task LoadInferenceAsync()
    {
        var inf = await Task.Run(CreateInference);
        _pipeline.InferenceService = inf;
    }

    private DefaultInferenceRunner CreateInference()
    {
        const string defaultEyeModelName = "eyeModel.onnx";
        var eyeModelName = _localSettings.ReadSetting<string>("EyeHome_EyeModel", defaultEyeModelName);
        var eyeModelPath = Path.Combine(AppContext.BaseDirectory, eyeModelName);

        if (File.Exists(eyeModelPath)) return _inferenceFactory.Create(eyeModelPath);
        _logger.LogError("{} Does not exists, Loading default...", eyeModelPath);

        eyeModelName = defaultEyeModelName;
        eyeModelPath = Path.Combine(AppContext.BaseDirectory, eyeModelName);

        return _inferenceFactory.Create(eyeModelPath);
    }


    public void LoadInference()
    {
        _pipeline.InferenceService = CreateInference();
    }

    public void LoadFilter()
    {
        var enabled = _localSettings.ReadSetting<bool>("AppSettings_OneEuroEnabled");
        var cutoff = _localSettings.ReadSetting<float>("AppSettings_OneEuroMinFreqCutoff");
        var speedCutoff = _localSettings.ReadSetting<float>("AppSettings_OneEuroSpeedCutoff");

        if (!enabled)
            return;

        var eyeArray = new float[Utils.EyeRawExpressions];
        var eyeFilter = new OneEuroFilter(
            eyeArray,
            minCutoff: cutoff,
            beta: speedCutoff
        );

        _pipeline.Filter = eyeFilter;
    }

    public void LoadEyeStabilization()
    {
        var stabilizeEyes = _localSettings.ReadSetting<bool>("AppSettings_StabilizeEyes", true);
        _pipeline.StabilizeEyes = stabilizeEyes;
    }

    public void SetEyelidMode(EyeProcessingPipeline.EyelidMode mode)
    {
        _pipeline.EyelidModeSetting = mode;
        try
        {
            _localSettings.SaveSetting("EyeHome_EyelidMode", mode.ToString());
        }
        catch { }

        // Recreate enhancer to pick up new mode
        LoadEyelidEnhancer();
    }

    public Task<bool> CalibrateEyelidsAsync(TimeSpan duration, CancellationToken cancellationToken = default)
    {
        if (_pipeline.EyelidEnhancer == null)
        {
            _logger.LogWarning("Eyelid calibration requested but enhancer is not available.");
            return Task.FromResult(false);
        }

        return _pipeline.EyelidEnhancer.CalibrateAsync(duration, cancellationToken);
    }

    internal void LoadEyelidEnhancer()
    {
        (_pipeline.EyelidEnhancer as IDisposable)?.Dispose();
        _pipeline.EyelidEnhancer = null;

        // Read setting; default true for backward compatibility
        var enabled = _localSettings.ReadSetting<bool>("EyeHome_EnablePfldEyelidModel", true);
        if (!enabled)
        {
            _logger.LogInformation("Pfld eyelid enhancer disabled through settings.");
            return;
        }

        const string defaultModelName = "pfld-sim.onnx";
        var modelName = _localSettings.ReadSetting<string>("EyeHome_EyelidModel", defaultModelName);
        var modelPath = Path.Combine(AppContext.BaseDirectory, modelName);
        if (!File.Exists(modelPath))
        {
            _logger.LogWarning("Pfld eyelid model {ModelPath} was not found; enhancer disabled.", modelPath);
            return;
        }

        try
        {
            var runner = _inferenceFactory.Create(modelPath);
            // Read mode setting: "Both" (default), "LeftOnly", "RightOnly"
            var modeStr = _localSettings.ReadSetting<string>("EyeHome_EyelidMode", "Both");
            var mode = modeStr switch
            {
                "LeftOnly" => EyeProcessingPipeline.EyelidMode.LeftOnly,
                "RightOnly" => EyeProcessingPipeline.EyelidMode.RightOnly,
                _ => EyeProcessingPipeline.EyelidMode.Both
            };

            _pipeline.EyelidEnhancer = new PfldSimEyelidEnhancer(runner, _logger, _localSettings, mode);
            _logger.LogInformation("Initialized pfld eyelid enhancer with {Model} (mode={Mode})", modelName, mode);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize pfld eyelid enhancer using {ModelPath}", modelPath);
        }
    }

    public void SetLeftTransformation(CameraSettings cameraSettings)
    {
        if (_pipeline.ImageTransformer is DualImageTransformer dualImageTransformer)
        {
            dualImageTransformer.LeftTransformer.Transformation = cameraSettings;
        }
    }
    public void SetRightTransformation(CameraSettings cameraSettings)
    {
        if (_pipeline.ImageTransformer is DualImageTransformer dualImageTransformer)
        {
            dualImageTransformer.RightTransformer.Transformation = cameraSettings;
        }
    }

    public async Task<bool> StartLeftVideoSource(string cameraAddress, string preferredBackend)
    {
        if (string.IsNullOrEmpty(cameraAddress))
            return false;

        if (_pipeline.VideoSource == null)
        {
            SingleCameraSource cam;
            if (string.IsNullOrEmpty(preferredBackend))
                cam = await _singleCameraSourceFactory.CreateStart(cameraAddress);
            else
                cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);

            if (cam == null)
                return false;

            var source = new DualCameraSource();
            source.LeftCam = cam;
            _pipeline.VideoSource = source;
            _currentLeftAddress = cameraAddress;
            return true;
        }

        if (_pipeline.VideoSource is DualCameraSource dualCameraSource)
            if (cameraAddress == _currentRightAddress && _currentRightAddress != null)
            {
                var tmp = dualCameraSource.RightCam;
                _pipeline.VideoSource = tmp;
                _currentLeftAddress = cameraAddress;
                return true;
            }
            else
            {
                if (dualCameraSource.LeftCam != null)
                {
                    dualCameraSource.LeftCam.Dispose();
                    dualCameraSource.LeftCam = null;
                }

                var cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);
                if (cam == null)
                    return false;
                dualCameraSource.LeftCam = cam;
                _currentLeftAddress = cameraAddress;
                return true;
            }

        if (_pipeline.VideoSource is SingleCameraSource singleCameraSource)
        {
            if (_currentLeftAddress == cameraAddress && _currentLeftAddress != null)
                return true;

            var cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);
            if (cam == null)
                return false;

            var tmp = singleCameraSource;
            _pipeline.VideoSource = null;
            var source = new DualCameraSource();
            source.LeftCam = cam;
            source.RightCam = tmp;

            _currentLeftAddress = cameraAddress;
            return true;
        }

        return true;
    }

    public async Task<bool> StartRightVideoSource(string cameraAddress, string preferredBackend)
    {
        if (string.IsNullOrEmpty(cameraAddress))
            return false;

        if (_pipeline.VideoSource == null)
        {
            SingleCameraSource cam;
            if (string.IsNullOrEmpty(preferredBackend))
                cam = await _singleCameraSourceFactory.CreateStart(cameraAddress);
            else
                cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);

            if (cam == null)
                return false;

            var source = new DualCameraSource();
            source.RightCam = cam;
            _pipeline.VideoSource = source;
            _currentRightAddress = cameraAddress;
            return true;
        }

        if (_pipeline.VideoSource is DualCameraSource dualCameraSource)
            if (cameraAddress == _currentLeftAddress && _currentLeftAddress != null)
            {
                var tmp = dualCameraSource.LeftCam;
                _pipeline.VideoSource = tmp;
                _currentRightAddress = cameraAddress;
                return true;
            }
            else
            {
                if (dualCameraSource.RightCam != null)
                {
                    dualCameraSource.RightCam.Dispose();
                    dualCameraSource.RightCam = null;
                }

                var cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);
                if (cam == null)
                    return false;
                dualCameraSource.RightCam = cam;
                _currentRightAddress = cameraAddress;
                return true;
            }

        if (_pipeline.VideoSource is SingleCameraSource singleCameraSource)
        {
            if (_currentRightAddress == cameraAddress && _currentRightAddress != null)
                return true;

            var cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);
            if (cam == null)
                return false;

            var tmp = singleCameraSource;
            _pipeline.VideoSource = null;
            var source = new DualCameraSource();
            source.RightCam = cam;
            source.LeftCam = tmp;

            _currentRightAddress = cameraAddress;
            return true;
        }

        return true;
    }

    public async Task<bool> TryStartLeftIfNotRunning(string cameraAddress, string preferredBackend)
    {
        switch (_pipeline.VideoSource)
        {
            case SingleCameraSource singleCameraSource:
            case DualCameraSource { LeftCam: not null }:
                return true;
            default:
                return await StartLeftVideoSource(cameraAddress, preferredBackend);
        }
    }
    public async Task<bool> TryStartRightIfNotRunning(string cameraAddress, string preferredBackend)
    {
        switch (_pipeline.VideoSource)
        {
            case SingleCameraSource singleCameraSource:
            case DualCameraSource { RightCam: not null }:
                return true;
            default:
                return await StartRightVideoSource(cameraAddress, preferredBackend);
        }
    }
    public void StopLeftCamera()
    {
        _currentLeftAddress = null;
        if (_pipeline.VideoSource is DualCameraSource dualCameraSource)
        {
            dualCameraSource.LeftCam?.Dispose();
            dualCameraSource.LeftCam = null;
        }

        if (_pipeline.VideoSource is SingleCameraSource singleCameraSource)
        {
            singleCameraSource.Dispose();
            _pipeline.VideoSource = null;
            _currentRightAddress = null;
        }
    }

    public void StopRightCamera()
    {
        _currentRightAddress = null;
        if (_pipeline.VideoSource is DualCameraSource dualCameraSource)
        {
            dualCameraSource.RightCam?.Dispose();
            dualCameraSource.RightCam = null;
        }

        if (_pipeline.VideoSource is SingleCameraSource singleCameraSource)
        {
            singleCameraSource.Dispose();
            _pipeline.VideoSource = null;
            _currentLeftAddress = null;
        }
    }

    public void StopAllCameras()
    {
        _currentRightAddress = null;
        _currentLeftAddress = null;
        _pipeline.VideoSource?.Dispose();
        _pipeline.VideoSource = null;
    }

    public bool IsUsingSameCamera()
    {
        return _currentLeftAddress == _currentRightAddress && _currentLeftAddress != null;
    }

    public void SetFilter(IFilter? filter)
    {
        _pipeline.Filter = filter;
    }
}
