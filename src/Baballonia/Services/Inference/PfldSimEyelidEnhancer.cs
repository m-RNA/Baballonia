using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Microsoft.Extensions.Logging;
using OpenCvSharp;

namespace Baballonia.Services.Inference;

/// <summary>
/// Uses the pfld-sim landmark model to produce more stable eyelid openness values.
/// </summary>
public sealed class PfldSimEyelidEnhancer : IEyelidEnhancer
{
    private const double DefaultOpenRatio = 0.4;
    private const double DefaultClosedRatio = 0.1;
    private const double CalibrationNaturalOpenScale = 0.9;
    private const string EyelidCalibrationSettingKey = "EyelidCalibrationState";

    private readonly DefaultInferenceRunner _inferenceRunner;
    private readonly MatToFloatTensorConverter _converter = new();
    private readonly PercentileTracker _leftTracker = new();
    private readonly PercentileTracker _rightTracker = new();
    private readonly ILogger? _logger;
    private readonly ILocalSettingsService? _localSettings;
    private readonly object _calibrationLock = new();
    private readonly List<double> _leftCalibrationSamples = new();
    private readonly List<double> _rightCalibrationSamples = new();

    private Mat? _leftEye;
    private Mat? _rightEye;
    private bool _warnedAboutChannels;
    private TaskCompletionSource<bool>? _calibrationTcs;
    private CancellationTokenRegistration _calibrationCancellationRegistration;
    private readonly EyeProcessingPipeline.EyelidMode _mode;

    public PfldSimEyelidEnhancer(DefaultInferenceRunner inferenceRunner, ILogger? logger = null, ILocalSettingsService? localSettings = null, EyeProcessingPipeline.EyelidMode mode = EyeProcessingPipeline.EyelidMode.Both)
    {
        _inferenceRunner = inferenceRunner;
        _logger = logger;
        _localSettings = localSettings;
        _mode = mode;
        LoadPersistedCalibration();
    }

    public void CaptureEyeImages(Mat transformedEyes)
    {
        if (transformedEyes == null || transformedEyes.Empty())
            return;

        var channels = Cv2.Split(transformedEyes);
        if (channels.Length < 2)
        {
            foreach (var channel in channels)
            {
                channel.Dispose();
            }

            if (!_warnedAboutChannels)
            {
                _warnedAboutChannels = true;
                _logger?.LogWarning("Dual-eye transformer did not produce two channels; pfld eyelid enhancer skipped for this frame.");
            }
            return;
        }

        UpdateEye(ref _leftEye, channels[0]);
        UpdateEye(ref _rightEye, channels[1]);

        foreach (var channel in channels)
        {
            channel.Dispose();
        }
    }

    private void RecordCalibrationSample(bool isLeft, double normalized)
    {
        if (double.IsNaN(normalized))
            return;

        lock (_calibrationLock)
        {
            if (_calibrationTcs == null)
                return;

            var samples = isLeft ? _leftCalibrationSamples : _rightCalibrationSamples;
            samples.Add(normalized);
        }
    }

    public void Enhance(float[] expressions)
    {
        if (expressions == null || expressions.Length < 6)
            return;

        // Compute according to configured mode. Downstream expects index 2 = right lid, index 5 = left lid.
        switch (_mode)
        {
            case EyeProcessingPipeline.EyelidMode.Both:
            {
                var left = ProcessEye(_leftEye, _leftTracker, true);
                var right = ProcessEye(_rightEye, _rightTracker, false);
                if (right.HasValue) expressions[2] = right.Value;
                if (left.HasValue) expressions[5] = left.Value;
                break;
            }
            case EyeProcessingPipeline.EyelidMode.LeftOnly:
            {
                var left = ProcessEye(_leftEye, _leftTracker, true);
                if (left.HasValue)
                {
                    // Use left eye value for both sides
                    expressions[2] = left.Value; // right lid
                    //expressions[5] = left.Value; // left lid
                }
                break;
            }
            case EyeProcessingPipeline.EyelidMode.RightOnly:
            {
                var right = ProcessEye(_rightEye, _rightTracker, false);
                if (right.HasValue)
                {
                    // Use right eye value for both sides
                    //expressions[2] = right.Value; // right lid
                    expressions[5] = right.Value; // left lid
                }
                break;
            }
        }

    }

    public Task<bool> CalibrateAsync(TimeSpan duration, CancellationToken cancellationToken)
    {
        TaskCompletionSource<bool> calibrationTask;
        lock (_calibrationLock)
        {
            if (_calibrationTcs != null)
                return _calibrationTcs.Task;

            _leftCalibrationSamples.Clear();
            _rightCalibrationSamples.Clear();

            _calibrationTcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            calibrationTask = _calibrationTcs;

            if (cancellationToken.CanBeCanceled)
            {
                _calibrationCancellationRegistration = cancellationToken.Register(() =>
                {
                    lock (_calibrationLock)
                    {
                        FinishCalibration_NoLock(true);
                    }
                });
            }

            _logger?.LogInformation("Starting eyelid calibration window for {Duration:F1} seconds", duration.TotalSeconds);
            _ = CompleteCalibrationAfterDelayAsync(duration);
        }

        return calibrationTask.Task;
    }

    public void Dispose()
    {
        _leftEye?.Dispose();
        _leftEye = null;
        _rightEye?.Dispose();
        _rightEye = null;
    }

    private float? ProcessEye(Mat? eye, PercentileTracker tracker, bool isLeft)
    {
        if (eye == null || eye.Empty())
            return null;

        using var prepared = eye.Clone();
        try
        {
            _converter.Convert(prepared, _inferenceRunner.GetInputTensor());
            var outputs = _inferenceRunner.Run();
            var normalized = ComputeNormalizedOpenness(outputs);
            if (double.IsNaN(normalized))
                return null;

            tracker.Add(normalized);
            RecordCalibrationSample(isLeft, normalized);
            var (open, closed) = tracker.GetBounds();
            var lid = PercentileTracker.MapToLid(normalized, open, closed);
            return (float)lid;
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Failed to run pfld eyelid inference");
            return null;
        }
    }

    private static double ComputeNormalizedOpenness(float[] output)
    {
        // pfld-sim outputs 7 landmarks (14 numbers) but we only need indices 0-5 (12 numbers)
        if (output == null || output.Length < 12)
            return double.NaN;

        static double Distance(float[] values, int a, int b)
        {
            var ax = values[a * 2];
            var ay = values[a * 2 + 1];
            var bx = values[b * 2];
            var by = values[b * 2 + 1];
            var dx = ax - bx;
            var dy = ay - by;
            return Math.Sqrt(dx * dx + dy * dy);
        }

        var width = Distance(output, 0, 3);
        if (width <= double.Epsilon)
            return double.NaN;

        var v1 = Distance(output, 1, 5);
        var v2 = Distance(output, 2, 4);
        return (v1 + v2) * 0.5 / width;
    }

    private static void UpdateEye(ref Mat? target, Mat source)
    {
        target?.Dispose();
        target = source.Clone();
        if (target.Channels() > 1)
        {
            Cv2.CvtColor(target, target, ColorConversionCodes.BGR2GRAY);
        }

        if (target.Type() != MatType.CV_8UC1)
        {
            target.ConvertTo(target, MatType.CV_8UC1);
        }

        Cv2.EqualizeHist(target, target);
    }

    private void FinishCalibration_NoLock(bool canceled)
    {
        if (_calibrationTcs == null)
            return;

        _calibrationCancellationRegistration.Dispose();
        var tcs = _calibrationTcs;
        _calibrationTcs = null;

        if (canceled)
        {
            tcs.TrySetCanceled();
        }
        else
        {
            var (leftOpen, leftClosed) = CalculateCalibrationBounds(_leftCalibrationSamples);
            var (rightOpen, rightClosed) = CalculateCalibrationBounds(_rightCalibrationSamples);
            _leftTracker.OverrideBounds(leftOpen, leftClosed);
            _rightTracker.OverrideBounds(rightOpen, rightClosed);
            SaveCalibrationState(leftOpen, leftClosed, rightOpen, rightClosed);
            _logger?.LogInformation(
                "Eyelid calibration complete. Left(open={LeftOpen:F3}, closed={LeftClosed:F3}) Right(open={RightOpen:F3}, closed={RightClosed:F3})",
                leftOpen,
                leftClosed,
                rightOpen,
                rightClosed);
            tcs.TrySetResult(true);
        }

        _leftCalibrationSamples.Clear();
        _rightCalibrationSamples.Clear();
    }

    private static (double open, double closed) CalculateCalibrationBounds(List<double> samples)
    {
        if (samples.Count < 10)
            return (ExpandNaturalOpen(DefaultOpenRatio), DefaultClosedRatio);

        var sorted = samples.OrderBy(v => v).ToArray();
        var closed = PercentileTracker.Percentile(sorted, 0.02);
        var open = PercentileTracker.Percentile(sorted, 0.90);
        if (open - closed < 1e-3)
            return (DefaultOpenRatio, DefaultClosedRatio);

        return (ExpandNaturalOpen(open), closed);
    }

    private async Task CompleteCalibrationAfterDelayAsync(TimeSpan duration)
    {
        await Task.Delay(duration);
        lock (_calibrationLock)
        {
            FinishCalibration_NoLock(false);
        }
    }

    private sealed class PercentileTracker
    {
        private readonly Queue<double> _samples = new();
        private readonly int _capacity;
        private bool _hasManualBounds;
        private double _manualOpen;
        private double _manualClosed;

        public PercentileTracker(int capacity = 400)
        {
            _capacity = capacity;
        }

        public void Add(double value)
        {
            if (double.IsNaN(value) || double.IsInfinity(value))
                return;

            _samples.Enqueue(value);
            if (_samples.Count > _capacity)
            {
                _samples.Dequeue();
            }
        }

        public (double open, double closed) GetBounds()
        {
            if (_hasManualBounds)
                return (_manualOpen, _manualClosed);

            if (_samples.Count < 20)
                return (DefaultOpenRatio, DefaultClosedRatio);

            var sorted = _samples.OrderBy(v => v).ToArray();
            var closed = Percentile(sorted, 0.05);
            var open = Percentile(sorted, 0.95);
            if (open - closed < 1e-3)
                return (DefaultOpenRatio, DefaultClosedRatio);

            return (open, closed);
        }

        internal static double Percentile(IReadOnlyList<double> sorted, double percentile)
        {
            if (sorted.Count == 0)
                return DefaultClosedRatio;

            var clamped = Math.Clamp(percentile, 0d, 1d);
            var index = (int)Math.Round(clamped * (sorted.Count - 1));
            return sorted[index];
        }

        public static double MapToLid(double value, double open, double closed)
        {
            var range = Math.Max(open - closed, 1e-3);
            var scaled = (value - closed) / range;
            return Math.Clamp(1.0 - scaled, 0.0, 1.0);
        }

        public void OverrideBounds(double open, double closed)
        {
            _hasManualBounds = true;
            _manualOpen = open;
            _manualClosed = closed;
        }
    }

    private static double ExpandNaturalOpen(double open)
    {
        var scaled = open / CalibrationNaturalOpenScale;
        return Math.Min(1.0, scaled);
    }

    private void LoadPersistedCalibration()
    {
        if (_localSettings == null)
            return;

        var state = _localSettings.ReadSetting<EyelidCalibrationState?>(EyelidCalibrationSettingKey);
        if (state == null)
            return;

        if (state.LeftOpen > 0 && state.LeftClosed >= 0)
        {
            _leftTracker.OverrideBounds(state.LeftOpen, state.LeftClosed);
        }

        if (state.RightOpen > 0 && state.RightClosed >= 0)
        {
            _rightTracker.OverrideBounds(state.RightOpen, state.RightClosed);
        }
    }

    private void SaveCalibrationState(double leftOpen, double leftClosed, double rightOpen, double rightClosed)
    {
        if (_localSettings == null)
            return;

        var state = new EyelidCalibrationState
        {
            LeftOpen = leftOpen,
            LeftClosed = leftClosed,
            RightOpen = rightOpen,
            RightClosed = rightClosed
        };

        _localSettings.SaveSetting(EyelidCalibrationSettingKey, state);
    }

    private sealed class EyelidCalibrationState
    {
        public double LeftOpen { get; set; }
        public double LeftClosed { get; set; }
        public double RightOpen { get; set; }
        public double RightClosed { get; set; }
    }
}
