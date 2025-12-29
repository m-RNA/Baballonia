using System;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services;
using Microsoft.Extensions.Logging;
using OpenCvSharp;

namespace Baballonia.Services.Inference;

/// <summary>
/// Uses a 6-class eye-state classifier (normal/closed/squint/wide/wink_left/wink_right)
/// to derive eyelid openness and auxiliary eye shapes.
/// 
/// Expected ONNX input: (1,2,128,128) dual-channel grayscale with ch0=left eye, ch1=right eye.
/// Expected ONNX output: 6 logits or probabilities in the class order:
/// normal, closed, squint, wide, wink_left, wink_right.
/// </summary>
public sealed class EyeStateClassifierEyelidEnhancer : IEyelidEnhancer
{
    private readonly DefaultInferenceRunner _inferenceRunner;
    private readonly MatToFloatTensorConverter _converter = new();
    private readonly ILogger? _logger;
    private readonly EyeProcessingPipeline.EyelidMode _mode;

    // Stores the dual-channel Mat (ch0=left, ch1=right) from DualImageTransformer
    private Mat? _dualEyeFrame;
    private long _lastDebugLogTicks;

    private void LogLowFreq(string message, params object?[] args)
    {
        if (_logger == null)
            return;

        var now = Environment.TickCount64;
        if (now - _lastDebugLogTicks < 200)
            return;

        _lastDebugLogTicks = now;
        _logger.LogInformation(message, args);
    }

    public EyeStateClassifierEyelidEnhancer(
        DefaultInferenceRunner inferenceRunner,
        ILogger? logger = null,
        EyeProcessingPipeline.EyelidMode mode = EyeProcessingPipeline.EyelidMode.Both)
    {
        _inferenceRunner = inferenceRunner;
        _logger = logger;
        _mode = mode;
    }

    /// <summary>
    /// Captures the dual-eye frame from DualImageTransformer.
    /// 
    /// Input: DualImageTransformer output - 2-channel Mat where ch0=left eye, ch1=right eye (each 128x128).
    /// This format directly matches the model's expected input (1,2,128,128).
    /// </summary>
    public void CaptureEyeImages(Mat transformedEyes)
    {
        if (transformedEyes == null || transformedEyes.Empty())
            return;

        // DualImageTransformer outputs a 2-channel Mat (ch0=left, ch1=right), each 128x128
        if (transformedEyes.Channels() < 2)
        {
            LogLowFreq("EyeStateCls: Expected 2-channel input, got {Channels} channels", transformedEyes.Channels());
            return;
        }

        _dualEyeFrame?.Dispose();
        _dualEyeFrame = transformedEyes.Clone();
    }

    public void Enhance(float[] expressions)
    {
        // Eye model output expected by the pipeline is 6 floats (raw eye model output before ProcessExpressions).
        // Indices: 0=leftPitch, 1=leftYaw, 2=leftLid, 3=rightPitch, 4=rightYaw, 5=rightLid
        if (expressions == null || expressions.Length < 6)
        {
            LogLowFreq("EyeStateCls skipped: input eye expressions missing/too short (len={Len})", expressions?.Length ?? 0);
            return;
        }

        if (_dualEyeFrame == null || _dualEyeFrame.Empty())
        {
            LogLowFreq("EyeStateCls skipped: no captured dual-eye frame yet");
            return;
        }

        try
        {
            // Convert the dual-channel Mat directly to the input tensor
            // MatToFloatTensorConverter handles the HWC to NCHW conversion
            _converter.Convert(_dualEyeFrame, _inferenceRunner.GetInputTensor());

            // Diagnostics: report tensor shape and per-channel stats
            var tensor = _inferenceRunner.GetInputTensor();
            var c = tensor.Dimensions[1];
            var h = tensor.Dimensions[2];
            var w = tensor.Dimensions[3];

            // Log tensor stats for both channels
            for (int ch = 0; ch < Math.Min(c, 2); ch++)
            {
                float min = float.MaxValue, max = float.MinValue, sum = 0;
                int count = h * w;
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        var v = tensor[0, ch, y, x];
                        if (v < min) min = v;
                        if (v > max) max = v;
                        sum += v;
                    }
                }
                //LogLowFreq("EyeStateCls input ch{Ch} stats: min={Min:F4} max={Max:F4} mean={Mean:F4}", ch, min, max, sum / count);
            }
            //LogLowFreq("EyeStateCls input tensor shape: 1x{C}x{H}x{W}", c, h, w);

            var raw = _inferenceRunner.Run();
            if (raw == null || raw.Length < 6)
            {
                LogLowFreq("EyeStateCls skipped: model output too short (len={Len})", raw?.Length ?? 0);
                return;
            }

            var probs = Softmax(raw);

            var normal = probs[0];
            var closed = probs[1];
            var squint = probs[2];
            var wide = probs[3];
            var winkLeft = probs[4];
            var winkRight = probs[5];

            // Log prediction for debugging
            {
                var pred = 0;
                var best = probs[0];
                for (var i = 1; i < 6; i++)
                {
                    if (probs[i] > best)
                    {
                        best = probs[i];
                        pred = i;
                    }
                }
                LogLowFreq(
                    "EyeStateCls raw=[{R0:F3},{R1:F3},{R2:F3},{R3:F3},{R4:F3},{R5:F3}] probs=[{P0:F2},{P1:F2},{P2:F2},{P3:F2},{P4:F2},{P5:F2}] pred={Pred}",
                    raw[0], raw[1], raw[2], raw[3], raw[4], raw[5],
                    probs[0], probs[1], probs[2], probs[3], probs[4], probs[5],
                    pred);
            }

            // Convert classifier outputs to eyelid values.
            // The classifier provides: normal, closed, squint, wide, wink_left, wink_right
            // 
            // For eyelid openness:
            // - "closed" probability = both eyes closed
            // - "wink_left" = left eye closed (right open)
            // - "wink_right" = right eye closed (left open)
            // 
            // Openness calculation:
            // - Left eye openness = 1 - closed - winkLeft (left eye is closed when "closed" or "wink_left")
            // - Right eye openness = 1 - closed - winkRight (right eye is closed when "closed" or "wink_right")
            //
            // The raw expression array format (before ProcessExpressions):
            // [0]=leftPitch, [1]=leftYaw, [2]=leftLid, [3]=rightPitch, [4]=rightYaw, [5]=rightLid
            // ProcessExpressions converts: leftLid = 1 - expressions[2], rightLid = 1 - expressions[5]
            // So we need to write: expressions[2] = 1 - leftOpenness = closed + winkLeft
            //                      expressions[5] = 1 - rightOpenness = closed + winkRight

            var leftLidValue = Math.Clamp(closed + squint * 0.5f + winkLeft + winkRight * 0.15f + normal *0.05f - wide, 0.0f, 1.0f);
            var rightLidValue = Math.Clamp(closed + squint * 0.5f + winkRight + winkLeft * 0.15f + normal * 0.05f - wide, 0.0f, 1.0f);

            // Apply based on mode
            switch (_mode)
            {
                case EyeProcessingPipeline.EyelidMode.Both:
                    expressions[2] = rightLidValue;
                    expressions[5] = leftLidValue;
                    break;

                case EyeProcessingPipeline.EyelidMode.LeftOnly:
                    // Mirror left eye to both sides
                    expressions[2] = leftLidValue;
                    expressions[5] = leftLidValue;
                    break;

                case EyeProcessingPipeline.EyelidMode.RightOnly:
                    // Mirror right eye to both sides
                    expressions[2] = rightLidValue;
                    expressions[5] = rightLidValue;
                    break;
            }

            LogLowFreq(
                "EyeStateCls write mode={Mode} leftLid={LLid:F2} rightLid={RLid:F2} (closed={C:F2} winkL={WL:F2} winkR={WR:F2})",
                _mode, leftLidValue, rightLidValue, closed, winkLeft, winkRight);
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Failed to run eye-state classifier eyelid inference");
        }
    }

    public Task<bool> CalibrateAsync(TimeSpan duration, CancellationToken cancellationToken)
    {
        // Classifier-based enhancer does not currently support calibration.
        return Task.FromResult(false);
    }

    public void Dispose()
    {
        _dualEyeFrame?.Dispose();
        _dualEyeFrame = null;
    }

    private static float[] Softmax(float[] values)
    {
        // Numerically stable softmax. If the input already looks like probabilities, this
        // still produces a valid distribution.
        float max = float.NegativeInfinity;
        for (var i = 0; i < values.Length; i++)
            max = Math.Max(max, values[i]);

        double sum = 0;
        var exps = new double[values.Length];
        for (var i = 0; i < values.Length; i++)
        {
            var e = Math.Exp(values[i] - max);
            exps[i] = e;
            sum += e;
        }

        if (sum <= double.Epsilon)
            return new float[values.Length];

        var result = new float[values.Length];
        for (var i = 0; i < values.Length; i++)
            result[i] = (float)(exps[i] / sum);

        return result;
    }
}
