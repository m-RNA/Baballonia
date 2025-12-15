using System;
using System.Threading;
using System.Threading.Tasks;
using OpenCvSharp;

namespace Baballonia.Services.Inference;

/// <summary>
/// Describes components that can improve eyelid estimates with extra inference passes.
/// </summary>
public interface IEyelidEnhancer : IDisposable
{
    /// <summary>
    /// Captures the latest transformed dual-eye frame to prepare inputs for an eyelid specific model.
    /// </summary>
    /// <param name="transformedEyes">Dual-eye image produced by <see cref="IImageTransformer"/>.</param>
    void CaptureEyeImages(Mat transformedEyes);

    /// <summary>
    /// Adjusts the raw inference outputs in-place.
    /// </summary>
    /// <param name="expressions">Array returned by the primary eye model.</param>
    void Enhance(float[] expressions);

    /// <summary>
    /// Collects dedicated eyelid samples for the provided duration and updates internal thresholds.
    /// </summary>
    /// <param name="duration">How long to gather samples.</param>
    /// <param name="cancellationToken">Cancels the calibration.</param>
    Task<bool> CalibrateAsync(TimeSpan duration, CancellationToken cancellationToken);
}
