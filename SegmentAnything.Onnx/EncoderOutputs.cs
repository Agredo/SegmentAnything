using Microsoft.ML.OnnxRuntime.Tensors;

namespace SegmentAnything.Onnx;

/// <summary>
/// Container for encoder outputs from SAM models.
/// Stores image embeddings and high-resolution features used by the decoder.
/// </summary>
public class EncoderOutputs
{
    /// <summary>
    /// Gets or sets the main image embedding tensor.
    /// Contains the compressed image representation used for segmentation.
    /// </summary>
    public DenseTensor<float> ImageEmbed { get; set; }

    /// <summary>
    /// Gets or sets the first high-resolution feature tensor.
    /// Contains detailed feature information for fine-grained segmentation.
    /// </summary>
    public DenseTensor<float> HighResFeats0 { get; set; }

    /// <summary>
    /// Gets or sets the second high-resolution feature tensor.
    /// Contains additional detailed feature information for fine-grained segmentation.
    /// </summary>
    public DenseTensor<float> HighResFeats1 { get; set; }
}
