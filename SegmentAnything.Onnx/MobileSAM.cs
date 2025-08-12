using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing;

namespace SegmentAnything.Onnx;

/// <summary>
/// MobileSAM implementation optimized for mobile and resource-constrained environments.
/// Provides fast image segmentation with reduced model size and computational requirements.
/// </summary>
public class MobileSAM : SAMModelBase
{
    /// <summary>
    /// Initializes a new instance of the MobileSAM class.
    /// </summary>
    /// <param name="encoderModelPath">Path to the MobileSAM encoder ONNX model file.</param>
    /// <param name="decoderModelPath">Path to the MobileSAM decoder ONNX model file.</param>
    /// <exception cref="FileNotFoundException">Thrown when model files are not found.</exception>
    /// <exception cref="ArgumentException">Thrown when model files are invalid.</exception>
    public MobileSAM(string encoderModelPath, string decoderModelPath)
        : base(encoderModelPath, decoderModelPath)
    {
    }

    /// <summary>
    /// Performs image segmentation using point and/or bounding box prompts.
    /// Optimized for fast inference on mobile devices and edge computing scenarios.
    /// </summary>
    /// <param name="image">The input image to segment.</param>
    /// <param name="points">Array of prompt points.</param>
    /// <param name="labels">Array of point labels (1 = positive, 0 = negative).</param>
    /// <param name="boundingBox">Optional bounding box prompt.</param>
    /// <returns>Segmentation results containing masks and confidence scores.</returns>
    /// <exception cref="ArgumentException">Thrown when points and labels arrays have different lengths.</exception>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    public override SAMResult Segment(Bitmap image, Point[] points, int[] labels, Rectangle? boundingBox = null)
    {
        if (points.Length != labels.Length)
            throw new ArgumentException("Points and labels must have the same length");

        // 1. Image Encoding
        var imageFeatures = EncodeImage(image);

        // 2. Prompt Encoding und Decoding
        return DecodeWithPrompts(imageFeatures, points, labels, boundingBox, image.Width, image.Height);
    }

    private DenseTensor<float> EncodeImage(Bitmap image)
    {
        var imageData = PreprocessImage(image);
        var imageTensor = new DenseTensor<float>(imageData, new[] { 1, 3, ImageSize, ImageSize });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image", imageTensor)
        };

        using var results = _encoderSession.Run(inputs);
        var features = results.First().AsTensor<float>();

        // Features kopieren für späteren Gebrauch
        var featuresCopy = new DenseTensor<float>(features.Dimensions);
        features.ToArray().CopyTo(featuresCopy.Buffer);

        return featuresCopy;
    }

    private SAMResult DecodeWithPrompts(DenseTensor<float> imageFeatures, Point[] points, int[] labels, Rectangle? boundingBox, int originalWidth, int originalHeight)
    {
        // Skalierung der Koordinaten auf das Modell-Format (1024x1024)
        float scaleX = (float)ImageSize / originalWidth;
        float scaleY = (float)ImageSize / originalHeight;

        // Prompts vorbereiten
        var pointCoords = new float[1, points.Length, 2];
        var pointLabels = new float[1, points.Length];

        for (int i = 0; i < points.Length; i++)
        {
            pointCoords[0, i, 0] = points[i].X * scaleX;
            pointCoords[0, i, 1] = points[i].Y * scaleY;
            pointLabels[0, i] = labels[i];
        }

        var coordsTensor = new DenseTensor<float>(pointCoords.Cast<float>().ToArray(), new[] { 1, points.Length, 2 });
        var labelsTensor = new DenseTensor<float>(pointLabels.Cast<float>().ToArray(), new[] { 1, points.Length });

        // Mask input (leer für ersten Durchlauf)
        var maskInput = new DenseTensor<float>(new[] { 1, 1, 256, 256 });
        var hasMaskInput = new DenseTensor<float>(new float[] { 0 }, new[] { 1 });

        var decoderInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image_embeddings", imageFeatures),
            NamedOnnxValue.CreateFromTensor("point_coords", coordsTensor),
            NamedOnnxValue.CreateFromTensor("point_labels", labelsTensor),
            NamedOnnxValue.CreateFromTensor("mask_input", maskInput),
            NamedOnnxValue.CreateFromTensor("has_mask_input", hasMaskInput),
            //NamedOnnxValue.CreateFromTensor("orig_im_size", new DenseTensor<int>(new int[] { originalHeight, originalWidth }, new[] { 2 }))
        };

        using var decoderResults = _decoderSession.Run(decoderInputs);

        var masks = decoderResults.First(x => x.Name == "masks").AsTensor<float>();
        var scores = decoderResults.First(x => x.Name == "iou_predictions").AsTensor<float>();

        return new SAMResult
        {
            Masks = ConvertTensorToMasks(masks),
            Scores = scores.ToArray(),
            OriginalWidth = originalWidth,
            OriginalHeight = originalHeight
        };
    }

    private float[][,] ConvertTensorToMasks(Tensor<float> maskTensor)
    {
        var dimensions = maskTensor.Dimensions.ToArray();
        int numMasks = dimensions[1];
        int height = dimensions[2];
        int width = dimensions[3];

        var masks = new float[numMasks][,];

        for (int m = 0; m < numMasks; m++)
        {
            masks[m] = new float[height, width];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    masks[m][y, x] = maskTensor[0, m, y, x];
                }
            }
        }

        return masks;
    }
}
