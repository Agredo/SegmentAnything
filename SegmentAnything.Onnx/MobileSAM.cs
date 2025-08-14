using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

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
    /// <param name="points">Array of prompt points. For MobileSam 2 Points are mandatory!</param>s
    /// <param name="labels">Array of point labels (1 = positive, 0 = negative).</param>
    /// <param name="boundingBox">Optional bounding box prompt.</param>
    /// <returns>Segmentation results containing masks and confidence scores.</returns>
    /// <exception cref="ArgumentException">Thrown when points and labels arrays have different lengths.</exception>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    public override SAMResult Segment(SKBitmap image, SKPointI[] points, int[] labels, SKRectI? boundingBox = null)
    {
        if (image == null) throw new ArgumentNullException(nameof(image));
        if (points == null) throw new ArgumentNullException(nameof(points));
        if (labels == null) throw new ArgumentNullException(nameof(labels));
        if (points.Length != labels.Length)
            throw new ArgumentException("Points and labels must have the same length");

        // 1. Image Encoding
        var imageFeatures = EncodeImage(image);

        // 2. Prompt Encoding und Decoding
        return DecodeWithPrompts(imageFeatures, points, labels, boundingBox, image.Width, image.Height);
    }

    /// <summary>
    /// Encodes the input image into feature embeddings using the MobileSAM encoder model.
    /// The image is preprocessed to the required model input format (1024x1024) and passed through
    /// the encoder network to generate high-dimensional feature representations.
    /// </summary>
    /// <param name="image">The input bitmap image to encode. Must not be null.</param>
    /// <returns>
    /// A dense tensor containing the encoded image features with dimensions matching the encoder output.
    /// These features are used as input for the decoder during the segmentation process.
    /// </returns>
    /// <exception cref="ArgumentNullException">Thrown when the image parameter is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the encoder session is not properly initialized.</exception>
    /// <remarks>
    /// This method performs the following steps:
    /// 1. Preprocesses the input image to match model requirements (normalization, resizing)
    /// 2. Creates a tensor from the preprocessed image data
    /// 3. Runs inference through the encoder model
    /// 4. Copies the resulting features to ensure memory safety
    /// 
    /// The returned features must be used immediately or stored appropriately as they represent
    /// the encoded representation of the input image required for segmentation.
    /// </remarks>
    private DenseTensor<float> EncodeImage(SKBitmap image)
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


    private SAMResult DecodeWithPrompts(DenseTensor<float> imageFeatures, SKPointI[] points, int[] labels, SKRectI? boundingBox, int originalWidth, int originalHeight)
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

        var pointCoordsFlat = new float[points.Length * 2];
        for (int i = 0; i < points.Length; i++)
        {
            pointCoordsFlat[i * 2] = points[i].X * scaleX;
            pointCoordsFlat[i * 2 + 1] = points[i].Y * scaleY;
        }
        var coordsTensor = new DenseTensor<float>(pointCoordsFlat, new[] { 1, points.Length, 2 });

        //var coordsTensor = new DenseTensor<float>(pointCoords.Cast<float>().ToArray(), new[] { 1, points.Length, 2 });
        var labelsTensor = new DenseTensor<float>(pointLabels.Cast<float>().ToArray(), new[] { 1, points.Length });

        // Mask input (leer für ersten Durchlauf)
        var maskInput = new DenseTensor<float>(new[] { 1, 1, 256, 256 });
        var hasMaskInput = new DenseTensor<float>(new float[] { 0 }, new[] { 1 });

        var decoderInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image_embeddings", imageFeatures),
            NamedOnnxValue.CreateFromTensor("point_coords", coordsTensor),
            NamedOnnxValue.CreateFromTensor("point_labels", labelsTensor),
            //NamedOnnxValue.CreateFromTensor("mask_input", maskInput),
            //NamedOnnxValue.CreateFromTensor("has_mask_input", hasMaskInput),
            //NamedOnnxValue.CreateFromTensor("orig_im_size", new DenseTensor<int>(new int[] { originalHeight, originalWidth }, new[] { 2 }))
        };

        using var decoderResults = _decoderSession.Run(decoderInputs);

        var masks = decoderResults.First(x => x.Name == "masks").AsTensor<float>();
        var scores = decoderResults.First(x => x.Name == "masks").AsTensor<float>();

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
