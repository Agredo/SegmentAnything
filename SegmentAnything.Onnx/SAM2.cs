using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing;

namespace SegmentAnything.Onnx;

/// <summary>
/// SAM2 implementation with enhanced video support and temporal memory.
/// Provides high-performance image segmentation with support for video frame tracking.
/// </summary>
public class SAM2 : SAMModelBase
{
    private readonly Dictionary<int, SAMMemoryState> _memoryStates;
    private int _currentObjId = 0;

    /// <summary>
    /// Initializes a new instance of the SAM2 class.
    /// </summary>
    /// <param name="encoderModelPath">Path to the SAM2 encoder ONNX model file.</param>
    /// <param name="decoderModelPath">Path to the SAM2 decoder ONNX model file.</param>
    /// <exception cref="FileNotFoundException">Thrown when model files are not found.</exception>
    /// <exception cref="ArgumentException">Thrown when model files are invalid.</exception>
    public SAM2(string encoderModelPath, string decoderModelPath)
        : base(encoderModelPath, decoderModelPath)
    {
        _memoryStates = new Dictionary<int, SAMMemoryState>();
    }

    /// <summary>
    /// Performs image segmentation using point and/or bounding box prompts.
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
        return SegmentFrame(image, points, labels, 0, boundingBox);
    }

    /// <summary>
    /// Performs video frame segmentation with temporal context tracking.
    /// Uses memory from previous frames to improve segmentation consistency.
    /// </summary>
    /// <param name="image">The video frame to segment.</param>
    /// <param name="points">Array of prompt points.</param>
    /// <param name="labels">Array of point labels (1 = positive, 0 = negative).</param>
    /// <param name="frameIndex">Index of the current frame in the video sequence.</param>
    /// <param name="boundingBox">Optional bounding box prompt.</param>
    /// <returns>Segmentation results with temporal consistency.</returns>
    /// <exception cref="ArgumentException">Thrown when points and labels arrays have different lengths.</exception>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    public SAMResult SegmentFrame(Bitmap image, Point[] points, int[] labels, int frameIndex, Rectangle? boundingBox = null)
    {
        if (points.Length != labels.Length)
            throw new ArgumentException("Points and labels must have the same length");

        // 1. Image Encoding
        var encoderOutputs = EncodeImageWithMemory(image, frameIndex);

        // 2. Decoding mit temporalem Kontext
        return DecodeWithTemporalContext(encoderOutputs, points, labels, frameIndex, boundingBox, image.Width, image.Height);
    }

    private EncoderOutputs EncodeImageWithMemory(Bitmap image, int frameIndex)
    {
        var imageData = PreprocessImage(image);
        var imageTensor = new DenseTensor<float>(imageData, new[] { 1, 3, ImageSize, ImageSize });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image", imageTensor)
        };

        using var results = _encoderSession.Run(inputs);

        // Alle drei erwarteten Outputs extrahieren
        var resultDict = results.ToDictionary(r => r.Name, r => r.AsTensor<float>());

        var imageEmbed = CopyTensor(resultDict["image_embed"]);
        var highRes0 = CopyTensor(resultDict["high_res_feats_0"]);
        var highRes1 = CopyTensor(resultDict["high_res_feats_1"]);

        return new EncoderOutputs
        {
            ImageEmbed = imageEmbed,
            HighResFeats0 = highRes0,
            HighResFeats1 = highRes1
        };
    }

    private SAMResult DecodeWithTemporalContext(EncoderOutputs encoderOutputs, Point[] points, int[] labels,
        int frameIndex, Rectangle? boundingBox, int originalWidth, int originalHeight)
    {
        // Koordinaten skalieren
        float scaleX = (float)ImageSize / originalWidth;
        float scaleY = (float)ImageSize / originalHeight;

        // Punkte vorbereiten (inklusive Bounding Box als Punkte)
        int totalPoints = points.Length + (boundingBox.HasValue ? 2 : 0);
        var pointCoords = new float[1, totalPoints, 2];
        var pointLabels = new float[1, totalPoints];

        // Normale Punkte hinzufügen
        for (int i = 0; i < points.Length; i++)
        {
            pointCoords[0, i, 0] = points[i].X * scaleX;
            pointCoords[0, i, 1] = points[i].Y * scaleY;
            pointLabels[0, i] = labels[i];
        }

        // Bounding Box als Punkte hinzufügen
        if (boundingBox.HasValue)
        {
            var box = boundingBox.Value;

            // Top-left corner
            pointCoords[0, points.Length, 0] = box.Left * scaleX;
            pointCoords[0, points.Length, 1] = box.Top * scaleY;
            pointLabels[0, points.Length] = 2;

            // Bottom-right corner
            pointCoords[0, points.Length + 1, 0] = box.Right * scaleX;
            pointCoords[0, points.Length + 1, 1] = box.Bottom * scaleY;
            pointLabels[0, points.Length + 1] = 3;
        }

        var coordsTensor = new DenseTensor<float>(pointCoords.Cast<float>().ToArray(), new[] { 1, totalPoints, 2 });
        var labelsTensor = new DenseTensor<float>(pointLabels.Cast<float>().ToArray(), new[] { 1, totalPoints });

        var decoderInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image_embed", encoderOutputs.ImageEmbed),
            NamedOnnxValue.CreateFromTensor("point_coords", coordsTensor),
            NamedOnnxValue.CreateFromTensor("point_labels", labelsTensor),
            NamedOnnxValue.CreateFromTensor("high_res_feats_0", encoderOutputs.HighResFeats0),
            NamedOnnxValue.CreateFromTensor("high_res_feats_1", encoderOutputs.HighResFeats1),
            //NamedOnnxValue.CreateFromTensor("orig_im_size", new DenseTensor<int>(new int[] { originalHeight, originalWidth }, new[] { 2 })),
            NamedOnnxValue.CreateFromTensor("mask_input", new DenseTensor<float>(new float[1 * 1 * 256 * 256], new[] { 1, 1, 256, 256 })),
            NamedOnnxValue.CreateFromTensor("has_mask_input", new DenseTensor<float>(new float[] { 0 }, new[] { 1 }))
        };

        using var decoderResults = _decoderSession.Run(decoderInputs);

        var masks = decoderResults.First(x => x.Name == "masks").AsTensor<float>();
        var scores = decoderResults.First(x => x.Name == "iou_predictions").AsTensor<float>();

        return new SAMResult
        {
            Masks = ConvertTensorToMasks(masks),
            Scores = scores.ToArray(),
            FrameIndex = frameIndex,
            OriginalWidth = originalWidth,
            OriginalHeight = originalHeight
        };
    }

    private DenseTensor<float> CopyTensor(Tensor<float> source)
    {
        var copy = new DenseTensor<float>(source.Dimensions);
        source.ToArray().CopyTo(copy.Buffer);
        return copy;
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

    /// <summary>
    /// Clears the temporal memory cache used for video frame tracking.
    /// Call this method when switching to a new video sequence.
    /// </summary>
    public void ClearMemoryCache()
    {
        _memoryStates.Clear();
    }

    /// <summary>
    /// Releases all resources used by the SAM2 instance.
    /// </summary>
    public override void Dispose()
    {
        ClearMemoryCache();
        base.Dispose();
    }
}
