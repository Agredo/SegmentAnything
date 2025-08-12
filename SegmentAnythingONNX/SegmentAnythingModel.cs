using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Linq;

namespace SegmentAnythingONNX;

public abstract class SegmentAnythingModel : IDisposable
{
    protected readonly InferenceSession encoderSession;
    protected readonly InferenceSession decoderSession;

    protected SegmentAnythingModel(string encoderPath, string decoderPath, SessionOptions opts = null)
    {
        if (!File.Exists(encoderPath))
            throw new FileNotFoundException($"Encoder ONNX nicht gefunden: {encoderPath}");
        if (!File.Exists(decoderPath))
            throw new FileNotFoundException($"Decoder ONNX nicht gefunden: {decoderPath}");

        encoderSession = opts == null
            ? new InferenceSession(encoderPath)
            : new InferenceSession(encoderPath, opts);

        decoderSession = opts == null
            ? new InferenceSession(decoderPath)
            : new InferenceSession(decoderPath, opts);
    }

    public (Tensor<float> imageEmbed, Tensor<float> highRes0, Tensor<float> highRes1) EncodeImageSAM2(float[] imageData, int[] dims)
    {
        var inputName = encoderSession.InputMetadata.Keys.First();
        var tensor = new DenseTensor<float>(imageData, dims);

        var input = NamedOnnxValue.CreateFromTensor(inputName, tensor);
        using var results = encoderSession.Run(new[] { input });

        var dict = results.ToDictionary(r => r.Name, r => r.AsTensor<float>());

        return (
            dict["image_embed"],
            dict["high_res_feats_0"],
            dict["high_res_feats_1"]
        );
    }

    public Tensor<float> EncodeImageAuto(float[] imageData, int[] dims)
    {
        var inputName = encoderSession.InputMetadata.Keys.First();
        var tensor = new DenseTensor<float>(imageData, dims);

        var input = NamedOnnxValue.CreateFromTensor(inputName, tensor);
        using var results = encoderSession.Run(new[] { input });

        // Alle Outputs prüfen
        foreach (var r in results)
        {
            var t = r.AsTensor<float>();
            var shape = t.Dimensions.ToArray();

            // SAM-Decoder erwartet in der Regel [1,256,64,64]
            if (shape.Length == 4 && shape[0] == 1 && shape[1] == 256 && shape[2] == 64 && shape[3] == 64)
            {
                Console.WriteLine($"Encoder-Output '{r.Name}' wird als image_embed verwendet.");
                return t;
            }
        }

        // Falls kein passender Output gefunden → Fehler + Debug-Info
        var msg = "Kein passender Encoder-Output gefunden. Verfügbare Shapes:\n";
        foreach (var r in results)
        {
            var t = r.AsTensor<float>();
            msg += $"- {r.Name}: [{string.Join(",", t.Dimensions.ToArray())}]\n";
        }
        throw new InvalidOperationException(msg);
    }

    public Tensor<float> DecodeMaskSAM2(
        Tensor<float> imageEmbed,
        Tensor<float> highRes0,
        Tensor<float> highRes1,
        float[] pointCoords,
        int[] pointLabels,      // jetzt int[]
        int origHeight,
        int origWidth)
    {
        // point_labels als int[] definieren, z.B. 1 für "Klickpunkt positiv"
        int[] pointLabelsInt = new int[] { 1 };

        var pointLabelsTensor = new DenseTensor<float>(pointLabelsInt.Select(x => (float)x).ToArray(), new[] { 1, pointLabelsInt.Length });

        // has_mask_input ebenfalls int Tensor (0 oder 1)
        var hasMaskInputTensor = new DenseTensor<int>(new int[] { 0 }, new[] { 1 });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image_embed", imageEmbed),
            NamedOnnxValue.CreateFromTensor("high_res_feats_0", highRes0),
            NamedOnnxValue.CreateFromTensor("high_res_feats_1", highRes1),
            NamedOnnxValue.CreateFromTensor("point_coords", new DenseTensor<float>(pointCoords, new[] { 1, pointCoords.Length / 2, 2 })),
            NamedOnnxValue.CreateFromTensor("point_labels", pointLabelsTensor),
            NamedOnnxValue.CreateFromTensor("mask_input", new DenseTensor<float>(new float[1 * 1 * 256 * 256], new[] { 1, 1, 256, 256 })),
            NamedOnnxValue.CreateFromTensor("has_mask_input", hasMaskInputTensor),
            NamedOnnxValue.CreateFromTensor("orig_im_size", new DenseTensor<float>(new float[] { origHeight, origWidth }, new[] { 2 }))
        };

        //var coordsTensor = new DenseTensor<float>(pointCoords.Cast<float>().ToArray(), new[] { 1, pointCoords.GetLength(1), 2 });
        //var labelsTensor = new DenseTensor<float>(pointLabels.Cast<float>().ToArray(), new[] { 1, pointLabels.GetLength(1) });

        var decoderInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image_embed", imageEmbed),
                NamedOnnxValue.CreateFromTensor("point_coords", new DenseTensor<float>(pointCoords, new[] { 1, pointCoords.Length / 2, 2 })),
                NamedOnnxValue.CreateFromTensor("point_labels", pointLabelsTensor),
                NamedOnnxValue.CreateFromTensor("high_res_feats_0", highRes0),
                NamedOnnxValue.CreateFromTensor("high_res_feats_1", highRes1),
                NamedOnnxValue.CreateFromTensor("has_mask_input", new DenseTensor<float>(new float[] { 0 }, new[] { 1 })),
                NamedOnnxValue.CreateFromTensor("orig_im_size", new DenseTensor<int>(new int[] { origHeight, origWidth }, new[] { 2 })),
                NamedOnnxValue.CreateFromTensor("mask_input", new DenseTensor<float>(new float[1 * 1 * 256 * 256], new[] { 1, 1, 256, 256 })),
            };

        foreach (var input in decoderInputs)
        {
            Console.WriteLine($"Input: {input.Name}");
            Console.WriteLine($" - Tensor-Typ: {input.Value.GetType()}");

            if (input.Value is DenseTensor<float> floatTensor)
                Console.WriteLine($" - Float Tensor mit Shape: {string.Join(", ", floatTensor.Dimensions.ToArray())}");
            else if (input.Value is DenseTensor<int> intTensor)
                Console.WriteLine($" - Int Tensor mit Shape: {string.Join(", ", intTensor.Dimensions.ToArray())}");
            else
                Console.WriteLine($" - Anderer Tensor-Typ");
        }

        using var results = decoderSession.Run(decoderInputs);
        return results.First().AsTensor<float>();
    }


    public Tensor<float> DecodeMask(float[] embeddings, int[] embedDims, float[] promptData, int[] promptDims)
    {
        var inputNames = decoderSession.InputMetadata.Keys.ToList();
        var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputNames[0], new DenseTensor<float>(embeddings, embedDims)),
                NamedOnnxValue.CreateFromTensor(inputNames[1], new DenseTensor<float>(promptData, promptDims))
            };

        using var results = decoderSession.Run(inputs);
        return results.First().AsTensor<float>();
    }

    public void Dispose()
    {
        encoderSession?.Dispose();
        decoderSession?.Dispose();
    }
}

public class MobileSAMModel : SegmentAnythingModel
{
    public MobileSAMModel(string encoderPath, string decoderPath, SessionOptions opts = null)
        : base(encoderPath, decoderPath, opts) { }
}

public class SAM2Model : SegmentAnythingModel
{
    public SAM2Model(string encoderPath, string decoderPath, SessionOptions opts = null)
        : base(encoderPath, decoderPath, opts) { }

    public Tensor<float> RunSAM2(string imagePath, float[] pointCoords, int[] pointLabels, int targetSize = 1024)
    {
        using var bmp = new Bitmap(imagePath);
        int origH = bmp.Height;
        int origW = bmp.Width;

        float[] imageData = ImagePreprocessor.LoadAndPreprocessImage(imagePath, targetSize);
        int[] dims = { 1, 3, targetSize, targetSize };

        var (imageEmbed, highRes0, highRes1) = EncodeImageSAM2(imageData, dims);

        return DecodeMaskSAM2(imageEmbed, highRes0, highRes1, pointCoords, pointLabels, origH, origW);
    }
}

public static class ImagePreprocessor
{
    // SAM-Standard: Normalisierungswerte (ImageNet mean/std)
    private static readonly float[] mean = { 123.675f, 116.28f, 103.53f };
    private static readonly float[] std = { 58.395f, 57.12f, 57.375f };

    /// <summary>
    /// Lädt ein Bild, resized auf Zielgröße, normalisiert es und gibt als Tensor-Array (CHW) zurück.
    /// </summary>
    public static float[] LoadAndPreprocessImage(string path, int targetSize = 1024)
    {
        using var bmp = new Bitmap(path);
        using var resized = ResizeImage(bmp, targetSize, targetSize);

        var data = new float[3 * targetSize * targetSize]; // CHW
        int idxR = 0;
        int idxG = targetSize * targetSize;
        int idxB = targetSize * targetSize * 2;

        for (int y = 0; y < targetSize; y++)
        {
            for (int x = 0; x < targetSize; x++)
            {
                var color = resized.GetPixel(x, y);
                data[idxR++] = ((float)color.R - mean[0]) / std[0];
                data[idxG++] = ((float)color.G - mean[1]) / std[1];
                data[idxB++] = ((float)color.B - mean[2]) / std[2];
            }
        }

        return data;
    }

    private static Bitmap ResizeImage(Bitmap img, int width, int height)
    {
        var destRect = new Rectangle(0, 0, width, height);
        var destImage = new Bitmap(width, height, PixelFormat.Format24bppRgb);

        destImage.SetResolution(img.HorizontalResolution, img.VerticalResolution);

        using (var graphics = Graphics.FromImage(destImage))
        {
            graphics.CompositingMode = CompositingMode.SourceCopy;
            graphics.CompositingQuality = CompositingQuality.HighQuality;
            graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
            graphics.SmoothingMode = SmoothingMode.HighQuality;
            graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

            using var wrapMode = new ImageAttributes();
            wrapMode.SetWrapMode(WrapMode.TileFlipXY);
            graphics.DrawImage(img, destRect, 0, 0, img.Width, img.Height, GraphicsUnit.Pixel, wrapMode);
        }

        return destImage;
    }
}

