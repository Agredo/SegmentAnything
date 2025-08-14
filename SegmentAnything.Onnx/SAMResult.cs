using SkiaSharp;

namespace SegmentAnything.Onnx;

/// <summary>
/// Container for segmentation results from SAM models.
/// Contains masks, confidence scores, and utility methods for mask processing.
/// </summary>
public class SAMResult
{
    /// <summary>
    /// Gets or sets the generated segmentation masks.
    /// Each mask is a 2D array of logit values that can be converted to binary masks.
    /// </summary>
    public float[][,] Masks { get; set; }

    /// <summary>
    /// Gets or sets the confidence scores for each mask.
    /// Higher scores indicate better quality segmentation.
    /// </summary>
    public float[] Scores { get; set; }

    /// <summary>
    /// Gets or sets the frame index for video sequences (SAM2 only).
    /// Default is 0 for single image segmentation.
    /// </summary>
    public int FrameIndex { get; set; } = 0;

    /// <summary>
    /// Gets or sets the original image width before preprocessing.
    /// </summary>
    public int OriginalWidth { get; set; }

    /// <summary>
    /// Gets or sets the original image height before preprocessing.
    /// </summary>
    public int OriginalHeight { get; set; }

    /// <summary>
    /// Gets the best mask as a bitmap based on confidence scores.
    /// The mask is resized to the specified dimensions or original image size.
    /// </summary>
    /// <param name="width">Target width for the output bitmap. Uses OriginalWidth if null.</param>
    /// <param name="height">Target height for the output bitmap. Uses OriginalHeight if null.</param>
    /// <returns>A bitmap representation of the best mask, or null if no masks are available.</returns>
    public SKBitmap GetBestMaskAsBitmap(int? width = null, int? height = null)
    {
        if (Masks == null || Masks.Length == 0)
            return null;

        int targetWidth = width ?? OriginalWidth;
        int targetHeight = height ?? OriginalHeight;

        // Beste Maske basierend auf Score auswählen
        int bestMaskIndex = 0;
        if (Scores != null && Scores.Length > 0)
        {
            for (int i = 1; i < Scores.Length; i++)
            {
                if (Scores[i] > Scores[bestMaskIndex])
                    bestMaskIndex = i;
            }
        }

        return MaskToBitmapCorrected(Masks[bestMaskIndex], targetWidth, targetHeight);
    }

    /// <summary>
    /// Gets all masks as bitmap arrays.
    /// Each mask is converted to a bitmap and resized to the specified dimensions.
    /// </summary>
    /// <param name="width">Target width for the output bitmaps. Uses OriginalWidth if null.</param>
    /// <param name="height">Target height for the output bitmaps. Uses OriginalHeight if null.</param>
    /// <returns>An array of bitmap representations for all masks.</returns>
    public SKBitmap[] GetAllMasksAsBitmaps(int? width = null, int? height = null)
    {
        if (Masks == null)
            return Array.Empty<SKBitmap>();

        int targetWidth = width ?? OriginalWidth;
        int targetHeight = height ?? OriginalHeight;

        var bitmaps = new SKBitmap[Masks.Length];
        for (int i = 0; i < Masks.Length; i++)
        {
            bitmaps[i] = MaskToBitmapCorrected(Masks[i], targetWidth, targetHeight);
        }
        return bitmaps;
    }

    private SKBitmap MaskToBitmapCorrected(float[,] mask, int targetWidth, int targetHeight)
    {
        int maskHeight = mask.GetLength(0);
        int maskWidth = mask.GetLength(1);

        var bitmap = new SKBitmap(targetWidth, targetHeight, SKColorType.Rgba8888, SKAlphaType.Opaque);
        using var pixmap = bitmap.PeekPixels();

        unsafe
        {
            byte* ptr = (byte*)pixmap.GetPixels();
            int stride = pixmap.RowBytes;
            for (int y = 0; y < targetHeight; y++)
            {
                for (int x = 0; x < targetWidth; x++)
                {
                    // Korrekte Skalierung von Target zu Mask Koordinaten
                    int maskX = Math.Min((int)((float)x / targetWidth * maskWidth), maskWidth - 1);
                    int maskY = Math.Min((int)((float)y / targetHeight * maskHeight), maskHeight - 1);

                    float logit = mask[maskY, maskX];
                    float probability = 1.0f / (1.0f + MathF.Exp(-logit));
                    byte intensity = probability > 0.5f ? (byte)255 : (byte)0;

                    int pixelIndex = y * stride + x * 4; // RGBA
                    ptr[pixelIndex + 0] = intensity; // R
                    ptr[pixelIndex + 1] = intensity; // G
                    ptr[pixelIndex + 2] = intensity; // B
                    ptr[pixelIndex + 3] = 255;       // A
                }
            }
        }

        return bitmap;
    }

    private SKBitmap MaskToBitmap(float[,] mask, int targetWidth, int targetHeight)
    {
        int maskHeight = mask.GetLength(0);
        int maskWidth = mask.GetLength(1);

        var bitmap = new SKBitmap(targetWidth, targetHeight, SKColorType.Rgba8888, SKAlphaType.Opaque);
        using var pixmap = bitmap.PeekPixels();

        unsafe
        {
            byte* ptr = (byte*)pixmap.GetPixels();
            int stride = pixmap.RowBytes;
            for (int y = 0; y < targetHeight; y++)
            {
                for (int x = 0; x < targetWidth; x++)
                {
                    int maskX = Math.Min((int)((float)x / targetWidth * maskWidth), maskWidth - 1);
                    int maskY = Math.Min((int)((float)y / targetHeight * maskHeight), maskHeight - 1);

                    float logit = mask[maskY, maskX];

                    // Sigmoid-Funktion anwenden für Wahrscheinlichkeit
                    float probability = 1.0f / (1.0f + MathF.Exp(-logit));

                    // Binäre Maske: Schwellenwert bei 0.5
                    byte intensity = probability > 0.5f ? (byte)255 : (byte)0;

                    int pixelIndex = y * stride + x * 4; // RGBA

                    // RBG Format für 32bpp
                    ptr[pixelIndex + 0] = intensity;
                    ptr[pixelIndex + 1] = intensity;
                    ptr[pixelIndex + 2] = intensity;
                    ptr[pixelIndex + 3] = 255;
                }
            }
        }
        return bitmap;
    }
}
