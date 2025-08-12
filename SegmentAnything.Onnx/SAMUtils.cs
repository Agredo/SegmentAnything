using System.Drawing;
using System.Drawing.Imaging;

namespace SegmentAnything.Onnx;

/// <summary>
/// Utility class for common Segment Anything Model operations.
/// Provides helper methods for point generation, label creation, and mask processing.
/// </summary>
public static class SAMUtils
{
    /// <summary>
    /// Creates a grid of evenly spaced points for automatic segmentation.
    /// Useful for generating multiple prompt points across an image.
    /// </summary>
    /// <param name="width">Width of the target area.</param>
    /// <param name="height">Height of the target area.</param>
    /// <param name="gridSize">Number of points per row/column in the grid.</param>
    /// <returns>An array of points arranged in a grid pattern.</returns>
    public static Point[] CreatePointGrid(int width, int height, int gridSize)
    {
        var points = new List<Point>();
        int stepX = width / gridSize;
        int stepY = height / gridSize;

        for (int y = stepY / 2; y < height; y += stepY)
        {
            for (int x = stepX / 2; x < width; x += stepX)
            {
                points.Add(new Point(x, y));
            }
        }

        return points.ToArray();
    }

    /// <summary>
    /// Creates an array of positive labels (value = 1) for the specified number of points.
    /// Positive points indicate areas that should be included in the segmentation.
    /// </summary>
    /// <param name="count">Number of positive labels to create.</param>
    /// <returns>An array of positive labels.</returns>
    public static int[] CreatePositiveLabels(int count)
    {
        return Enumerable.Repeat(1, count).ToArray();
    }

    /// <summary>
    /// Creates an array of negative labels (value = 0) for the specified number of points.
    /// Negative points indicate areas that should be excluded from the segmentation.
    /// </summary>
    /// <param name="count">Number of negative labels to create.</param>
    /// <returns>An array of negative labels.</returns>
    public static int[] CreateNegativeLabels(int count)
    {
        return Enumerable.Repeat(0, count).ToArray();
    }

    /// <summary>
    /// Applies a colored mask overlay to the original image.
    /// Creates a visual representation of the segmentation results.
    /// </summary>
    /// <param name="originalImage">The original image to apply the mask to.</param>
    /// <param name="mask">The segmentation mask as a 2D float array of logit values.</param>
    /// <param name="maskColor">The color to use for the mask overlay.</param>
    /// <param name="alpha">The transparency of the mask overlay (0.0 = transparent, 1.0 = opaque).</param>
    /// <returns>A new bitmap with the mask applied as an overlay.</returns>
    /// <exception cref="ArgumentNullException">Thrown when originalImage or mask is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when alpha is not between 0.0 and 1.0.</exception>
    public static Bitmap ApplyMaskToImage(Bitmap originalImage, float[,] mask, Color maskColor, float alpha = 0.5f)
    {
        if (originalImage == null)
            throw new ArgumentNullException(nameof(originalImage));
        if (mask == null)
            throw new ArgumentNullException(nameof(mask));
        if (alpha < 0.0f || alpha > 1.0f)
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be between 0.0 and 1.0");

        // Direkter Klon für maximale Kompatibilität
        var result = (Bitmap)originalImage.Clone();

        int maskHeight = mask.GetLength(0);
        int maskWidth = mask.GetLength(1);

        // Lock das geklonte Bild im Original-Format
        var bitmapData = result.LockBits(
            new Rectangle(0, 0, result.Width, result.Height),
            ImageLockMode.ReadWrite,
            result.PixelFormat);

        try
        {
            unsafe
            {
                byte* ptr = (byte*)bitmapData.Scan0;
                int stride = bitmapData.Stride;
                int bytesPerPixel = Image.GetPixelFormatSize(result.PixelFormat) / 8;

                for (int y = 0; y < result.Height; y++)
                {
                    for (int x = 0; x < result.Width; x++)
                    {
                        // Korrekte Koordinaten-Transformation
                        int maskX = Math.Min((int)((float)x / result.Width * maskWidth), maskWidth - 1);
                        int maskY = Math.Min((int)((float)y / result.Height * maskHeight), maskHeight - 1);

                        float logit = mask[maskY, maskX];
                        float probability = 1.0f / (1.0f + MathF.Exp(-logit));

                        if (probability > 0.5f)
                        {
                            int pixelIndex = y * stride + x * bytesPerPixel;

                            // Blending mit Alpha (BGR/BGRA Format beachten)
                            ptr[pixelIndex] = (byte)((1 - alpha) * ptr[pixelIndex] + alpha * maskColor.B);         // B
                            if (bytesPerPixel > 1)
                                ptr[pixelIndex + 1] = (byte)((1 - alpha) * ptr[pixelIndex + 1] + alpha * maskColor.G); // G
                            if (bytesPerPixel > 2)
                                ptr[pixelIndex + 2] = (byte)((1 - alpha) * ptr[pixelIndex + 2] + alpha * maskColor.R); // R
                        }
                    }
                }
            }
        }
        finally
        {
            result.UnlockBits(bitmapData);
        }

        return result;
    }

    /// <summary>
    /// Converts a binary mask to a colored bitmap.
    /// Useful for visualizing segmentation results.
    /// </summary>
    /// <param name="mask">The binary mask as a 2D float array.</param>
    /// <param name="width">Target width for the output bitmap.</param>
    /// <param name="height">Target height for the output bitmap.</param>
    /// <param name="foregroundColor">Color for mask areas (probability > 0.5).</param>
    /// <param name="backgroundColor">Color for background areas (probability <= 0.5).</param>
    /// <returns>A bitmap representation of the mask.</returns>
    public static Bitmap MaskToBitmap(float[,] mask, int width, int height, Color? foregroundColor = null, Color? backgroundColor = null)
    {
        var fgColor = foregroundColor ?? Color.White;
        var bgColor = backgroundColor ?? Color.Black;

        int maskHeight = mask.GetLength(0);
        int maskWidth = mask.GetLength(1);

        var bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb);

        var bitmapData = bitmap.LockBits(
            new Rectangle(0, 0, width, height),
            ImageLockMode.WriteOnly,
            PixelFormat.Format24bppRgb);

        try
        {
            unsafe
            {
                byte* ptr = (byte*)bitmapData.Scan0;
                int stride = bitmapData.Stride;

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int maskX = Math.Min((int)((float)x / width * maskWidth), maskWidth - 1);
                        int maskY = Math.Min((int)((float)y / height * maskHeight), maskHeight - 1);

                        float logit = mask[maskY, maskX];
                        float probability = 1.0f / (1.0f + MathF.Exp(-logit));

                        var color = probability > 0.5f ? fgColor : bgColor;

                        int pixelIndex = y * stride + x * 3;
                        ptr[pixelIndex] = color.B;     // B
                        ptr[pixelIndex + 1] = color.G; // G
                        ptr[pixelIndex + 2] = color.R; // R
                    }
                }
            }
        }
        finally
        {
            bitmap.UnlockBits(bitmapData);
        }

        return bitmap;
    }

    private static Color[] GenerateColors(int count)
    {
        var colors = new Color[count];
        var random = new Random(42); // Seed für konsistente Farben

        for (int i = 0; i < count; i++)
        {
            colors[i] = Color.FromArgb(
                random.Next(100, 256),
                random.Next(100, 256),
                random.Next(100, 256)
            );
        }

        return colors;
    }
}