using SkiaSharp;
using System.Collections.Concurrent;

namespace SegmentAnything.Onnx;

public static class SAMUtils
{
    public static SKPointI[] CreatePointGrid(int width, int height, int gridSize)
    {
        var points = new List<SKPointI>();
        int stepX = width / gridSize;
        int stepY = height / gridSize;
        for (int y = stepY / 2; y < height; y += stepY)
        {
            for (int x = stepX / 2; x < width; x += stepX)
            {
                points.Add(new SKPointI(x, y));
            }
        }
        return points.ToArray();
    }

    public static int[] CreatePositiveLabels(int count) => Enumerable.Repeat(1, count).ToArray();
    public static int[] CreateNegativeLabels(int count) => Enumerable.Repeat(0, count).ToArray();

    public static SKBitmap ApplyMaskToImage(SKBitmap originalImage, float[,] mask, SKColor maskColor, float alpha = 0.5f)
    {
        if (originalImage == null) throw new ArgumentNullException(nameof(originalImage));
        if (mask == null) throw new ArgumentNullException(nameof(mask));
        if (alpha < 0f || alpha > 1f) throw new ArgumentOutOfRangeException(nameof(alpha));

        var result = originalImage.Copy();

        int maskHeight = mask.GetLength(0);
        int maskWidth = mask.GetLength(1);

        using var pixmap = result.PeekPixels();
        unsafe
        {
            byte* ptr = (byte*)pixmap.GetPixels();
            int stride = pixmap.RowBytes; // bytes per row
            for (int y = 0; y < result.Height; y++)
            {
                for (int x = 0; x < result.Width; x++)
                {
                    int maskX = Math.Min((int)((float)x / result.Width * maskWidth), maskWidth - 1);
                    int maskY = Math.Min((int)((float)y / result.Height * maskHeight), maskHeight - 1);

                    float logit = mask[maskY, maskX];
                    float probability = 1f / (1f + MathF.Exp(-logit));
                    if (probability > 0.5f)
                    {
                        int pixelIndex = y * stride + x * 4; // RGBA8888
                        byte r = ptr[pixelIndex + 0];
                        byte g = ptr[pixelIndex + 1];
                        byte b = ptr[pixelIndex + 2];
                        byte a = ptr[pixelIndex + 3];

                        ptr[pixelIndex + 0] = (byte)((1 - alpha) * r + alpha * maskColor.Red);
                        ptr[pixelIndex + 1] = (byte)((1 - alpha) * g + alpha * maskColor.Green);
                        ptr[pixelIndex + 2] = (byte)((1 - alpha) * b + alpha * maskColor.Blue);
                        ptr[pixelIndex + 3] = a; // keep alpha
                    }
                }
            }
        }
        return result;
    }

    /// <summary>
    /// Alternative ultra-fast version without pre-computing probabilities
    /// Good for when memory is a concern with very large masks
    /// </summary>
    public static SKBitmap ApplyMaskToImageFast(SKBitmap originalImage, float[,] mask, SKColor maskColor, float alpha = 0.5f)
    {
        if (originalImage == null) throw new ArgumentNullException(nameof(originalImage));
        if (mask == null) throw new ArgumentNullException(nameof(mask));
        if (alpha < 0f || alpha > 1f) throw new ArgumentOutOfRangeException(nameof(alpha));

        var result = originalImage.Copy();

        int imageWidth = result.Width;
        int imageHeight = result.Height;
        int maskHeight = mask.GetLength(0);
        int maskWidth = mask.GetLength(1);

        // Pre-calculate all constants
        float scaleX = (float)maskWidth / imageWidth;
        float scaleY = (float)maskHeight / imageHeight;
        float invAlpha = 1f - alpha;
        float alphaMaskR = alpha * maskColor.Red;
        float alphaMaskG = alpha * maskColor.Green;
        float alphaMaskB = alpha * maskColor.Blue;

        using var pixmap = result.PeekPixels();
        unsafe
        {
            byte* basePtr = (byte*)pixmap.GetPixels();
            int stride = pixmap.RowBytes;

            // Use parallel processing with partitioner for better load balancing
            var partitioner = Partitioner.Create(0, imageHeight);
            Parallel.ForEach(partitioner, range =>
            {
                for (int y = range.Item1; y < range.Item2; y++)
                {
                    int maskY = Math.Min((int)(y * scaleY), maskHeight - 1);
                    byte* rowPtr = basePtr + (y * stride);

                    for (int x = 0; x < imageWidth; x++)
                    {
                        int maskX = Math.Min((int)(x * scaleX), maskWidth - 1);

                        // Inline sigmoid check - avoid function call overhead
                        float logit = mask[maskY, maskX];
                        if (logit > 0) // Simplified check: logit > 0 means probability > 0.5
                        {
                            int pixelOffset = x * 4;
                            rowPtr[pixelOffset + 0] = (byte)(invAlpha * rowPtr[pixelOffset + 0] + alphaMaskR);
                            rowPtr[pixelOffset + 1] = (byte)(invAlpha * rowPtr[pixelOffset + 1] + alphaMaskG);
                            rowPtr[pixelOffset + 2] = (byte)(invAlpha * rowPtr[pixelOffset + 2] + alphaMaskB);
                        }
                    }
                }
            });
        }

        return result;
    }

    public static SKBitmap MaskToBitmap(float[,] mask, int width, int height, SKColor? foregroundColor = null, SKColor? backgroundColor = null)
    {
        var fg = foregroundColor ?? SKColors.White;
        var bg = backgroundColor ?? SKColors.Black;

        int maskHeight = mask.GetLength(0);
        int maskWidth = mask.GetLength(1);

        var bitmap = new SKBitmap(width, height, SKColorType.Rgba8888, SKAlphaType.Opaque);
        using var pixmap = bitmap.PeekPixels();
        unsafe
        {
            byte* ptr = (byte*)pixmap.GetPixels();
            int stride = pixmap.RowBytes;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int maskX = Math.Min((int)((float)x / width * maskWidth), maskWidth - 1);
                    int maskY = Math.Min((int)((float)y / height * maskHeight), maskHeight - 1);

                    float logit = mask[maskY, maskX];
                    float probability = 1f / (1f + MathF.Exp(-logit));
                    var color = probability > 0.5f ? fg : bg;

                    int pixelIndex = y * stride + x * 4;
                    ptr[pixelIndex + 0] = color.Red;
                    ptr[pixelIndex + 1] = color.Green;
                    ptr[pixelIndex + 2] = color.Blue;
                    ptr[pixelIndex + 3] = 255;
                }
            }
        }
        return bitmap;
    }

    private static SKColor[] GenerateColors(int count)
    {
        var colors = new SKColor[count];
        var random = new Random(42);
        for (int i = 0; i < count; i++)
        {
            colors[i] = new SKColor(
                (byte)random.Next(100, 256),
                (byte)random.Next(100, 256),
                (byte)random.Next(100, 256));
        }
        return colors;
    }
}