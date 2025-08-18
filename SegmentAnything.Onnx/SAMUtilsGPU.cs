using SkiaSharp;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Runtime.InteropServices;
using System.Collections.Concurrent;

namespace SegmentAnything.Onnx;

/// <summary>
/// Cross-platform GPU-accelerated utility functions for SAM operations
/// Supports Windows, macOS, iOS, and Android
/// </summary>
public static class SAMUtilsGPU
{
    private static GRContext _grContext;
    private static readonly object _contextLock = new object();

    /// <summary>
    /// Platform detection helper
    /// </summary>
    public static class PlatformHelper
    {
        public static bool IsAndroid =>
            RuntimeInformation.IsOSPlatform(OSPlatform.Create("ANDROID")) ||
#if ANDROID
            true;
#else
            false;
#endif

        public static bool IsIOS =>
            RuntimeInformation.IsOSPlatform(OSPlatform.Create("IOS")) ||
#if IOS
            true;
#else
            false;
#endif

        public static bool IsMacOS =>
            RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ||
#if MACCATALYST || MACOS
            true;
#else
            false;
#endif

        public static bool IsWindows =>
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ||
#if WINDOWS
            true;
#else
            false;
#endif
    }

    /// <summary>
    /// Initialize GPU context based on platform
    /// Note: SkiaSharp GPU support is limited and may not work on all platforms
    /// </summary>
    public static bool InitializeGPU()
    {
        lock (_contextLock)
        {
            if (_grContext != null)
                return true;

            try
            {
                // SkiaSharp's GPU support is limited in .NET MAUI
                // For now, we'll use CPU rendering which is still optimized
                System.Diagnostics.Debug.WriteLine("Note: Direct GPU context creation is limited in SkiaSharp for .NET MAUI");
                System.Diagnostics.Debug.WriteLine("Using optimized CPU rendering with hardware acceleration where available");

                // The GRContext.Create methods are deprecated in newer SkiaSharp versions
                // GPU acceleration in SkiaSharp for MAUI requires platform-specific setup
                // which is handled automatically by the platform's graphics layer

                return false; // Will use optimized CPU path
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"GPU initialization failed: {ex.Message}");
                return false;
            }
        }
    }

    /// <summary>
    /// Hardware-accelerated mask application using SkiaSharp's optimized blending
    /// Even without direct GPU context, SkiaSharp uses platform optimizations
    /// </summary>
    public static SKBitmap ApplyMaskToImageAccelerated(SKBitmap originalImage, float[,] mask, SKColor maskColor, float alpha = 0.5f)
    {
        if (originalImage == null) throw new ArgumentNullException(nameof(originalImage));
        if (mask == null) throw new ArgumentNullException(nameof(mask));
        if (alpha < 0f || alpha > 1f) throw new ArgumentOutOfRangeException(nameof(alpha));

        int width = originalImage.Width;
        int height = originalImage.Height;

        // Create surface with hardware acceleration hints
        var imageInfo = new SKImageInfo(width, height, SKColorType.Rgba8888, SKAlphaType.Premul);

        // Even without direct GPU context, SkiaSharp will use platform optimizations
        using var surface = SKSurface.Create(imageInfo);
        if (surface == null)
        {
            // Fallback to direct bitmap manipulation
            return ApplyMaskToImageOptimized(originalImage, mask, maskColor, alpha);
        }

        var canvas = surface.Canvas;

        // Draw original image
        canvas.DrawBitmap(originalImage, 0, 0);

        // Create and apply mask with optimized blending
        using (var maskBitmap = CreateMaskBitmapAccelerated(mask, width, height))
        using (var paint = new SKPaint())
        {
            // Use SkiaSharp's optimized color matrix
            var colorMatrix = new float[]
            {
                0, 0, 0, 0, maskColor.Red / 255f,
                0, 0, 0, 0, maskColor.Green / 255f,
                0, 0, 0, 0, maskColor.Blue / 255f,
                0, 0, 0, alpha, 0
            };

            paint.ColorFilter = SKColorFilter.CreateColorMatrix(colorMatrix);
            paint.BlendMode = SKBlendMode.SrcOver;

            // Platform-specific quality settings
            if (PlatformHelper.IsAndroid || PlatformHelper.IsIOS)
            {
                paint.FilterQuality = SKFilterQuality.Low; // Better performance on mobile
                paint.IsAntialias = false;
            }
            else
            {
                paint.FilterQuality = SKFilterQuality.Medium;
                paint.IsAntialias = true;
            }

            canvas.DrawBitmap(maskBitmap, 0, 0, paint);
        }

        // Get the result
        using var image = surface.Snapshot();
        return SKBitmap.FromImage(image);
    }

    /// <summary>
    /// Highly optimized CPU version with platform-specific optimizations
    /// </summary>
    public static SKBitmap ApplyMaskToImageOptimized(SKBitmap originalImage, float[,] mask, SKColor maskColor, float alpha = 0.5f)
    {
        var result = originalImage.Copy();

        int imageWidth = result.Width;
        int imageHeight = result.Height;
        int maskHeight = mask.GetLength(0);
        int maskWidth = mask.GetLength(1);

        // Platform-specific parallelization
        int maxParallelism = Environment.ProcessorCount;
        if (PlatformHelper.IsAndroid || PlatformHelper.IsIOS)
        {
            // Mobile: limit threads to avoid thermal throttling
            maxParallelism = Math.Min(maxParallelism, 4);
        }

        // Pre-calculate constants
        float scaleX = (float)maskWidth / imageWidth;
        float scaleY = (float)maskHeight / imageHeight;
        float invAlpha = 1f - alpha;

        // Pre-calculate color blending values
        byte alphaR = (byte)(alpha * maskColor.Red);
        byte alphaG = (byte)(alpha * maskColor.Green);
        byte alphaB = (byte)(alpha * maskColor.Blue);

        // Pre-process mask to binary for faster lookup
        var binaryMask = PreprocessMaskToBinary(mask);

        using var pixmap = result.PeekPixels();
        if (pixmap == null)
            return result;

        unsafe
        {
            byte* basePtr = (byte*)pixmap.GetPixels();
            int stride = pixmap.RowBytes;

            var options = new ParallelOptions
            {
                MaxDegreeOfParallelism = maxParallelism
            };

            // Process in chunks for better cache locality
            Parallel.For(0, imageHeight, options, y =>
            {
                int maskY = Math.Min((int)(y * scaleY), maskHeight - 1);
                byte* rowPtr = basePtr + (y * stride);

                // Process pixels in groups of 4 for better CPU utilization
                int x = 0;
                for (; x <= imageWidth - 4; x += 4)
                {
                    for (int dx = 0; dx < 4 && x + dx < imageWidth; dx++)
                    {
                        int currentX = x + dx;
                        int maskX = Math.Min((int)(currentX * scaleX), maskWidth - 1);

                        if (binaryMask[maskY, maskX])
                        {
                            int offset = currentX * 4; // RGBA

                            // Optimized blending without division
                            rowPtr[offset] = (byte)((rowPtr[offset] * invAlpha) + alphaR);
                            rowPtr[offset + 1] = (byte)((rowPtr[offset + 1] * invAlpha) + alphaG);
                            rowPtr[offset + 2] = (byte)((rowPtr[offset + 2] * invAlpha) + alphaB);
                            // Alpha channel (offset + 3) remains unchanged
                        }
                    }
                }

                // Handle remaining pixels
                for (; x < imageWidth; x++)
                {
                    int maskX = Math.Min((int)(x * scaleX), maskWidth - 1);

                    if (binaryMask[maskY, maskX])
                    {
                        int offset = x * 4;
                        rowPtr[offset] = (byte)((rowPtr[offset] * invAlpha) + alphaR);
                        rowPtr[offset + 1] = (byte)((rowPtr[offset + 1] * invAlpha) + alphaG);
                        rowPtr[offset + 2] = (byte)((rowPtr[offset + 2] * invAlpha) + alphaB);
                    }
                }
            });
        }

        return result;
    }

    /// <summary>
    /// Preprocess mask to binary for faster lookup
    /// </summary>
    private static bool[,] PreprocessMaskToBinary(float[,] mask)
    {
        int height = mask.GetLength(0);
        int width = mask.GetLength(1);
        var binaryMask = new bool[height, width];

        Parallel.For(0, height, y =>
        {
            for (int x = 0; x < width; x++)
            {
                // Simplified sigmoid: logit > 0 means probability > 0.5
                binaryMask[y, x] = mask[y, x] > 0;
            }
        });

        return binaryMask;
    }

    /// <summary>
    /// Create mask bitmap with hardware acceleration hints
    /// </summary>
    private static SKBitmap CreateMaskBitmapAccelerated(float[,] mask, int targetWidth, int targetHeight)
    {
        int maskHeight = mask.GetLength(0);
        int maskWidth = mask.GetLength(1);

        var bitmap = new SKBitmap(targetWidth, targetHeight, SKColorType.Alpha8, SKAlphaType.Premul);

        using var pixmap = bitmap.PeekPixels();
        if (pixmap == null)
            return bitmap;

        unsafe
        {
            byte* ptr = (byte*)pixmap.GetPixels();
            int stride = pixmap.RowBytes;

            // Platform-specific parallelization
            int maxParallelism = PlatformHelper.IsAndroid || PlatformHelper.IsIOS ? 4 : Environment.ProcessorCount;
            var options = new ParallelOptions { MaxDegreeOfParallelism = maxParallelism };

            Parallel.For(0, targetHeight, options, y =>
            {
                int maskY = Math.Min((int)((float)y / targetHeight * maskHeight), maskHeight - 1);
                byte* rowPtr = ptr + (y * stride);

                for (int x = 0; x < targetWidth; x++)
                {
                    int maskX = Math.Min((int)((float)x / targetWidth * maskWidth), maskWidth - 1);
                    // Simplified sigmoid check
                    rowPtr[x] = mask[maskY, maskX] > 0 ? (byte)255 : (byte)0;
                }
            });
        }

        return bitmap;
    }

    /// <summary>
    /// Configure ONNX Runtime session for platform-specific acceleration
    /// </summary>
    public static SessionOptions ConfigureOnnxOptimized()
    {
        var sessionOptions = new SessionOptions();
        sessionOptions.EnableCpuMemArena = false;
        sessionOptions.EnableMemoryPattern = false;

        // Platform-specific optimization
        if (PlatformHelper.IsAndroid || PlatformHelper.IsIOS)
        {
            // Mobile: optimize for lower memory usage
            sessionOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            sessionOptions.InterOpNumThreads = 2;
            sessionOptions.IntraOpNumThreads = 2;
        }
        else
        {
            // Desktop: optimize for speed
            sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;
            sessionOptions.InterOpNumThreads = Environment.ProcessorCount;
            sessionOptions.IntraOpNumThreads = Environment.ProcessorCount;
        }

        var availableProviders = OrtEnv.Instance().GetAvailableProviders();
        System.Diagnostics.Debug.WriteLine($"Available ONNX providers: {string.Join(", ", availableProviders)}");

        // Try to use hardware acceleration
        bool providerSet = false;

        if (availableProviders.Contains("CUDAExecutionProvider"))
        {
            try
            {
                sessionOptions.AppendExecutionProvider_CUDA();
                System.Diagnostics.Debug.WriteLine("Using CUDA execution provider");
                providerSet = true;
            }
            catch { }
        }

        if (!providerSet && availableProviders.Contains("DmlExecutionProvider"))
        {
            try
            {
                sessionOptions.AppendExecutionProvider_DML();
                System.Diagnostics.Debug.WriteLine("Using DirectML execution provider");
                providerSet = true;
            }
            catch { }
        }

        if (!providerSet && availableProviders.Contains("NnapiExecutionProvider"))
        {
            try
            {
                sessionOptions.AppendExecutionProvider_Nnapi();
                System.Diagnostics.Debug.WriteLine("Using NNAPI execution provider");
                providerSet = true;
            }
            catch { }
        }

        if (!providerSet && availableProviders.Contains("CoreMLExecutionProvider"))
        {
            try
            {
                sessionOptions.AppendExecutionProvider_CoreML();
                System.Diagnostics.Debug.WriteLine("Using CoreML execution provider");
                providerSet = true;
            }
            catch { }
        }

        if (!providerSet)
        {
            // Default to CPU with optimizations
            sessionOptions.AppendExecutionProvider_CPU();
            System.Diagnostics.Debug.WriteLine("Using optimized CPU execution provider");
        }

        return sessionOptions;
    }

    /// <summary>
    /// Alternative: Simple direct optimization without GPU context
    /// This is the most reliable cross-platform approach
    /// </summary>
    public static class SimpleFast
    {
        private static readonly ConcurrentDictionary<(int, int), bool[,]> _maskCache = new();

        public static SKBitmap ApplyMask(SKBitmap original, float[,] mask, SKColor color, float alpha = 0.5f)
        {
            var result = original.Copy();
            int w = result.Width;
            int h = result.Height;
            int mh = mask.GetLength(0);
            int mw = mask.GetLength(1);

            // Cache binary mask
            var key = (mh, mw);
            if (!_maskCache.TryGetValue(key, out var binary))
            {
                binary = new bool[mh, mw];
                for (int y = 0; y < mh; y++)
                    for (int x = 0; x < mw; x++)
                        binary[y, x] = mask[y, x] > 0;
                _maskCache[key] = binary;
            }

            // Direct pixel manipulation
            result.Erase(SKColors.Transparent);
            using (var canvas = new SKCanvas(result))
            {
                canvas.DrawBitmap(original, 0, 0);

                using var paint = new SKPaint
                {
                    Color = color.WithAlpha((byte)(alpha * 255)),
                    BlendMode = SKBlendMode.SrcOver
                };

                // Draw mask regions
                for (int y = 0; y < h; y++)
                {
                    int my = (int)(y * (float)mh / h);
                    for (int x = 0; x < w; x++)
                    {
                        int mx = (int)(x * (float)mw / w);
                        if (binary[my, mx])
                            canvas.DrawPoint(x, y, paint);
                    }
                }
            }

            return result;
        }
    }
}