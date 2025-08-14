using Microsoft.ML.OnnxRuntime;
using SkiaSharp;

namespace SegmentAnything.Onnx;

// Base-Klasse für SAM-Modelle (SkiaSharp Variante)
public abstract class SAMModelBase : IDisposable
{
    protected InferenceSession _encoderSession;
    protected InferenceSession _decoderSession;
    protected bool _disposed = false;

    public const int ImageSize = 1024;

    protected SAMModelBase(string encoderModelPath, string decoderModelPath)
    {
        var sessionOptions = new SessionOptions();
        sessionOptions.EnableCpuMemArena = false;
        sessionOptions.EnableMemoryPattern = false;

        ConfigureExecutionProvider(sessionOptions);

        _encoderSession = new InferenceSession(encoderModelPath, sessionOptions);
        _decoderSession = new InferenceSession(decoderModelPath, sessionOptions);
    }

    private static void ConfigureExecutionProvider(SessionOptions sessionOptions)
    {
        // GPU unterstützung falls verfügbar
        var availableProviders = OrtEnv.Instance().GetAvailableProviders();
        if (availableProviders.Contains("CUDAExecutionProvider"))
        {
            sessionOptions.AppendExecutionProvider_CUDA();
        }
        else if (availableProviders.Contains("DmlExecutionProvider"))
        {
            sessionOptions.AppendExecutionProvider_DML();
        }
        else if (availableProviders.Contains("CPUExecutionProvider"))
        {
            sessionOptions.AppendExecutionProvider_CPU();
        }
        else if (availableProviders.Contains("OpenVINOExecutionProvider"))
        {
            sessionOptions.AppendExecutionProvider_OpenVINO();
        }
        else if (availableProviders.Contains("NnapiExecutionProvider"))
        {
            sessionOptions.AppendExecutionProvider_Nnapi();
        }
        else if (availableProviders.Contains("CANNExecutionProvider"))
        {
            sessionOptions.AppendExecutionProvider("CANNExecutionProvider");
        }
        else
        {
            throw new NotSupportedException("No supported execution provider found. Please install ONNX Runtime with GPU support.");
        }
    }

    protected virtual float[] PreprocessImage(SKBitmap image)
    {
        if (image == null) throw new ArgumentNullException(nameof(image));

        // Resize (1024x1024) & Normalisierung (ImageNet Stats)
        using var source = EnsureFormat(image, SKColorType.Rgba8888);
        using var resized = new SKBitmap(ImageSize, ImageSize, SKColorType.Rgba8888, SKAlphaType.Premul);
        source.ScalePixels(resized, SKFilterQuality.Medium);

        var pixels = new float[3 * ImageSize * ImageSize];

        // Direkter Zugriff auf Pixel (RGBA8888)
        using var pixmap = resized.PeekPixels();
        unsafe
        {
            byte* ptr = (byte*)pixmap.GetPixels();
            int stride = pixmap.RowBytes; // Bytes pro Zeile

            for (int y = 0; y < ImageSize; y++)
            {
                for (int x = 0; x < ImageSize; x++)
                {
                    int pixelIndex = y * stride + x * 4; // RGBA

                    float r = (ptr[pixelIndex + 0] / 255.0f - 0.485f) / 0.229f;
                    float g = (ptr[pixelIndex + 1] / 255.0f - 0.456f) / 0.224f;
                    float b = (ptr[pixelIndex + 2] / 255.0f - 0.406f) / 0.225f;

                    int baseIndex = y * ImageSize + x;
                    pixels[baseIndex] = r;                                         // R
                    pixels[baseIndex + ImageSize * ImageSize] = g;                // G
                    pixels[baseIndex + 2 * ImageSize * ImageSize] = b;            // B
                }
            }
        }

        return pixels;
    }

    private static SKBitmap EnsureFormat(SKBitmap bmp, SKColorType targetType)
    {
        if (bmp.ColorType == targetType) return bmp.Copy();
        var converted = new SKBitmap(bmp.Width, bmp.Height, targetType, SKAlphaType.Premul);
        using var canvas = new SKCanvas(converted);
        canvas.DrawBitmap(bmp, 0, 0);
        return converted;
    }

    public abstract SAMResult Segment(SKBitmap image, SKPointI[] points, int[] labels, SKRectI? boundingBox = null);

    public virtual void Dispose()
    {
        if (!_disposed)
        {
            _encoderSession?.Dispose();
            _decoderSession?.Dispose();
            _disposed = true;
        }
    }
}
