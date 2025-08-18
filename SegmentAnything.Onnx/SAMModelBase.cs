using Microsoft.ML.OnnxRuntime;
using SkiaSharp;
using System.Diagnostics;

namespace SegmentAnything.Onnx;

// Base-Klasse für SAM-Modelle (SkiaSharp Variante)
public abstract class SAMModelBase : IDisposable
{
    protected Stopwatch stopwatch = new Stopwatch();
    protected InferenceSession _encoderSession;
    protected InferenceSession _decoderSession;
    protected bool _disposed = false;

    public const int ImageSize = 1024;

    public SessionOptions SessionOptions { get; set; }

    protected SAMModelBase(string encoderModelPath, string decoderModelPath, Action<SessionOptions> ConfigureSessionOptions)
    {
        SessionOptions = new SessionOptions();

        if (ConfigureSessionOptions != null)
            ConfigureSessionOptions(SessionOptions);
        else
            ConfigureExecutionProvider(SessionOptions);

        _encoderSession = new InferenceSession(encoderModelPath, SessionOptions);
        _decoderSession = new InferenceSession(decoderModelPath, SessionOptions);
    }

    private static void ConfigureExecutionProvider(SessionOptions sessionOptions)
    {
        sessionOptions.EnableCpuMemArena = true;
        sessionOptions.EnableMemoryPattern = true;
        sessionOptions.InterOpNumThreads = Environment.ProcessorCount;
        sessionOptions.IntraOpNumThreads = Environment.ProcessorCount;
        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;

        // GPU unterstützung falls verfügbar
        var availableProviders = OrtEnv.Instance().GetAvailableProviders();
        Debug.WriteLine("Available Execution Providers: " + string.Join(", ", availableProviders));

        if (availableProviders.Contains("CUDAExecutionProvider"))
        {
            sessionOptions.AppendExecutionProvider_CUDA();
            Debug.WriteLine("CUDA execution provider added successfully.");
        }
        if (availableProviders.Contains("DmlExecutionProvider"))
        {
            sessionOptions.AppendExecutionProvider_DML();
            Debug.WriteLine("DirectML execution provider added successfully.");
        }
        if (availableProviders.Contains("OpenVINOExecutionProvider"))
        {
            sessionOptions.AppendExecutionProvider_OpenVINO();
            Debug.WriteLine("OpenVINO execution provider added successfully.");
        }
        if (availableProviders.Contains("CANNExecutionProvider"))
        {
            sessionOptions.AppendExecutionProvider("CANNExecutionProvider");
            
        }
        if (availableProviders.Contains("QNNExecutionProvider"))
        {
            sessionOptions.AppendExecutionProvider("QNNExecutionProvider");
            Debug.WriteLine("QNN execution provider added successfully.");
        }

        if (availableProviders.Contains("NnapiExecutionProvider"))
        {
            sessionOptions.AppendExecutionProvider("NnapiExecutionProvider");
            Debug.WriteLine("NnapiExecutionProvider execution provider added successfully.");
        }
        else if(availableProviders.Contains("XnnpackExecutionProvider"))
        {
            sessionOptions.AppendExecutionProvider("XnnpackExecutionProvider");
            Debug.WriteLine("XNNPACK execution provider added successfully.");
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
