using Microsoft.ML.OnnxRuntime;
using System.Drawing;
using System.Drawing.Imaging;

namespace SegmentAnything.Onnx;

// Base-Klasse für SAM-Modelle
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
        else
        {
            throw new NotSupportedException("No supported execution provider found. Please install ONNX Runtime with GPU support.");
        }
    }

    protected virtual float[] PreprocessImage(Bitmap image)
    {
        // Resize und Normalisierung für SAM (ImageNet stats)
        var resized = new Bitmap(image, ImageSize, ImageSize);
        var pixels = new float[3 * ImageSize * ImageSize];

        var bitmapData = resized.LockBits(
            new Rectangle(0, 0, ImageSize, ImageSize),
            ImageLockMode.ReadOnly,
            PixelFormat.Format24bppRgb);

        unsafe
        {
            byte* ptr = (byte*)bitmapData.Scan0;

            for (int y = 0; y < ImageSize; y++)
            {
                for (int x = 0; x < ImageSize; x++)
                {
                    int pixelIndex = y * bitmapData.Stride + x * 3;

                    // BGR zu RGB und Normalisierung (ImageNet Stats)
                    float r = (ptr[pixelIndex + 2] / 255.0f - 0.485f) / 0.229f;
                    float g = (ptr[pixelIndex + 1] / 255.0f - 0.456f) / 0.224f;
                    float b = (ptr[pixelIndex] / 255.0f - 0.406f) / 0.225f;

                    // Channel-first Format: [C, H, W]
                    int baseIndex = y * ImageSize + x;
                    pixels[baseIndex] = r;                                    // R channel
                    pixels[baseIndex + ImageSize * ImageSize] = g;           // G channel
                    pixels[baseIndex + 2 * ImageSize * ImageSize] = b;       // B channel
                }
            }
        }

        resized.UnlockBits(bitmapData);
        resized.Dispose();

        return pixels;
    }

    public abstract SAMResult Segment(Bitmap image, Point[] points, int[] labels, Rectangle? boundingBox = null);

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
