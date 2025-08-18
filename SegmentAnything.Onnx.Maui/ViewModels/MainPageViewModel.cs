using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.ML.OnnxRuntime;
using SkiaSharp;
using System.Diagnostics;

namespace SegmentAnything.Onnx.Maui.ViewModels;

[ObservableObject]
public partial class MainPageViewModel
{
    // Windows paths for SAM2 models
    string encoderPath = @"C:\Projects\Models\SAM\SAM2\sam2_hiera_tiny.encoder.onnx";
    string decoderPath = @"C:\Projects\Models\SAM\SAM2\sam2_hiera_tiny.decoder.onnx";

    // Windows paths for MobileSAM models
    string mobileSamDecoderPath = @"C:\Projects\Models\SAM\MobileSam\Qualcom\mobilesam-mobilesamdecoder.onnx\model.onnx\model.onnx";
    string mobileSamEncoderPath = @"C:\Projects\Models\SAM\MobileSam\Qualcom\mobilesam-mobilesamencoder.onnx\model.onnx\model.onnx";

    ////Android paths for MobileSAM models
    //string mobileSamDecoderPath = @"/storage/emulated/0/Models/SAM/MobileSAM/decoder/model.onnx";
    //string mobileSamEncoderPath = @"/storage/emulated/0/Models/SAM/MobileSAM/encoder/model.onnx";

    ////Android paths for SAM2 models
    //string encoderPath = @"/storage/emulated/0/Models/SAM/SAM2/encoder.onnx";
    //string decoderPath = @"/storage/emulated/0/Models/SAM/SAM2/decoder.onnx";

    public Stream Image { get; set; }

    [ObservableProperty]
    public byte[] maskedImage;

    private SAMModelBase sam;

    public MainPageViewModel()
    {
        PermissionCheck();
    }

    private async void PermissionCheck()
    {
        var status = await Permissions.CheckStatusAsync<Permissions.StorageRead>();
        if (status != PermissionStatus.Granted)
        {
            status = await Permissions.RequestAsync<Permissions.StorageRead>();
        }
        if (status != PermissionStatus.Granted)
        {
            Debug.WriteLine("Permission denied");
            return;
        }
        
        Debug.WriteLine("Permission granted");

        //sam = new SAM2(encoderPath, decoderPath);

        sam = new MobileSAM(mobileSamEncoderPath, mobileSamDecoderPath, ConfigureExecutionProvider);
    }

    public void MaskImage()
    {
        using var managedStream = new SKManagedStream(Image);
        using var codec = SKCodec.Create(managedStream);
        var info = codec.Info;        
        var imageBmp = SKBitmap.Decode(codec);
        SegmentPersonWithBoundingBox(sam, imageBmp);
    }

    private void SegmentPersonWithBoundingBox(SAMModelBase sam2, SKBitmap image)
    {
        var stopwatch = Stopwatch.StartNew();

        int imageWidth = image.Width;
        int imageHeight = image.Height;
        Debug.WriteLine($"Image Size: {imageWidth} x {imageHeight}");

        var boundingBox = new SKRectI((int)(imageWidth * 0.02), (int)(imageHeight * 0.02), (int)(imageWidth * 0.98), (int)(imageHeight * 0.98));

        ////Sam2 expects points in the format of SKPointI, which is an integer point type.
        //var points = new SKPointI[]
        //{
        //    new SKPointI(imageWidth/2, (int)(imageHeight * 0.04)), //Oberer Rand (1% des Bildes)
        //    new SKPointI(imageWidth / 2, (int)(imageHeight * 0.92)), // Unterer Rand (98% des Bildes)
        //    new SKPointI((int)(imageWidth * 0.04), imageHeight / 2), // Linker Rand (1% des Bildes)
        //    new SKPointI((int)(imageWidth * 0.92), imageHeight / 2), //Rechter Rand (1
        //    new SKPointI(imageWidth / 2, imageHeight / 2)
        //};
        //var labels = new int[] { 0, 0, 0, 0, 1 };

        //MobileSAM requires exactly 2 points, so we use the first two points for segmentation.
        var points = new SKPointI[]
        {
            new SKPointI((int)(imageWidth * 0.92), imageHeight / 2), //Rechter Rand (1
            new SKPointI(imageWidth / 2, imageHeight / 2)
        };
        var labels = new int[] { 0, 1 };

        stopwatch = Stopwatch.StartNew();
        var result = sam2.Segment(image, points, labels, boundingBox);
        stopwatch.Stop();
        Debug.WriteLine($"Segment: {stopwatch.ElapsedMilliseconds}ms");
       
        stopwatch.Restart();
        using var mi = SAMUtils.ApplyMaskToImageFast(image, result.Masks[0], SKColors.Blue, 0.35f);
        stopwatch.Stop();
        Debug.WriteLine($"Optimized: {stopwatch.ElapsedMilliseconds}ms");

        //stopwatch.Restart();
        //using var mi2 = SAMUtilsGPU.ApplyMaskToImageAccelerated(image, result.Masks[0], SKColors.Blue, 0.35f);
        //stopwatch.Stop();
        //Debug.WriteLine($"GPU Optimized: {stopwatch.ElapsedMilliseconds}ms");

        using var imageData = mi.Encode(SKEncodedImageFormat.Png, 100);
        MaskedImage = imageData.ToArray();
    }

    private static void ConfigureExecutionProvider(SessionOptions so)
    {
        // General tuning
        so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        so.IntraOpNumThreads = Math.Min(Environment.ProcessorCount, 4);

        var providers = OrtEnv.Instance().GetAvailableProviders();
        System.Diagnostics.Debug.WriteLine("Available EPs: " + string.Join(", ", providers));

#if ANDROID
    // Android: NNAPI → XNNPACK
    if (providers.Contains("NnapiExecutionProvider"))
    {
            try
            {
                // Enable FP16 for better perf/efficiency on NPUs
                so.AppendExecutionProvider_Nnapi(NnapiFlags.NNAPI_FLAG_USE_FP16 /* NNAPI_FLAG_USE_FP16 */);
                System.Diagnostics.Debug.WriteLine("NNAPI added (FP16).");
            }
            catch (Exception ex)
            {
            }
    }
    if (providers.Contains("XnnpackExecutionProvider"))
    {
        // Threads use IntraOpNumThreads
        try 
            { 
                so.AppendExecutionProvider("XnnpackExecutionProvider");
            }
        catch
            {
            }
        System.Diagnostics.Debug.WriteLine($"XNNPACK added (threads={so.IntraOpNumThreads}).");
    }

#elif IOS || MACCATALYST
    // iOS/macOS: CoreML → CPU (XNNPACK optional)
    if (providers.Contains("CoreMLExecutionProvider"))
    {
        try
        {
            // 1 = COREML_FLAG_USE_CPU_ONLY false, enable FP16 where allowed
            so.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_USE_CPU_AND_GPU);
            System.Diagnostics.Debug.WriteLine("CoreML added.");
        }
        catch { so.AppendExecutionProvider("CoreMLExecutionProvider"); }
    }
    if (providers.Contains("XnnpackExecutionProvider"))
    {
        try 
            { 
                so.AppendExecutionProvider("XnnpackExecutionProvider"); 
            }
        catch 
            {
            }
            System.Diagnostics.Debug.WriteLine("XNNPACK added.");
    }

#elif WINDOWS
        // Windows: prefer CUDA if present, else DirectML
        if (providers.Contains("CUDAExecutionProvider"))
        {
            try { so.AppendExecutionProvider_CUDA(); }
            catch { so.AppendExecutionProvider("CUDAExecutionProvider"); }
            System.Diagnostics.Debug.WriteLine("CUDA added.");
        }
        else if (providers.Contains("QnnExecutionProvider"))
        {
            so.AppendExecutionProvider("QnnExecutionProvider");
            System.Diagnostics.Debug.WriteLine("QNN added.");
        }
        else if (providers.Contains("OpenVINOExecutionProvider"))
        {
            so.AppendExecutionProvider_OpenVINO();
            System.Diagnostics.Debug.WriteLine("OpenVINO added.");
        }

#elif LINUX
    // Linux: CUDA → OpenVINO (optional) → CPU
    if (providers.Contains("CUDAExecutionProvider"))
    {
        so.AppendExecutionProvider("CUDAExecutionProvider");
        System.Diagnostics.Debug.WriteLine("CUDA added.");
    }
    else if (providers.Contains("OpenVINOExecutionProvider"))
    {
        so.AppendExecutionProvider_OpenVINO();
        System.Diagnostics.Debug.WriteLine("OpenVINO added.");
    }
#endif

        // Optional extras if you ship these EPs
        if (providers.Contains("QnnExecutionProvider"))
        {
            so.AppendExecutionProvider("QnnExecutionProvider");
            System.Diagnostics.Debug.WriteLine("QNN added.");
        }
        if (providers.Contains("CANNExecutionProvider"))
        {
            so.AppendExecutionProvider("CANNExecutionProvider");
            System.Diagnostics.Debug.WriteLine("CANN added.");
        }
    }

}
