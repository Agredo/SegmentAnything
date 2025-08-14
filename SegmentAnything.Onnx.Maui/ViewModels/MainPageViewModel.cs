using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Diagnostics;
using SkiaSharp;

namespace SegmentAnything.Onnx.Maui.ViewModels;

[ObservableObject]
public partial class MainPageViewModel
{
    string encoderPath = @"C:\Projects\Models\SAM\SAM2\sam2_hiera_tiny.encoder.onnx";
    string decoderPath = @"C:\Projects\Models\SAM\SAM2\sam2_hiera_tiny.decoder.onnx";

    string mobileSamDecoderPath = @"C:\Projects\Models\SAM\MobileSam\Qualcom\mobilesam-mobilesamdecoder.onnx\model.onnx\model.onnx";
    string mobileSamEncoderPath = @"C:\Projects\Models\SAM\MobileSam\Qualcom\mobilesam-mobilesamencoder.onnx\model.onnx\model.onnx";

    ////Android paths for MobileSAM models
    //string mobileSamDecoderPath = @"/storage/emulated/0/Models/SAM/MobileSAM/decoder";
    //string mobileSamEncoderPath = @"/storage/emulated/0/Models/SAM/MobileSAM/encoder";

    ////Android paths for SAM2 models
    //string encoderPath = @"/storage/emulated/0/Models/SAM/SAM2/encoder.onnx";
    //string decoderPath = @"/storage/emulated/0/Models/SAM/SAM2/decoder.onnx";

    public Stream Image { get; set; }

    [ObservableProperty]
    public byte[] maskedImage;

    public void MaskImage()
    {
        using var managedStream = new SKManagedStream(Image);
        using var codec = SKCodec.Create(managedStream);
        var info = codec.Info;        
        var imageBmp = SKBitmap.Decode(codec);

        using var mobileSam = new MobileSAM(mobileSamEncoderPath, mobileSamDecoderPath);
        SegmentPersonWithBoundingBox(mobileSam, imageBmp);
    }

    private void SegmentPersonWithBoundingBox(SAMModelBase sam2, SKBitmap image)
    {
        int imageWidth = image.Width;
        int imageHeight = image.Height;
        Debug.WriteLine($"Image Size: {imageWidth} x {imageHeight}");

        var boundingBox = new SKRectI((int)(imageWidth * 0.02), (int)(imageHeight * 0.02), (int)(imageWidth * 0.98), (int)(imageHeight * 0.98));

        var points = new SKPointI[]
        {
            new SKPointI((int)(imageWidth * 0.98), imageHeight / 2),
            new SKPointI(imageWidth / 2, imageHeight / 2)
        };

        var labels = new int[] { 0, 1 };
        var result = sam2.Segment(image, points, labels, boundingBox);

        using var mi = SAMUtils.ApplyMaskToImage(image, result.Masks[0], SKColors.Blue, 0.35f);
        using var imageData = mi.Encode(SKEncodedImageFormat.Png, 100);
        MaskedImage = imageData.ToArray();
    }
}
