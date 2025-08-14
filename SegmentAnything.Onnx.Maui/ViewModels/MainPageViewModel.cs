using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Diagnostics;
using System.Drawing;
using Point = System.Drawing.Point;

namespace SegmentAnything.Onnx.Maui.ViewModels;

[ObservableObject]
public partial class MainPageViewModel
{
    //string encoderPath = @"C:\Projects\Models\SAM\SAM2\sam2_hiera_tiny.encoder.onnx";
    //string decoderPath = @"C:\Projects\Models\SAM\SAM2\sam2_hiera_tiny.decoder.onnx";

    //string mobileSamDecoderPath = @"C:\Projects\Models\SAM\MobileSam\Qualcom\mobilesam-mobilesamdecoder.onnx\model.onnx\model.onnx";
    //string mobileSamEncoderPath = @"C:\Projects\Models\SAM\MobileSam\Qualcom\mobilesam-mobilesamencoder.onnx\model.onnx\model.onnx";

    // Android internal Path
    string mobileSamDecoderPath = @"/storage/emulated/0/Models/SAM/MobileSAM/decoder";
    string mobileSamEncoderPath = @"/storage/emulated/0/Models/SAM/MobileSAM/encoder";

    string encoderPath = @"/storage/emulated/0/Models/SAM/SAM2/encoder.onnx";
    string decoderPath = @"/storage/emulated/0/Models/SAM/SAM2/decoder.onnx";

    public Stream Image { get; set; }

    [ObservableProperty]
    public byte[] maskedImage;

    public void MaskImage()
    {
        var image = new Bitmap(Image);
        //using var sam2 = new SAM2(encoderPath, decoderPath);
        //SegmentPersonWithBoundingBox(sam2, image);

        using var mobileSam = new MobileSAM(mobileSamEncoderPath, mobileSamDecoderPath);

        SegmentPersonWithBoundingBox(mobileSam, image);
    }

    private void SegmentPersonWithBoundingBox(SAMModelBase sam2, Bitmap image)
    {
        int imageWidth = image.Width;
        int imageHeight = image.Height;

        //Console output image size
        Debug.WriteLine($"Image Size: {imageWidth} x {imageHeight}");

        var boundingBox = new Rectangle((int)(imageWidth * 0.02), (int)(imageHeight * 0.02), (int)(imageWidth * 0.96), (int)(imageHeight * 0.96));

        // Ein zentraler Punkt als zusätzlicher Hinweis
        var points = new Point[]
        {
            //new Point(imageWidth/2, (int)(imageHeight * 0.01)), //Oberer Rand (1% des Bildes)
            //new Point(imageWidth / 2, (int)(imageHeight * 0.98)), // Unterer Rand (98% des Bildes)
            //new Point((int)(imageWidth * 0.01), imageHeight / 2), // Linker Rand (1% des Bildes)
            new Point((int)(imageWidth * 0.98), imageHeight / 2),  // Rechter Rand (98% des Bildes)
            new Point(imageWidth / 2, imageHeight / 2) // Mitte des Bildes
        };

        var labels = new int[] { 0,1 };
        var result = sam2.Segment(image, points, labels, boundingBox);

        var mi = SAMUtils.ApplyMaskToImage(image, result.Masks[0], System.Drawing.Color.Blue, 0.35f);

        //Bitmap to byte array
        using var ms = new MemoryStream();
        mi.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
        MaskedImage = ms.ToArray();
    }
}
