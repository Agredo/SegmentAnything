using SegmentAnything.Onnx;
using System.Diagnostics;
using SkiaSharp;

string encoderPath = @"C:\Projects\Models\SAM\SAM2\sam2_hiera_tiny.encoder.onnx";
string decoderPath = @"C:\Projects\Models\SAM\SAM2\sam2_hiera_tiny.decoder.onnx";
string imagePath = @"C:\Users\chris\OneDrive\Bilder\Camera Roll\IMG_20250108_024734.jpg";

Main();

void Main()
{
    try
    {
        using var image = LoadBitmap(imagePath);
        using var sam2 = new SAM2(encoderPath, decoderPath);

        Console.WriteLine($"Bildgröße: {image.Width}x{image.Height}");

        string input = string.Empty;
        while (input != "exit")
        {
            Console.WriteLine("Insert Image path");
            input = Console.ReadLine();
            if (string.IsNullOrEmpty(input) || input.ToLower() == "exit")
                break;
            imagePath = input;
            using var newImage = LoadBitmap(imagePath);
            SegmentPersonWithBoundingBox(sam2, newImage);
            Console.WriteLine("Segmentierung abgeschlossen! Überprüfen Sie die Output-Dateien.");
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Fehler: {ex.Message}");
        Console.WriteLine($"Stacktrace: {ex.StackTrace}");
    }
}

static SKBitmap LoadBitmap(string path)
{
    using var fs = File.OpenRead(path);
    return SKBitmap.Decode(fs);
}

static void SegmentPersonWithBoundingBox(SAM2 sam2, SKBitmap image)
{
    Console.WriteLine("\n=== Person-Segmentierung mit Bounding Box ===");

    int personLeft = 0;
    int personTop = 0;
    int personWidth = image.Width;
    int personHeight = image.Height;

    var boundingBox = new SKRectI(personLeft, personTop, personLeft + personWidth, personTop + personHeight);

    var points = new SKPointI[]
    {
        new SKPointI(personWidth/2, (int)(personHeight * 0.01)),
        new SKPointI(personWidth/2, (int)(personHeight * 0.99)),
        new SKPointI((int)(personWidth * 0.01), personHeight / 2),
        new SKPointI((int)(personWidth * 0.99), personHeight / 2),
        new SKPointI(personWidth / 2, personHeight / 2)
    };

    var labels = new int[] {0,0,0,0,1};
    Stopwatch stopwatch = Stopwatch.StartNew();
    var result = sam2.Segment(image, points, labels, boundingBox);
    stopwatch.Stop();
    Console.WriteLine($"Segmentierung mit Bounding Box abgeschlossen in {stopwatch.ElapsedMilliseconds} ms");

    Console.WriteLine($"Bounding Box Segmentierung - Score: {result.Scores[0]:F3}");

    stopwatch.Restart();
    var maskBitmap = result.GetBestMaskAsBitmap();
    if (maskBitmap != null)
    {
        SaveBitmap(maskBitmap, "person_mask_boundingbox.png");
        maskBitmap.Dispose();
    }
    stopwatch.Stop();
    Console.WriteLine($"Maske gespeichert in {stopwatch.ElapsedMilliseconds} ms");

    stopwatch.Restart();
    using (var maskedImage = SAMUtils.ApplyMaskToImage(image, result.Masks[0], SKColors.Blue, 0.35f))
    {
        SaveBitmap(maskedImage, "person_highlighted_boundingbox.png");
    }
    stopwatch.Stop();
    Console.WriteLine($"Maskiertes Bild gespeichert in {stopwatch.ElapsedMilliseconds} ms");

    stopwatch.Restart();
    VisualizeBoundingBox(image, boundingBox, points, "debug_boundingbox.png");
    stopwatch.Stop();
    Console.WriteLine($"Bounding Box visualisiert in {stopwatch.ElapsedMilliseconds} ms");
}

static void SaveBitmap(SKBitmap bmp, string filename)
{
    using var data = bmp.Encode(SKEncodedImageFormat.Png, 100);
    using var fs = File.OpenWrite(filename);
    data.SaveTo(fs);
}

static void VisualizePoints(SKBitmap image, SKPointI[] points, int[] labels, string filename)
{
    using var temp = new SKBitmap(image.Width, image.Height, SKColorType.Rgba8888, SKAlphaType.Premul);
    using (var canvas = new SKCanvas(temp))
    {
        canvas.DrawBitmap(image, 0, 0);
        for (int i = 0; i < points.Length; i++)
        {
            var color = labels[i] == 1 ? SKColors.Lime : SKColors.Red;
            using var paint = new SKPaint { Color = color, IsAntialias = true, Style = SKPaintStyle.Fill };
            canvas.DrawCircle(points[i].X, points[i].Y, 8, paint);
        }
    }
    SaveBitmap(temp, filename);
    temp.Dispose();
    Console.WriteLine($"Debug-Bild mit Punkten gespeichert: {filename}");
}

static void VisualizeBoundingBox(SKBitmap image, SKRectI boundingBox, SKPointI[] points, string filename)
{
    using var temp = new SKBitmap(image.Width, image.Height, SKColorType.Rgba8888, SKAlphaType.Premul);
    using (var canvas = new SKCanvas(temp))
    {
        canvas.DrawBitmap(image, 0, 0);
        using (var pen = new SKPaint { Color = SKColors.Yellow, StrokeWidth = 3, Style = SKPaintStyle.Stroke })
        {
            canvas.DrawRect(new SKRect(boundingBox.Left, boundingBox.Top, boundingBox.Right, boundingBox.Bottom), pen);
        }
        using var pointFill = new SKPaint { Color = SKColors.Lime, Style = SKPaintStyle.Fill, IsAntialias = true };
        using var pointStroke = new SKPaint { Color = SKColors.Green, Style = SKPaintStyle.Stroke, StrokeWidth = 2, IsAntialias = true };
        foreach (var p in points)
        {
            canvas.DrawCircle(p.X, p.Y, 6, pointFill);
            canvas.DrawCircle(p.X, p.Y, 6, pointStroke);
        }
    }
    SaveBitmap(temp, filename);
    temp.Dispose();
    Console.WriteLine($"Debug-Bild mit Bounding Box gespeichert: {filename}");
}





