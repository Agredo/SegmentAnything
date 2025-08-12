using SegmentAnything.Onnx;
using System.Diagnostics;
using System.Drawing;

string encoderPath = @"C:\Projects\Models\SAM\SAM2\sam2_hiera_tiny.encoder.onnx";
string decoderPath = @"C:\Projects\Models\SAM\SAM2\sam2_hiera_tiny.decoder.onnx";
string imagePath = @"C:\Users\chris\OneDrive\Bilder\Camera Roll\IMG_20250108_024734.jpg";

Main();


void Main()
{
    try
    {
        var image = new Bitmap(imagePath);
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
            image.Dispose();
            image = new Bitmap(imagePath);

            SegmentPersonWithBoundingBox(sam2, image);
            Console.WriteLine("Segmentierung abgeschlossen! Überprüfen Sie die Output-Dateien.");
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Fehler: {ex.Message}");
        Console.WriteLine($"Stacktrace: {ex.StackTrace}");
    }
}

static void SegmentPersonWithBoundingBox(SAM2 sam2, Bitmap image)
{
    Console.WriteLine("\n=== Person-Segmentierung mit Bounding Box ===");

    // Bounding Box um die gesamte Person
    // Geschätzt basierend auf der Position im Bild
    //int personLeft = image.Width / 8;     // Person beginnt etwa bei 1/8 der Breite
    //int personTop = image.Height/ 8;     // Kopf ist etwa bei 1/8 der Höhe
    //int personWidth = image.Width / 3;    // Person ist etwa 1/4 der Bildbreite breit
    //int personHeight = image.Height * 4 / 5; // Person ist etwa 3/4 der Bildhöhe hoch

    int personLeft = 0;
    int personTop = 0;
    int personWidth = image.Width;    
    int personHeight = image.Height;

    var boundingBox = new Rectangle(personLeft, personTop, personWidth, personHeight);

    // Ein zentraler Punkt als zusätzlicher Hinweis
    var points = new Point[]
    {
            new Point(personWidth/2, (int)(personHeight * 0.01)), //Oberer Rand (1% des Bildes)
            new Point(personWidth / 2, (int)(personHeight * 0.99)), // Unterer Rand (99% des Bildes)
            new Point((int)(personWidth * 0.01), personHeight / 2), // Linker Rand (1% des Bildes)
            new Point((int)(personWidth * 0.99), personHeight / 2),  // Rechter Rand (99% des Bildes)
            new Point(personWidth / 2, personHeight / 2) // Mitte des Bildes
    };

    var labels = new int[] { 0,0,0,0,1 };
    Stopwatch stopwatch = Stopwatch.StartNew();
    var result = sam2.Segment(image, points, labels, boundingBox);
    stopwatch.Stop();
    Console.WriteLine($"Segmentierung mit Bounding Box abgeschlossen in {stopwatch.ElapsedMilliseconds} ms");

    Console.WriteLine($"Bounding Box Segmentierung - Score: {result.Scores[0]:F3}");

    stopwatch.Restart();
    // Ergebnisse speichern
    var maskBitmap = result.GetBestMaskAsBitmap();
    maskBitmap?.Save("person_mask_boundingbox.png");
    maskBitmap?.Dispose();
    stopwatch.Stop();
    Console.WriteLine($"Maske gespeichert in {stopwatch.ElapsedMilliseconds} ms");

    stopwatch.Restart();
    var maskedImage = SAMUtils.ApplyMaskToImage(image, result.Masks[0], Color.Blue, 0.35f);
    maskedImage.Save("person_highlighted_boundingbox.png");
    maskedImage.Dispose();
    stopwatch.Stop();
    Console.WriteLine($"Maskiertes Bild gespeichert in {stopwatch.ElapsedMilliseconds} ms");

    // Bounding Box visualisieren
    stopwatch.Restart();
    VisualizeBoundingBox(image, boundingBox, points, "debug_boundingbox.png");
    stopwatch.Stop();
    Console.WriteLine($"Bounding Box visualisiert in {stopwatch.ElapsedMilliseconds} ms");
}

static void VisualizePoints(Bitmap image, Point[] points, int[] labels, string filename)
{
    var debugImage = new Bitmap(image);
    using (var g = Graphics.FromImage(debugImage))
    {
        for (int i = 0; i < points.Length; i++)
        {
            var brush = labels[i] == 1 ? Brushes.Lime : Brushes.Red;
            var pen = labels[i] == 1 ? Pens.Green : Pens.Red;

            // Punkt als Kreis zeichnen
            int radius = 8;
            g.FillEllipse(brush, points[i].X - radius, points[i].Y - radius, radius * 2, radius * 2);
            g.DrawEllipse(pen, points[i].X - radius, points[i].Y - radius, radius * 2, radius * 2);

            // Punkt-Nummer hinzufügen
            g.DrawString(i.ToString(), SystemFonts.DefaultFont, Brushes.White, points[i].X + 10, points[i].Y - 10);
        }
    }

    debugImage.Save(filename);
    debugImage.Dispose();
    Console.WriteLine($"Debug-Bild mit Punkten gespeichert: {filename}");
}

static void VisualizeBoundingBox(Bitmap image, Rectangle boundingBox, Point[] points, string filename)
{
    var debugImage = new Bitmap(image);
    using (var g = Graphics.FromImage(debugImage))
    {
        // Bounding Box zeichnen
        using (var pen = new Pen(Color.Yellow, 3))
        {
            g.DrawRectangle(pen, boundingBox);
        }

        // Punkte zeichnen
        foreach (var point in points)
        {
            g.FillEllipse(Brushes.Lime, point.X - 6, point.Y - 6, 12, 12);
            g.DrawEllipse(Pens.Green, point.X - 6, point.Y - 6, 12, 12);
        }
    }

    debugImage.Save(filename);
    debugImage.Dispose();
    Console.WriteLine($"Debug-Bild mit Bounding Box gespeichert: {filename}");
}





