# Agredo.SegmentAnything.Onnx

A high-performance .NET library for running Segment Anything Model (SAM) and SAM2 using ONNX Runtime. This library provides easy-to-use APIs for advanced image segmentation with support for multiple SAM model variants.

## Features

- ?? **High Performance**: GPU acceleration support (CUDA, DirectML, OpenVINO)
- ?? **Multiple Model Support**: SAM, SAM2, and MobileSAM implementations
- ?? **Video Support**: SAM2 with temporal memory for video frame tracking
- ?? **Flexible Prompting**: Point-based prompts, bounding box prompts, and combinations
- ??? **Comprehensive Utilities**: Built-in image processing and mask manipulation tools
- ?? **Easy Integration**: Simple API designed for .NET applications

## Installation

Install via NuGet Package Manager:

```bash
dotnet add package Agredo.SegmentAnything.Onnx
```

Or via Package Manager Console:

```
Install-Package Agredo.SegmentAnything.Onnx
```

## Quick Start

### Basic SAM2 Usage

```csharp
using SegmentAnything.Onnx;
using System.Drawing;

// Initialize SAM2 with your ONNX model files
string encoderPath = "path/to/sam2_encoder.onnx";
string decoderPath = "path/to/sam2_decoder.onnx";

using var sam2 = new SAM2(encoderPath, decoderPath);

// Load your image
var image = new Bitmap("path/to/your/image.jpg");

// Define points and labels (1 = positive point, 0 = negative point)
var points = new Point[] { new Point(400, 300) };  // Click coordinates
var labels = new int[] { 1 };  // Positive point

// Perform segmentation
var result = sam2.Segment(image, points, labels);

// Get the best mask as bitmap
var maskBitmap = result.GetBestMaskAsBitmap(image.Width, image.Height);
maskBitmap.Save("output_mask.png");

// Apply mask to original image
var highlightedImage = SAMUtils.ApplyMaskToImage(image, result.Masks[0], Color.Red, 0.5f);
highlightedImage.Save("highlighted_result.png");
```

### MobileSAM for Lightweight Applications

```csharp
using var mobileSam = new MobileSAM(encoderPath, decoderPath);

var result = mobileSam.Segment(image, points, labels);
var maskBitmap = result.GetBestMaskAsBitmap();
```

### Using Bounding Box Prompts

```csharp
// Define a bounding box around the object
var boundingBox = new Rectangle(100, 100, 200, 300);

// Combine with point prompts for better accuracy
var result = sam2.Segment(image, points, labels, boundingBox);
```

### Video Frame Tracking (SAM2 only)

```csharp
// Process multiple frames with temporal consistency
for (int frameIndex = 0; frameIndex < videoFrames.Length; frameIndex++)
{
    var frame = videoFrames[frameIndex];
    var result = sam2.SegmentFrame(frame, points, labels, frameIndex);
    
    // Process result...
}

// Clear memory cache when done
sam2.ClearMemoryCache();
```

## API Reference

### SAM2 Class

Main class for SAM2 model inference with video support.

#### Constructor
```csharp
public SAM2(string encoderModelPath, string decoderModelPath)
```

#### Methods
```csharp
// Single image segmentation
public SAMResult Segment(Bitmap image, Point[] points, int[] labels, Rectangle? boundingBox = null)

// Video frame segmentation with temporal context
public SAMResult SegmentFrame(Bitmap image, Point[] points, int[] labels, int frameIndex, Rectangle? boundingBox = null)

// Clear temporal memory cache
public void ClearMemoryCache()
```

### MobileSAM Class

Lightweight version of SAM for resource-constrained environments.

```csharp
public MobileSAM(string encoderModelPath, string decoderModelPath)
public SAMResult Segment(Bitmap image, Point[] points, int[] labels, Rectangle? boundingBox = null)
```

### SAMResult Class

Container for segmentation results.

#### Properties
```csharp
public float[][,] Masks { get; set; }        // Generated masks
public float[] Scores { get; set; }          // Confidence scores
public int OriginalWidth { get; set; }       // Original image width
public int OriginalHeight { get; set; }      // Original image height
public int FrameIndex { get; set; }          // Frame index (SAM2)
```

#### Methods
```csharp
// Get the highest-scoring mask as bitmap
public Bitmap GetBestMaskAsBitmap(int? width = null, int? height = null)

// Get all masks as bitmaps
public Bitmap[] GetAllMasksAsBitmaps(int? width = null, int? height = null)
```

### SAMUtils Class

Utility functions for common operations.

```csharp
// Apply mask overlay to image
public static Bitmap ApplyMaskToImage(Bitmap originalImage, float[,] mask, Color maskColor, float alpha = 0.5f)

// Create point grid for automatic segmentation
public static Point[] CreatePointGrid(int width, int height, int gridSize)

// Create label arrays
public static int[] CreatePositiveLabels(int count)
public static int[] CreateNegativeLabels(int count)
```

## Model Files

You'll need ONNX model files for the encoder and decoder. These can be obtained from:

- **SAM2**: [Official SAM2 repository](https://github.com/facebookresearch/segment-anything-2)
- **MobileSAM**: [MobileSAM repository](https://github.com/ChaoningZhang/MobileSAM)

Supported model formats:
- SAM2: `sam2_hiera_tiny.encoder.onnx` / `sam2_hiera_tiny.decoder.onnx`
- SAM2: `sam2_hiera_small.encoder.onnx` / `sam2_hiera_small.decoder.onnx`
- SAM2: `sam2_hiera_base_plus.encoder.onnx` / `sam2_hiera_base_plus.decoder.onnx`
- MobileSAM: `mobile_sam.encoder.onnx` / `mobile_sam.decoder.onnx`

## System Requirements

- .NET 9.0 or later
- Windows, Linux, or macOS
- For GPU acceleration:
  - CUDA-compatible GPU (NVIDIA)
  - DirectML-compatible GPU (Windows)
  - OpenVINO-compatible hardware (Intel)

## Performance Tips

1. **GPU Acceleration**: The library automatically detects and uses available GPU acceleration
2. **Memory Management**: Dispose of SAM instances when done to free GPU memory
3. **Batch Processing**: For multiple images, reuse the same SAM instance
4. **Image Size**: Larger images may require more processing time and memory

## Error Handling

```csharp
try
{
    using var sam2 = new SAM2(encoderPath, decoderPath);
    var result = sam2.Segment(image, points, labels);
}
catch (FileNotFoundException ex)
{
    // Handle missing model files
    Console.WriteLine($"Model file not found: {ex.Message}");
}
catch (ArgumentException ex)
{
    // Handle invalid arguments (e.g., mismatched points/labels)
    Console.WriteLine($"Invalid input: {ex.Message}");
}
catch (Exception ex)
{
    // Handle other errors
    Console.WriteLine($"Segmentation failed: {ex.Message}");
}
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Meta AI for the original Segment Anything Model
- ONNX Runtime team for the inference engine
- The open-source community for various model implementations

## Support

For issues, questions, or feature requests, please open an issue on our [GitHub repository](https://github.com/agredo/SegmentAnything.Onnx/issues).