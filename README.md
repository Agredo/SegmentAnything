# SegmentAnything.Onnx

A high-performance .NET library for running Segment Anything Model (SAM) and SAM2 using ONNX Runtime. Provides easy-to-use APIs for advanced image segmentation with support for multiple SAM model variants.

## ✨ Features

- 🚀 **High Performance**: GPU acceleration support (CUDA, DirectML, OpenVINO)
- 🎯 **Multiple Models**: SAM, SAM2, and MobileSAM implementations
- 🎥 **Video Support**: SAM2 with temporal memory for video frame tracking
- 📍 **Flexible Prompting**: Point-based and bounding box prompts
- 🛠️ **Comprehensive Utils**: Built-in image processing and mask manipulation
- 💻 **Easy Integration**: Simple .NET API

## 🚀 Quick Start

### Installation
```bash
dotnet add package SegmentAnything.Onnx
```

### Basic Usage
```csharp
using SegmentAnything.Onnx;
using System.Drawing;

// Initialize SAM2
using var sam2 = new SAM2("encoder.onnx", "decoder.onnx");

// Load image and define prompts
var image = new Bitmap("image.jpg");
var points = new Point[] { new Point(400, 300) };
var labels = new int[] { 1 }; // 1 = positive, 0 = negative

// Segment and save results
var result = sam2.Segment(image, points, labels);
var mask = result.GetBestMaskAsBitmap();
mask.Save("output_mask.png");
```

### Video Frame Tracking (SAM2)
```csharp
// Process video frames with temporal consistency
for (int i = 0; i < frames.Length; i++)
{
    var result = sam2.SegmentFrame(frames[i], points, labels, i);
    // Process result...
}
sam2.ClearMemoryCache();
```

## 📋 Requirements

- .NET 9.0+
- ONNX model files (SAM2, MobileSAM)
- Optional: GPU for acceleration

## 🏗️ Project Structure

- **SegmentAnything.Onnx** - Main library with SAM implementations
- **SegmentAnythingONNX** - Example console application

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📚 Documentation

For detailed API documentation and examples, see the [library README](SegmentAnything.Onnx/README.md).