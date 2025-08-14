# SegmentAnything.Onnx - Complete Solution

A comprehensive .NET solution for advanced image segmentation using Segment Anything Model (SAM) and SAM2 with ONNX Runtime. This repository contains multiple projects demonstrating different use cases and implementations of the SAM models.

## 🏗️ Solution Structure

This solution contains three main projects:

### 📚 **SegmentAnything.Onnx** - Core Library
A high-performance .NET 9 library providing ONNX Runtime integration for SAM models.
- **Location**: `SegmentAnything.Onnx/`
- **Type**: Class Library (.NET 9)
- **Purpose**: Core functionality for SAM/SAM2 inference
- **NuGet Package**: `SegmentAnything.Onnx`

### 📱 **SegmentAnything.Onnx.Maui** - Mobile & Desktop App
A cross-platform .NET MAUI application demonstrating real-time image segmentation with camera integration.
- **Location**: `SegmentAnything.Onnx.Maui/`
- **Type**: .NET MAUI App (.NET 9)
- **Platforms**: Windows, Android, iOS, macOS
- **Features**: Live camera segmentation, real-time mask overlay

### 🖥️ **SegmentAnythingONNX** - Console Application
A console application for batch processing and testing SAM models.
- **Location**: `SegmentAnythingONNX/`
- **Type**: Console Application (.NET 9)
- **Purpose**: Command-line interface for image segmentation

## 🚀 Key Features

- **🔥 High Performance**: GPU acceleration support (CUDA, DirectML, OpenVINO)
- **🎯 Multiple Model Support**: SAM, SAM2, and MobileSAM implementations
- **🎥 Video Support**: SAM2 with temporal memory for video frame tracking
- **📍 Flexible Prompting**: Point-based prompts, bounding box prompts, and combinations
- **🛠️ Comprehensive Utilities**: Built-in image processing and mask manipulation tools
- **📱 Cross-Platform**: Works on Windows, Linux, macOS, Android, and iOS
- **📸 Real-Time Processing**: Live camera integration in MAUI app
- **⚡ Easy Integration**: Simple API designed for .NET applications

## 🎯 MAUI App Features

The SegmentAnything.Onnx.Maui project showcases:

- **📷 Live Camera Integration**: Uses CommunityToolkit.Maui.Camera for real-time image capture
- **🎯 Intelligent Segmentation**: Automatic person detection using strategic point placement
- **🖼️ Visual Feedback**: Real-time mask overlay with bounding box visualization
- **📱 Cross-Platform UI**: Runs on Windows, Android, iOS, and macOS
- **⚡ MVVM Pattern**: Clean architecture using CommunityToolkit.Mvvm

## 🛠️ Quick Start

### Prerequisites

- .NET 9 SDK or later
- Visual Studio 2022 17.8+ or Visual Studio Code
- For MAUI development: MAUI workload installed

### 1. Clone and Build

```bash
git clone https://github.com/your-repo/SegmentAnything.git
cd SegmentAnything
dotnet restore
dotnet build
```

### 2. Get SAM2 Model Files

Download ONNX model files and place them in `C:\Projects\Models\SAM\SAM2\`:
- `sam2_hiera_tiny.encoder.onnx`
- `sam2_hiera_tiny.decoder.onnx`

Model files can be obtained from:
- [Official SAM2 repository](https://github.com/facebookresearch/segment-anything-2)
- [Hugging Face SAM2 models](https://huggingface.co/facebook/sam2-hiera-tiny)

### 3. Run the Projects

#### MAUI App (Recommended)
```bash
cd SegmentAnything.Onnx.Maui
dotnet run --framework net9.0-windows10.0.19041.0
```

#### Console App
```bash
cd SegmentAnythingONNX
dotnet run
```

## 📦 Using the NuGet Package

Install the core library in your own projects:

```bash
dotnet add package SegmentAnything.Onnx
```

### Basic Usage Example

```csharp
using SegmentAnything.Onnx;
using System.Drawing;

// Initialize SAM2 with your ONNX model files
string encoderPath = @"C:\Projects\Models\SAM\SAM2\sam2_hiera_tiny.encoder.onnx";
string decoderPath = @"C:\Projects\Models\SAM\SAM2\sam2_hiera_tiny.decoder.onnx";

using var sam2 = new SAM2(encoderPath, decoderPath);

// Load your image
var image = new Bitmap("path/to/your/image.jpg");

// Define segmentation points (positive and negative)
var points = new Point[] 
{ 
    new Point(400, 300),  // Positive point (inside object)
    new Point(100, 100)   // Negative point (outside object)
};
var labels = new int[] { 1, 0 };  // 1=positive, 0=negative

// Perform segmentation
var result = sam2.Segment(image, points, labels);

// Apply mask to image with blue overlay
var maskedImage = SAMUtils.ApplyMaskToImage(image, result.Masks[0], Color.Blue, 0.35f);
maskedImage.Save("result.png");
```

## 🏗️ Architecture Overview

### Core Library Components

```
SegmentAnything.Onnx/
├── SAM2.cs              # Main SAM2 implementation with video support
├── MobileSAM.cs         # Lightweight SAM implementation
├── SAMModelBase.cs      # Base class for all SAM models
├── SAMResult.cs         # Container for segmentation results
├── SAMUtils.cs          # Utility functions for image processing
├── EncoderOutputs.cs    # Encoder output container
└── SAMMemoryState.cs    # Memory state for video tracking
```

### MAUI App Components

```
SegmentAnything.Onnx.Maui/
├── MainPage.xaml        # UI with camera view and controls
├── MainPage.xaml.cs     # Code-behind for UI interactions
├── ViewModels/
│   └── MainPageViewModel.cs  # MVVM logic for segmentation
└── Platforms/           # Platform-specific implementations
```

## 🎯 Advanced Segmentation Strategies

The MAUI app demonstrates an advanced segmentation approach:

1. **Strategic Point Placement**: Places points at image edges and center
2. **Bounding Box Integration**: Uses nearly full-image bounding box for context
3. **Positive/Negative Labels**: Combines edge negatives with center positive
4. **Real-time Processing**: Processes camera frames as they're captured

```csharp
// Strategic point placement for person segmentation
var points = new Point[]
{
    new Point(imageWidth/2, (int)(imageHeight * 0.01)),    // Top edge (negative)
    new Point(imageWidth/2, (int)(imageHeight * 0.98)),    // Bottom edge (negative)  
    new Point((int)(imageWidth * 0.01), imageHeight/2),    // Left edge (negative)
    new Point((int)(imageWidth * 0.98), imageHeight/2),    // Right edge (negative)
    new Point(imageWidth/2, imageHeight/2)                 // Center (positive)
};
var labels = new int[] { 0, 0, 0, 0, 1 };  // Edges negative, center positive
```

## 📱 Platform Support

### MAUI App Platform Support
- ✅ **Windows** 10.0.17763.0+
- ✅ **Android** API 21+
- ✅ **iOS** 15.0+
- ✅ **macOS** 13.0+ (Mac Catalyst)

### Core Library Platform Support
- ✅ **Windows** (x64, ARM64)
- ✅ **Linux** (x64, ARM64)
- ✅ **macOS** (x64, ARM64)

## 🔧 Development Setup

### Required Visual Studio Workloads
- .NET Multi-platform App UI development
- .NET desktop development
- Mobile development with .NET

### Required NuGet Packages (Already Included)
- `Microsoft.ML.OnnxRuntime.QNN` - ONNX Runtime with hardware acceleration
- `CommunityToolkit.Maui` - MAUI community extensions
- `CommunityToolkit.Maui.Camera` - Camera integration
- `CommunityToolkit.Mvvm` - MVVM toolkit
- `System.Drawing.Common` - Image processing

## 🎯 Performance Optimization

### GPU Acceleration
The library automatically detects and configures the best available execution provider:
1. **CUDA** (NVIDIA GPUs)
2. **DirectML** (Windows GPUs)
3. **OpenVINO** (Intel hardware)
4. **CPU** (fallback)

### Memory Management
- Dispose of SAM instances to free GPU memory
- Reuse SAM instances for batch processing
- Clear video memory cache when switching sequences

## 🐛 Troubleshooting

### Common Issues

**Model Files Not Found**
- Ensure ONNX model files are in the correct directory
- Update paths in `MainPageViewModel.cs` if using different locations

**GPU Memory Issues**
- Reduce image resolution
- Dispose of SAM instances after use
- Check available GPU memory

**Platform-Specific Issues**
- Ensure proper platform workloads are installed
- Check minimum platform version requirements

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta AI** for the original Segment Anything Model
- **Microsoft** for ONNX Runtime and .NET MAUI
- **CommunityToolkit** maintainers for MAUI extensions
- **The open-source community** for various model implementations

## 📞 Support

For issues, questions, or feature requests:
- 🐛 [GitHub Issues](https://github.com/agredo/SegmentAnything.Onnx/issues)
- 💬 [Discussions](https://github.com/agredo/SegmentAnything.Onnx/discussions)
- 📧 Contact: support@agredo.com

---

**⭐ If this project helps you, please give it a star!**