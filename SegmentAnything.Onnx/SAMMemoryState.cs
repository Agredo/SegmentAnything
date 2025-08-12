using Microsoft.ML.OnnxRuntime.Tensors;

namespace SegmentAnything.Onnx;

public class SAMMemoryState
{
    public Dictionary<string, DenseTensor<float>> Memory { get; set; } = new();
    public int LastFrameIndex { get; set; }
}
