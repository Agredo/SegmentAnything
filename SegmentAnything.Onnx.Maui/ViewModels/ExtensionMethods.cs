namespace SegmentAnything.Onnx.Maui.ViewModels;

public static class ExtensionMethods
{
    public static async void Await(this Task task)
    {
        await task;
    }
}
