using SegmentAnything.Onnx.Maui.ViewModels;
using System.Diagnostics;
using System.Linq;

namespace SegmentAnything.Onnx.Maui
{
    public partial class MainPage : ContentPage
    {
        public MainPage(MainPageViewModel viewModel)
        {
            InitializeComponent();

            BindingContext = viewModel;

            this.Loaded += MyCamera_Loaded;
        }

        private async void MyCamera_Loaded(object? sender, EventArgs e)
        {
            MyCamera.SelectedCamera = (await MyCamera.GetAvailableCameras(new CancellationToken()))[0];

            CommunityToolkit.Maui.Core.CameraInfo? selectedCamera = MyCamera.SelectedCamera;
            IList<Size> suportedResolutions = selectedCamera.SupportedResolutions.OrderBy(r => r.Width).ThenBy(r => r.Height).Select(r => r).ToList();

            //debug print supported resolutions
            foreach (var resolution in suportedResolutions)
            {
                Debug.WriteLine($"Supported Resolution: {resolution.Width}x{resolution.Height}");
            }

            if (suportedResolutions.Count > 0)
            {
                //min 1024 width the height must be greater than 1024
                Size size = suportedResolutions.First(size => size.Width >= 1024 && size.Height > 1024 && size.Width > size.Height);

                MyCamera.ImageCaptureResolution = size;
            }
        }

        private async void CaptureButton_Clicked(object sender, EventArgs e)
        {


            var image = await MyCamera.CaptureImage(new CancellationToken());
            if (image != null)
            {
                ((MainPageViewModel)BindingContext).Image = image;
                ((MainPageViewModel)BindingContext).MaskImage();

                // Display the captured image in an Image control or process it as needed
                DisplayAlert("Media Captured", "Image captured successfully!", "OK");
            }
            else
            {
                DisplayAlert("Error", "No media file captured.", "OK");
            }
        }
    }
}
