using SegmentAnything.Onnx.Maui.ViewModels;

namespace SegmentAnything.Onnx.Maui
{
    public partial class MainPage : ContentPage
    {
        public MainPage(MainPageViewModel viewModel)
        {
            InitializeComponent();

            BindingContext = viewModel;
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
