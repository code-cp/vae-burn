use burn::{
    module::Module,
    record::{
        BinFileRecorder, NoStdTrainingRecorder, CompactRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder,
    },
    tensor::backend::AutodiffBackend,
    tensor::{activation, backend::Backend, Bool, Device, ElementConversion, Int, Tensor},
};
use image::ImageBuffer;
use ndarray::{s, Array, Array1, Array2, Array3, Axis};
use std::env;

use crate::model::Model;

pub fn load_model<B: Backend>(device: &B::Device) -> Model<B> {
    let current_dir = env::current_dir().expect("Failed to get current directory");
    let model_dir = current_dir.join("artifacts/");

    // Load model in full precision from MessagePack file
    let model_path = model_dir.join("model.bin");

    // let recorder: NamedMpkFileRecorder<burn::record::HalfPrecisionSettings> = CompactRecorder::new();
    let recorder = NoStdTrainingRecorder::new();

    let model = Model::new(device);
    let model = model
        .load_file(model_path, &recorder, device)
        .expect("Should be able to load the model weights from the provided file");

    model
}

pub fn infer<B: Backend>(device: &B::Device) {
    let model = load_model(device);

    let z_mean = Tensor::<B, 1>::from_floats([0.0, 0.0], &device).unsqueeze_dim(0);
    let z_var = Tensor::<B, 1>::from_floats([0.1, 0.1], &device).unsqueeze_dim(0);
    let image = model.infer(z_mean, z_var);

    // remove batch, channel
    let image: Tensor<B, 3> = image.squeeze(0);
    let image = image.clone() * 255.;

    let arr = Array3::from_shape_vec(
        (image.dims()[0] as usize, image.dims()[1] as usize, image.dims()[2] as usize),
        image.clone().into_data().convert::<f32>().value,
    )
    .unwrap();

    let image_buffer =
        ImageBuffer::from_fn(image.dims()[1] as u32, image.dims()[2] as u32, |x, y| {
            let r = arr[[0, y as usize, x as usize]];
            let g = arr[[1, y as usize, x as usize]];
            let b = arr[[2, y as usize, x as usize]];
            image::Rgb([r, g, b])
        });

    image_buffer.save(format!("./images/result.png")).unwrap();
}
