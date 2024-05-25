use burn::backend::wgpu::WgpuDevice;
use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{
        BinFileRecorder, CompactRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder,
    },
    tensor::backend::AutodiffBackend,
    tensor::{activation, backend::Backend, Bool, Device, ElementConversion, Int, Tensor},
};
use burn_tensor::Data;
use image::ImageBuffer;
use ndarray::{s, Array, Array1, Array2, Array3, Axis};
use std::env;

use crate::model::Model;

pub fn load_model<B: Backend>(device: &B::Device) -> Model<B> {
    let current_dir = env::current_dir().expect("Failed to get current directory");
    let model_dir = current_dir.join("artifacts/");

    // Load model in full precision from MessagePack file
    let model_path = model_dir.join("model.bin");
    let recorder = CompactRecorder::new();

    let model = Model::new(device);
    let model = model
        .load_file(model_path, &recorder, device)
        .expect("Should be able to load the model weights from the provided file");

    model
}

pub fn infer<B: Backend>(device: &B::Device) {
    let model = load_model(device);

    let z_mean = Tensor::<B, 1>::from_floats([0.0, 0.0], &device);
    let z_var = Tensor::<B, 1>::from_floats([0.1, 0.1], &device);
    let image = model.infer(z_mean, z_var);

    // remove batch, channel
    let image: Tensor<B, 3> = image.squeeze(0);
    let image = image.clone() * 255.;

    let arr = Array2::from_shape_vec(
        (image.dims()[1] as usize, image.dims()[2] as usize),
        image.clone().into_data().convert::<f32>().value,
    )
    .unwrap();

    let image_buffer =
        ImageBuffer::from_fn(image.dims()[0] as u32, image.dims()[1] as u32, |x, y| {
            let pixel = arr[[y as usize, x as usize]];
            image::Luma([pixel as u16])
        });

    image_buffer.save(format!("./images/result.png")).unwrap();
}
