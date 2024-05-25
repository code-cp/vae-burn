use burn::backend::wgpu::WgpuDevice;
use burn_tensor::Data;
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
use ndarray::{Array, Array1, Array2, Array3, s, Axis};
use std::env;
use image::{ImageBuffer};

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
    let image = image.squeeze(0).squeeze(0);
    let image = image * 255.;

    let shape = image.dims();
    let arr = Array3::from_shape_vec((1, shape[0] as usize, shape[1] as usize), image.into_data().value.to_vec()).unwrap();

    let image_buffer = ImageBuffer::from_fn(shape[0] as u32, shape[1] as u32, |x, y| {
        let pixel = arr[[y as usize, x as usize]];
        image::Luma([pixel])
    });

    image_buffer.save(format!("./images/result.png")).unwrap();
}