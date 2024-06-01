use burn::{
    module::Module,
    record::{
        BinFileRecorder, CompactRecorder, FullPrecisionSettings, NamedMpkFileRecorder,
        NoStdTrainingRecorder, Recorder,
    },
    tensor::{backend::Backend, Bool, Device, Distribution, ElementConversion, Int, Tensor},
};
use std::env;

use crate::model::{reconstruction_to_image, Model};

pub fn load_model<B: Backend>(device: &B::Device) -> Model<B> {
    let current_dir = env::current_dir().expect("Failed to get current directory");
    let model_dir = current_dir.join("artifacts/");

    // Load model in full precision from MessagePack file
    let model_path = model_dir.join("model.mpk");

    let recorder: NamedMpkFileRecorder<burn::record::HalfPrecisionSettings> =
        CompactRecorder::new();
    // let recorder = NoStdTrainingRecorder::new();

    let model = Model::new(device);
    let model = model
        .load_file(model_path, &recorder, device)
        .expect("Should be able to load the model weights from the provided file");

    model
}

pub fn infer<B: Backend>(device: &B::Device) {
    let model: Model<B> = load_model(device);

    let embedding_dim = 256;
    let z = Tensor::random([1, embedding_dim], Distribution::Normal(-1., 1.), device);
    let image = model.decoder.forward(z.clone());

    let file_name = format!("./images/result.png");
    reconstruction_to_image(image, file_name);
}
