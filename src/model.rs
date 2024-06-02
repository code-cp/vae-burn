#[cfg(feature = "custom")]
use crate::custom_batcher::ImageDatasetBatch;
#[cfg(feature = "mnist")]
use crate::mnist_batcher::ImageDatasetBatch;

use burn::nn;
use burn::{
    config::Config,
    module::Module,
    nn::loss::{MseLoss, Reduction::Mean, Reduction::Sum},
    tensor::{
        activation,
        backend::{AutodiffBackend, Backend},
        Distribution, Tensor,
    },
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use image::ImageBuffer;
use ndarray::{s, Array, Array1, Array2, Array3, Axis};
use std::fs::File;
use std::io::Write;

#[derive(Config)]
pub struct EncoderConfig {
    #[config(default = 32)]
    conv1_channel_out: usize,
    #[config(default = 64)]
    conv2_channel_out: usize,
    #[config(default = 128)]
    conv3_channel_out: usize,
    #[config(default = 32)]
    image_size: usize,
    embedding_dim: usize,
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    conv1: nn::conv::Conv2d<B>,
    conv2: nn::conv::Conv2d<B>,
    conv3: nn::conv::Conv2d<B>,
    linear_mean: nn::Linear<B>,
    linear_var: nn::Linear<B>,
    activation: nn::Relu,
}

impl EncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        let kernel_size = [3, 3];

        let conv1 = nn::conv::Conv2dConfig::new([1, self.conv1_channel_out], kernel_size)
            .with_padding(nn::PaddingConfig2d::Explicit(1, 1))
            .with_stride([2, 2])
            .init(device);
        let conv2 = nn::conv::Conv2dConfig::new(
            [self.conv1_channel_out, self.conv2_channel_out],
            kernel_size,
        )
        .with_padding(nn::PaddingConfig2d::Explicit(1, 1))
        .with_stride([2, 2])
        .init(device);
        let conv3 = nn::conv::Conv2dConfig::new(
            [self.conv2_channel_out, self.conv3_channel_out],
            kernel_size,
        )
        .with_padding(nn::PaddingConfig2d::Explicit(1, 1))
        .with_stride([2, 2])
        .init(device);

        let linear_mean =
            nn::LinearConfig::new(128 * (self.image_size / 8).pow(2), self.embedding_dim)
                .init(device);
        let linear_var =
            nn::LinearConfig::new(128 * (self.image_size / 8).pow(2), self.embedding_dim)
                .init(device);

        Encoder {
            conv1,
            conv2,
            conv3,
            linear_mean,
            linear_var,
            activation: nn::Relu::new(),
        }
    }
}

pub fn sampling<B: Backend>(z_mean: Tensor<B, 2>, z_var: Tensor<B, 2>) -> Tensor<B, 2> {
    let device = B::Device::default();
    let epsilon = Tensor::random(
        z_mean.clone().shape(),
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    z_mean + z_var.mul_scalar(0.5).exp() * epsilon
}

pub struct LatentTensors<B: Backend> {
    z_mean: Tensor<B, 2>,
    z_var: Tensor<B, 2>,
    z: Tensor<B, 2>,
}

impl<B: Backend> Encoder<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> LatentTensors<B> {
        let x = self.conv1.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv3.forward(x);
        let x = self.activation.forward(x);

        // x shape is batch x 128 x 4 x 4
        let x: Tensor<B, 2> = x.flatten(1, 3);

        let z_mean = self.linear_mean.forward(x.clone());
        let z_var = self.linear_var.forward(x.clone());
        let z = sampling(z_mean.clone(), z_var.clone());

        LatentTensors { z_mean, z_var, z }
    }
}

#[derive(Config)]
struct DecoderConfig {
    embedding_dim: usize,
    shape_before_flattening: Vec<usize>,
    deconv1_channel_in: usize,
    deconv1_channel_out: usize,
    deconv2_channel_out: usize,
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    deconv1: nn::conv::ConvTranspose2d<B>,
    deconv2: nn::conv::ConvTranspose2d<B>,
    deconv3: nn::conv::ConvTranspose2d<B>,
    linear_fc: nn::Linear<B>,
    shape_before_flattening: Vec<usize>,
    activation: nn::Relu,
}

impl DecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Decoder<B> {
        let kernel_size = [3, 3];

        let linear_fc = nn::LinearConfig::new(
            self.embedding_dim,
            self.shape_before_flattening[0]
                * self.shape_before_flattening[1]
                * self.shape_before_flattening[1],
        )
        .init(device);

        let deconv1 = nn::conv::ConvTranspose2dConfig::new(
            [self.deconv1_channel_in, self.deconv1_channel_out],
            kernel_size,
        )
        .with_stride([2, 2])
        .with_padding([1, 1])
        .with_padding_out([1, 1])
        .init(device);
        let deconv2 = nn::conv::ConvTranspose2dConfig::new(
            [self.deconv1_channel_out, self.deconv2_channel_out],
            kernel_size,
        )
        .with_stride([2, 2])
        .with_padding([1, 1])
        .with_padding_out([1, 1])
        .init(device);
        let deconv3 =
            nn::conv::ConvTranspose2dConfig::new([self.deconv2_channel_out, 1], kernel_size)
                .with_stride([2, 2])
                .with_padding([1, 1])
                .with_padding_out([1, 1])
                .init(device);

        Decoder {
            deconv1,
            deconv2,
            deconv3,
            linear_fc,
            shape_before_flattening: self.shape_before_flattening.clone(),
            activation: nn::Relu::new(),
        }
    }
}

impl<B: Backend> Decoder<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 4> {
        let x = self.linear_fc.forward(x);

        let batch_size = x.dims()[0];
        let x = x.reshape([
            batch_size,
            self.shape_before_flattening[0],
            self.shape_before_flattening[1],
            self.shape_before_flattening[2],
        ]);

        let x = self.deconv1.forward(x);
        let x = self.activation.forward(x);
        let x = self.deconv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.deconv3.forward(x);
        let x = activation::sigmoid(x);

        x
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    encoder: Encoder<B>,
    pub decoder: Decoder<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Model<B> {
        let image_size = 32;
        let embedding_dim = 256;

        let conv1_channel_out = 32;
        let conv2_channel_out = 64;
        let conv3_channel_out = 128;

        let encoder_config = EncoderConfig {
            conv1_channel_out,
            conv2_channel_out,
            conv3_channel_out,
            image_size,
            embedding_dim,
        };
        let encoder = encoder_config.init(device);

        let shape_before_flattening = vec![128, image_size / 8, image_size / 8];
        let deconv1_channel_in = conv3_channel_out;
        let deconv1_channel_out = conv2_channel_out;
        let deconv2_channel_out = conv1_channel_out;
        let decoder_config = DecoderConfig {
            embedding_dim,
            shape_before_flattening,
            deconv1_channel_in,
            deconv1_channel_out,
            deconv2_channel_out,
        };
        let decoder = decoder_config.init(device);

        Self { encoder, decoder }
    }

    pub fn infer(&self, z_mean: Tensor<B, 2>, z_var: Tensor<B, 2>) -> Tensor<B, 4> {
        let z = sampling(z_mean.clone(), z_var.clone());
        self.decoder.forward(z)
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, LatentTensors<B>) {
        let latent_tensors = self.encoder.forward(x);
        let reconstruction = self.decoder.forward(latent_tensors.z.clone());
        (reconstruction, latent_tensors)
    }

    pub fn forward_loss(&self, item: ImageDatasetBatch<B>) -> RegressionOutput<B> {
        let targets: Tensor<B, 4> = item.targets.clone();
        let outputs = self.forward(item.images.clone());

        let reconstruction = outputs.0.clone();
        let z_var = outputs.1.z_var.clone();
        let z_mean = outputs.1.z_mean.clone();

        // RegressionOutput can only accept Tensor<B, 2>, not Tensor<B, 4>
        let image_size = reconstruction.dims()[2];
        let batch_size = reconstruction.dims()[0] * reconstruction.dims()[1];
        let output = reconstruction.reshape([batch_size, image_size * image_size]);

        let targets = targets.reshape([batch_size, image_size * image_size]);

        let reconstruction_loss = MseLoss::new().forward(output.clone(), targets.clone(), Sum);
        let kl_divergence_loss =
            (z_var.clone().add_scalar(1.) - z_mean.clone().powf_scalar(2.) - z_var.clone().exp())
                .mul_scalar(-0.5);
        let loss = reconstruction_loss + kl_divergence_loss.mean();

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

pub fn reconstruction_to_image<B: Backend>(image: Tensor<B, 4>, file_name: String) {
    // remove batch
    let image: Tensor<B, 3> = image.squeeze(0);
    let image = image.clone() * 255.;

    let arr = Array2::from_shape_vec(
        (image.dims()[1] as usize, image.dims()[2] as usize),
        image.clone().into_data().convert::<f32>().value,
    )
    .unwrap();

    let image_buffer =
        ImageBuffer::from_fn(image.dims()[1] as u32, image.dims()[2] as u32, |x, y| {
            let pixel = arr[[y as usize, x as usize]];
            image::Luma([pixel as u8])
        });

    image_buffer.save(file_name).unwrap();
}

impl<B: AutodiffBackend> TrainStep<ImageDatasetBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: ImageDatasetBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_loss(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<ImageDatasetBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: ImageDatasetBatch<B>) -> RegressionOutput<B> {
        self.forward_loss(item)
    }
}
