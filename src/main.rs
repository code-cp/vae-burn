use burn::backend::wgpu::WgpuDevice;
use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::optim::decay::WeightDecayConfig;

type Backend = burn::backend::Autodiff<burn::backend::Wgpu>;

use vae_burn::train::train;
use vae_burn::infer::infer;
    
fn main() {
    let device = WgpuDevice::default();
    train(&device);
    infer(&device);
}