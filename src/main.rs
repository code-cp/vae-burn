use burn::backend::wgpu::WgpuDevice;

type Backend = burn::backend::Autodiff<burn::backend::Wgpu>;

use vae_burn::infer::infer;
use vae_burn::train::train;

fn main() {
    let device = WgpuDevice::default();
    // train::<Backend>(&device);
    infer::<Backend>(&device);
}
