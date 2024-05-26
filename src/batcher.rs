use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
};

#[derive(Clone, Debug)]
pub struct MnistBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<MnistItem, MnistBatch<B>> for MnistBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| {
                Tensor::<B, 2>::from_data(data.convert(), &self.device).unsqueeze_dims(&[0, 1])
            })
            // normalize to [0, 1]
            .map(|tensor| tensor / 255.0)
            .collect();

        let pad_size = (32 - 28) / 2; 
        let targets = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| {
                Tensor::<B, 2>::from_data(data.convert(), &self.device).unsqueeze_dims(&[0, 1])
            })
            // normalize to [0, 1]
            .map(|tensor| tensor / 255.0)
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        MnistBatch { images, targets }
    }
}
