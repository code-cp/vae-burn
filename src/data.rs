use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
    tensor::{ops::FloatElem, ElementConversion},
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
        // need extra padding for input images
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| {
                Tensor::<B, 2>::from_data(data.convert(), &self.device).unsqueeze_dims(&[0, 1])
            })
            // normalize to [0, 1]
            .map(|tensor| tensor / 255.0)
            // NOTE, can use 0.0f32.elem() to convert f32 to
            .map(|tensor| tensor.pad((2, 2, 2, 2), 0.0f32.elem()))
            .collect();

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
