use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};

#[derive(Clone, Debug)]
pub struct ImageDatasetBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct ImageDatasetBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> ImageDatasetBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<MnistItem, ImageDatasetBatch<B>> for ImageDatasetBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>) -> ImageDatasetBatch<B> {
        // need extra padding for input images
        let pad_size = (32 + 4 - 28) / 2 - 2;
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| {
                Tensor::<B, 2>::from_data(data.convert(), &self.device).unsqueeze_dims(&[0, 1])
            })
            // normalize to [0, 1]
            .map(|tensor| tensor / 255.0)
            // NOTE, can use 0.0f32.elem() to convert f32 to
            .map(|tensor| tensor.pad((pad_size, pad_size, pad_size, pad_size), 0.0f32.elem()))
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
            .map(|tensor| tensor.pad((pad_size, pad_size, pad_size, pad_size), 0.0f32.elem()))
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        ImageDatasetBatch { images, targets }
    }
}
