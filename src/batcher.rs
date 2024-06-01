use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
};

use crate::custom_dataset::ImageDatasetItem; 

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

impl<B: Backend> Batcher<ImageDatasetItem, ImageDatasetBatch<B>> for ImageDatasetBatcher<B> {
    fn batch(&self, items: Vec<ImageDatasetItem>) -> ImageDatasetBatch<B> {
        let image_size = 32; 

        let images = items
            .iter()
            .map(|item| Data::new(item.pixels.clone(), Shape::new([image_size, image_size])))
            .map(|data| {
                Tensor::<B, 2>::from_data(data.convert(), &self.device).unsqueeze_dims(&[0, 1])
            })
            // normalize to [0, 1]
            .map(|tensor| tensor / 255.0)
            .collect();

        let targets = items
            .iter()
            .map(|item| Data::new(item.pixels.clone(), Shape::new([image_size, image_size])))
            .map(|data| {
                Tensor::<B, 2>::from_data(data.convert(), &self.device).unsqueeze_dims(&[0, 1])
            })
            // normalize to [0, 1]
            .map(|tensor| tensor / 255.0)
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        ImageDatasetBatch { images, targets }
    }
}
