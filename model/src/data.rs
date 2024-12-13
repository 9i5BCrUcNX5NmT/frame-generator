use burn::{data::dataloader::batcher::Batcher, prelude::*};
use image::{DynamicImage, GenericImageView};

use crate::images::ImagePixelData;

#[derive(Clone)]
pub struct FrameBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> FrameBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct FrameBatch<B: Backend> {
    images: Tensor<B, 4>,
    targets: Tensor<B, 4>,
}

impl<B: Backend> Batcher<ImagePixelData, FrameBatch<B>> for FrameBatcher<B> {
    fn batch(&self, images: Vec<ImagePixelData>) -> FrameBatch<B> {
        let images = images
            .iter()
            .map(|image| TensorData::from(image.pixels).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 3>::from_data(data, &self.device))
            .map(|tensor| tensor.reshape([1, 4, 200, 200]))
            // Простая нормализация
            .map(|tensor| tensor / 255)
            .collect();

        // let images = Tensor::cat(images, 0).to_device(&self.device);
        // let targets = Tensor::cat(images, 0).to_device(&self.device);

        let images = Tensor::cat(images, 0).to_device(&self.device);

        let targets = (images
            .clone()
            .iter_dim(0)
            .skip(1)
            .chain(images.clone().iter_dim(0).take(1)))
        .collect();

        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        FrameBatch { images, targets }
    }
}
