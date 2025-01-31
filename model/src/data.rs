use burn::{data::dataloader::batcher::Batcher, prelude::*};

use crate::{csv_processing::KeysRecord, images::ImagePixelData, types::MyData};

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
    pub inputs: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> Batcher<MyData, FrameBatch<B>> for FrameBatcher<B> {
    fn batch(&self, mydata: Vec<MyData>) -> FrameBatch<B> {
        let images = mydata
            .iter()
            .map(|data| TensorData::from(data.image.pixels).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 3>::from_data(data, &self.device))
            // 1 штука, 4 параметра цвета, 200 на 200 размер
            .map(|tensor| tensor.reshape([1, 4, 200, 200]))
            // Простая нормализация
            .map(|tensor| tensor / 255)
            .collect();

        // let inputs = Tensor::cat(inputs, 0).to_device(&self.device);
        // let targets = Tensor::cat(inputs, 0).to_device(&self.device);

        let images = Tensor::cat(images, 0).to_device(&self.device);

        let inputs: Tensor<B, 4> = todo!();

        // Сдвинутые изображения на 1
        // TODO: Изменить?
        let targets = inputs
            .clone()
            .iter_dim(0)
            .skip(1)
            .chain(inputs.clone().iter_dim(0).take(1))
            .collect();

        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        FrameBatch { inputs, targets }
    }
}
