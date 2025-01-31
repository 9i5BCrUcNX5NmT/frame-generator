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
            // Простая нормализация цветов
            .map(|tensor| tensor / 255)
            .collect();

        // let inputs = Tensor::cat(inputs, 0).to_device(&self.device);
        // let targets = Tensor::cat(inputs, 0).to_device(&self.device);

        let images = Tensor::cat(images, 0).to_device(&self.device);

        let keys = mydata
            .iter()
            .map(|data| {
                let mut keys_vector = [0; 200];

                for i in &data.key.keys {
                    keys_vector[*i as usize] += 1;
                }

                keys_vector
            })
            .map(|vector| TensorData::from(vector).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
            // 1 штука, 4 параметра цвета, 200 на 200 размер
            .map(|tensor| tensor.reshape([1, 1, 1, 200]))
            // // Простая нормализация
            // .map(|tensor| tensor / 255)
            .collect();

        let keys = Tensor::cat(keys, 0);

        let mouse = mydata
            .iter()
            .map(|data| {
                let mut mouse_vector = [[0; 2]; 200]; // Может не хватить, тк в коде нет ограничений на количество передвижений мыши

                for (i, value) in data.key.mouse.iter().enumerate() {
                    mouse_vector[i as usize] = *value;
                }

                mouse_vector
            })
            .map(|vector| TensorData::from(vector).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
            // 1 штука, 4 параметра цвета, 200 на 200 размер
            .map(|tensor| tensor.reshape([1, 1, 2, 200]))
            // // Простая нормализация
            // .map(|tensor| tensor / 255)
            .collect();

        let mouse = Tensor::cat(mouse, 0);

        let inputs = Tensor::cat(vec![keys, mouse], 0).to_device(&self.device);
        let inputs = inputs.reshape([1, 1, 3, 200]); // на всякий случай
        todo!("Как соединить images и mouse, keys");

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
