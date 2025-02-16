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

    fn extract_keys(&self, mydata: &Vec<MyData>) -> Tensor<B, 4> {
        let keys = mydata
            .iter()
            .map(|data| {
                let mut keys_vector = [0; 200];

                for i in &data.keys.keys {
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
        keys
    }

    fn extract_mouse(&self, mydata: &Vec<MyData>) -> Tensor<B, 4> {
        let mouse = mydata
            .iter()
            .map(|data| {
                let mut mouse_vector = [[0; 2]; 200]; // Может не хватить, тк в коде нет ограничений на количество передвижений мыши

                for (i, value) in data.keys.mouse.iter().enumerate() {
                    mouse_vector[i as usize] = *value;
                }

                mouse_vector
            })
            .map(|vector| TensorData::from(vector).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, &self.device))
            // 1 штука, 4 параметра цвета, 200 на 200 размер
            .map(|tensor| tensor.reshape([1, 1, 2, 200]))
            // // Простая нормализация
            // .map(|tensor| tensor / 255)
            .collect();

        Tensor::cat(mouse, 0)
    }

    fn extract_targets(&self, images: &Tensor<B, 4>) -> Tensor<B, 4> {
        let targets = images
            .clone()
            .iter_dim(0)
            .skip(1)
            .chain(images.clone().iter_dim(0).take(1))
            .collect();

        Tensor::cat(targets, 0).to_device(&self.device)
    }

    fn extract_images(&self, mydata: &Vec<MyData>) -> Tensor<B, 4> {
        let images = mydata
            .iter()
            .map(|data| TensorData::from(data.image.pixels).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 3>::from_data(data, &self.device))
            // 1 штука, 4 параметра цвета, 200 на 200 размер
            .map(|tensor| tensor.reshape([1, 4, 200, 200]))
            // Простая нормализация цветов
            .map(|tensor| tensor / 255)
            .collect();

        Tensor::cat(images, 0)
    }
}

#[derive(Clone, Debug)]
pub struct FrameBatch<B: Backend> {
    pub inputs: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> Batcher<MyData, FrameBatch<B>> for FrameBatcher<B> {
    fn batch(&self, mydata: Vec<MyData>) -> FrameBatch<B> {
        let images = self.extract_images(&mydata);
        let keys = self.extract_keys(&mydata);
        let mouse = self.extract_mouse(&mydata);

        let inputs = Tensor::cat(vec![images.clone(), keys, mouse], 1).to_device(&self.device);

        // Сдвинутые изображения на 1
        // TODO: Изменить?
        let targets = self.extract_targets(&images);

        FrameBatch { inputs, targets }
    }
}
