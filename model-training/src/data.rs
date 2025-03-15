use burn::{data::dataloader::batcher::Batcher, prelude::*};
use common::{HEIGHT, MOUSE_VECTOR_LENGTH, WIDTH};
use preprocessor::types::MyConstData;

#[derive(Clone)]
pub struct FrameBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> FrameBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    fn extract_const_keys(&self, mydata: &Vec<MyConstData>) -> Tensor<B, 2> {
        let keys = mydata
            .iter()
            .map(|data| {
                let mut keys_vector = [0; 108];

                for i in &data.keys_record.keys {
                    keys_vector[*i as usize] += 1;
                }

                keys_vector
            })
            .map(|vector| TensorData::from(vector).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
            .map(|tensor| tensor.reshape([1 as usize, 108]))
            // // Простая нормализация
            // .map(|tensor| tensor / 255)
            .collect();

        Tensor::cat(keys, 0)
    }

    fn extract_const_mouse(&self, mydata: &Vec<MyConstData>) -> Tensor<B, 3> {
        let mouse = mydata
            .iter()
            .map(|data| {
                let mut mouse_vector: [[i32; 2]; MOUSE_VECTOR_LENGTH] =
                    [[0; 2]; MOUSE_VECTOR_LENGTH]; // Может не хватить, тк в коде нет ограничений на количество передвижений мыши

                for (i, value) in data.keys_record.mouse.iter().enumerate() {
                    mouse_vector[i as usize] = *value;
                }

                mouse_vector
            })
            .map(|vector| TensorData::from(vector).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, &self.device))
            .map(|tensor| tensor.reshape([1, 2, MOUSE_VECTOR_LENGTH]))
            // // Простая нормализация
            // .map(|tensor| tensor.div_scalar(255))
            .collect();

        Tensor::cat(mouse, 0)
    }

    fn extract_const_images(&self, mydata: &Vec<MyConstData>) -> Tensor<B, 4> {
        let images = mydata
            .iter()
            .map(|data| TensorData::from(data.image.pixels).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 3>::from_data(data, &self.device))
            // 1 штука, 4 параметра цвета
            .map(|tensor| tensor.reshape([1, 4, HEIGHT, WIDTH]))
            // Простая нормализация цветов
            .map(|tensor| tensor / 255)
            .collect();

        Tensor::cat(images, 0)
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
}

#[derive(Clone, Debug)]
pub struct FrameBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub keys: Tensor<B, 2>,
    pub mouse: Tensor<B, 3>,
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> Batcher<MyConstData, FrameBatch<B>> for FrameBatcher<B> {
    fn batch(&self, mydata: Vec<MyConstData>) -> FrameBatch<B> {
        let images = self.extract_const_images(&mydata);
        let keys = self.extract_const_keys(&mydata);
        let mouse = self.extract_const_mouse(&mydata);

        // Сдвинутые изображения на 1
        // TODO: Изменить?
        let targets = self.extract_targets(&images);

        FrameBatch {
            images,
            keys,
            mouse,
            targets,
        }
    }
}
