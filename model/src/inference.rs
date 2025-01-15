use std::path::Path;

use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    prelude::Backend,
    record::{CompactRecorder, Recorder},
};

use crate::{
    data::FrameBatcher,
    images::{self, convert_image_pixel_data_to_images, ImagePixelData},
    training::TrainingConfig,
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: ImagePixelData) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    // let label = item.label;
    let batcher = FrameBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    // let predicted = output.flatten::<1>(0, 3);

    // let images = images
    //         .iter()
    //         .map(|image| TensorData::from(image.pixels).convert::<B::IntElem>())
    //         .map(|data| Tensor::<B, 3>::from_data(data, &self.device))
    //         // 1 штука, 4 параметра цвета, 200 на 200 размер
    //         .map(|tensor| tensor.reshape([1, 4, 200, 200]))
    //         // Простая нормализация
    //         .map(|tensor| tensor / 255)
    //         .collect();

    let images_data: Vec<ImagePixelData> = output
        .iter_dim(0)
        // Возвращение из нормализации
        .map(|tensor| tensor * 255)
        // Убираем лишнюю размерность
        .map(|tensor| tensor.reshape([4, 200, 200]))
        .map(|tensor| tensor.to_data())
        .map(|data| data.to_vec::<f32>().unwrap())
        // vector = 4 * 200 * 200
        .map(|vector| {
            let mut pixels: [[[u8; 200]; 200]; 4] = [[[0; 200]; 200]; 4];

            for (color_index, i) in vector.chunks(200 * 200).enumerate() {
                for (height, j) in i.chunks(200).enumerate() {
                    for (width, k) in j.iter().enumerate() {
                        pixels[color_index][height][width] = *k as u8;
                    }
                }
            }

            ImagePixelData { pixels }
        })
        .collect();

    let images = convert_image_pixel_data_to_images(images_data);

    let mut i = 0;

    for image in images {
        i += 1;

        images::save_image(
            &image,
            Path::new(&format!("{artifact_dir}/output/image_{i}.png")),
        );
    }

    // println!("Predicted {} ", output);
}
