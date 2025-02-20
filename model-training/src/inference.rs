use std::{fs, path::Path};

use burn::{
    backend::Wgpu,
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    prelude::Backend,
    record::{CompactRecorder, Recorder},
};

use crate::{
    csv_processing::KeysRecord,
    data::FrameBatcher,
    images::{
        self, convert_image_pixel_data_to_images, convert_images_to_image_pixel_data,
        load_images_from_directory, ImagePixelData,
    },
    training::TrainingConfig,
    types::MyData,
};

fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MyData) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    let batcher = FrameBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images, batch.keys, batch.mouse);

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

    // Создаем выходную директорию, если она не существует
    let output_str = format!("{artifact_dir}/output");
    let output_dir = Path::new(&output_str);
    fs::create_dir_all(output_dir).unwrap();

    images::save_image(&images[0], &output_dir.join(format!("image.png")));

    // println!("Predicted {} ", output);
}

pub fn run(artifact_dir: &str, image_path: &str) {
    type MyBackend = Wgpu<f32, i32>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    // TODO: хз
    // let image_path = Path::new(image_path).to_path_buf();
    let image_data = load_images_from_directory(image_path).unwrap();
    let image_pixel_data = convert_images_to_image_pixel_data(image_data);

    let item = MyData {
        image: image_pixel_data[0],
        keys: KeysRecord {
            keys: vec![],
            mouse: vec![[100, 100]],
        },
    };

    crate::inference::infer::<MyBackend>(artifact_dir, device.clone(), item);
}
