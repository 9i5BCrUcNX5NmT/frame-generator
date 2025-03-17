use burn::{
    backend,
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    prelude::Backend,
    record::{CompactRecorder, Recorder},
};
use common::*;
use image::DynamicImage;
use preprocessor::{
    csv_processing::{KeysRecordConst, key_to_num},
    images::MyImage,
    types::MyConstData,
};

use crate::{data::FrameBatcher, training::TrainingConfig};

fn infer<B: Backend>(
    artifact_dir: &str,
    device: B::Device,
    item: MyConstData,
) -> Vec<DynamicImage> {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    let batcher = FrameBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images, batch.keys, batch.mouse);

    let images: Vec<MyImage<HEIGHT, WIDTH>> = output
        .iter_dim(0)
        // Возвращение из нормализации
        .map(|tensor| tensor * 255)
        // Убираем лишнюю размерность
        // .map(|tensor| tensor.reshape([4 * HEIGHT * WIDTH]))
        .map(|tensor| tensor.to_data())
        .map(|data| data.to_vec::<f32>().unwrap())
        .map(|vector| {
            let mut pixels: [[[u8; WIDTH]; HEIGHT]; 4] = [[[0; WIDTH]; HEIGHT]; 4];

            for (color_index, i) in vector.chunks(HEIGHT * WIDTH).enumerate() {
                for (height, j) in i.chunks(WIDTH).enumerate() {
                    for (width, k) in j.iter().enumerate() {
                        pixels[color_index][height][width] = *k as u8;
                    }
                }
            }

            MyImage { pixels }
        })
        .collect();

    let dynamic_images = images.iter().map(|image| image.to_image()).collect();

    dynamic_images
}

pub fn generate(
    current_image: &DynamicImage,
    keys: Vec<String>,
    mouse: Vec<[i32; 2]>,
) -> DynamicImage {
    let artifact_dir = "tmp/test";

    type MyBackend = backend::CudaJit<f32, i32>;
    let device = backend::cuda_jit::CudaDevice::default();
    // type MyBackend = backend::Wgpu<f32, i32>;
    // let device = backend::wgpu::WgpuDevice::default();
    // type MyBackend = backend::NdArray<f32>;
    // let device = backend::ndarray::NdArrayDevice::default();

    let my_image = MyImage::from_image(current_image);

    // TODO: мб пофиксить?
    // Да не, пока норм вроде

    let keys: Vec<u8> = keys
        .into_iter()
        .filter(|key| !key.is_empty())
        .map(|key| key.to_lowercase())
        .map(|key| key_to_num(&key))
        .collect();

    let mut const_keys: [u8; 200] = [0; 200];

    for (i, value) in keys.iter().enumerate() {
        const_keys[i] = *value;
    }

    let mut const_mouse: [[i32; 2]; 200] = [[0; 2]; 200];

    for (i, value) in mouse.iter().enumerate() {
        const_mouse[i] = *value;
    }

    let item = MyConstData {
        image: my_image,
        keys_record: KeysRecordConst {
            keys: const_keys,
            mouse: const_mouse,
        },
    };

    let next_image =
        crate::inference::infer::<MyBackend>(artifact_dir, device.clone(), item)[0].clone();

    next_image
}
