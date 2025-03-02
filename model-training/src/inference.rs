use burn::{
    backend::{cuda_jit::CudaDevice, wgpu::WgpuDevice, CudaJit, Wgpu},
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    prelude::Backend,
    record::{CompactRecorder, Recorder},
};
use common::*;
use image::DynamicImage;

use crate::{
    csv_processing::{key_to_num, KeysRecord},
    data::FrameBatcher,
    images::{convert_image_pixel_data_to_images, MyImage},
    training::TrainingConfig,
    types::MyData,
};

fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MyData) -> Vec<DynamicImage> {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    let batcher = FrameBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images, batch.keys, batch.mouse);

    let images_data: Vec<MyImage<HEIGHT, WIDTH>> = output
        .iter_dim(0)
        // Возвращение из нормализации
        .map(|tensor| tensor * 255)
        // Убираем лишнюю размерность
        .map(|tensor| tensor.reshape([4, HEIGHT, WIDTH]))
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
            // pixels
            // vector.iter().map(|v| *v as u8).collect::<Vec<u8>>()
        })
        .collect();

    let images = convert_image_pixel_data_to_images(images_data);

    // // Создаем выходную директорию, если она не существует
    // let output_str = format!("{artifact_dir}/output");
    // let output_dir = Path::new(&output_str);
    // fs::create_dir_all(output_dir).unwrap();

    // images::save_image(&images[0], &output_dir.join(format!("image.png")));

    // println!("Predicted {} ", output);

    images
}

pub fn generate(
    current_image: &DynamicImage,
    keys: Vec<String>,
    mouse: Vec<[i32; 2]>,
) -> DynamicImage {
    let artifact_dir = "tmp/test";
    // let image_path = "tmp/test/output";

    type MyBackend = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    // type MyBackend = CudaJit<f32, i32>;
    // let device = CudaDevice::default();

    // TODO: хз
    // let image_path = Path::new(image_path).to_path_buf();
    // let image_data = load_images_from_directory(image_path).unwrap();
    let image_pixel_data = MyImage::from_image(current_image);

    let keys = keys
        .iter()
        .filter(|key| !key.is_empty())
        .map(|key| key.to_lowercase())
        .map(|key| key_to_num(&key))
        .collect();

    let item = MyData {
        image: image_pixel_data,
        keys: KeysRecord { keys, mouse },
    };

    let next_image =
        crate::inference::infer::<MyBackend>(artifact_dir, device.clone(), item)[0].clone();

    next_image
}
