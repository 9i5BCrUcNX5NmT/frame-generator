use burn::{
    backend,
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    prelude::Backend,
    record::{CompactRecorder, Recorder},
    tensor::Tensor,
};
use common::*;
use image::{DynamicImage, Rgba32FImage};
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

    let batcher = FrameBatcher::new(device.clone());
    let batch = batcher.batch(vec![item], &device);

    let noise = batch
        .images
        .random_like(burn::tensor::Distribution::Normal(0.0, 1.0));

    let output = model.forward(
        batch.images.clone(),
        batch.keys.clone(),
        batch.mouse.clone(),
        noise,
    );
    let output = batch.images * 0.9 + output * 0.1;

    let images = output
        .iter_dim(0)
        // Возвращение из нормализации
        // .map(|tensor| tensor * 255)
        .map(&mut |tensor: Tensor<B, 4>| tensor.to_data())
        .map(&mut |data: burn::prelude::TensorData| data.to_vec().unwrap())
        // .map(|vector| vector.iter().map(|v| *v as u8).collect())
        .map(&mut |vector| {
            let image = Rgba32FImage::from_vec(WIDTH as u32, HEIGHT as u32, vector).unwrap();

            DynamicImage::from(image)
            // RgbaImage::from_vec(WIDTH as u32, HEIGHT as u32, vector).unwrap()
        })
        .collect();

    // let dynamic_images = images.iter().map(|image| image.to_image()).collect();
    // dynamic_images

    images
}

pub fn generate(
    current_image: &DynamicImage,
    keys: Vec<String>,
    mouse: Vec<[i32; 2]>,
) -> DynamicImage {
    let artifact_dir = "tmp/test";

    #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
    type MyBackend = backend::NdArray<f32>;
    #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
    let device = backend::ndarray::NdArrayDevice::default();

    #[cfg(feature = "wgpu")]
    type MyBackend = backend::Wgpu<f32, i32>;
    #[cfg(feature = "wgpu")]
    let device = backend::wgpu::WgpuDevice::default();

    #[cfg(feature = "cuda")]
    type MyBackend = backend::Cuda<f32, i32>;
    #[cfg(feature = "cuda")]
    let device = backend::cuda::CudaDevice::default();

    let my_image: MyImage<HEIGHT, WIDTH, CHANNELS> = MyImage::from_image(current_image);

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
