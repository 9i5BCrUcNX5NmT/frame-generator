mod csv_processing;
mod data;
mod images;
mod inference;
mod model;
mod training;
mod types;

use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};
use model::ModelConfig;
use serde::Deserialize;
use training::TrainingConfig;

#[derive(Debug, Deserialize, Clone)]
pub struct Record {
    pub key: String,
    pub time: String,
    // Добавьте другие поля в зависимости от вашего CSV
}

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "tmp/test";
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
        device.clone(),
    );
}

#[cfg(test)]
mod tests {
    use images::{convert_images_to_image_pixel_data, load_images_from_directory, process_images};

    use super::*;

    #[ignore = "for developing"]
    #[test]
    fn generate() {
        type MyBackend = Wgpu<f32, i32>;

        let device = burn::backend::wgpu::WgpuDevice::default();
        let artifact_dir = "tmp/test";

        let input_dir = "../data/images/test";

        let images = load_images_from_directory(input_dir).unwrap();

        let images_data = convert_images_to_image_pixel_data(images);

        // let image_dataset: InMemDataset<images::ImagePixelData> = InMemDataset::new(images);

        // crate::inference::infer::<MyBackend>(artifact_dir, device.clone(), images_data[0].clone());
        // crate::inference::infer::<MyBackend>(
        //     artifact_dir,
        //     device.clone(),
        //     images_data[images_data.len() - 1].clone(),
        // );
    }

    #[ignore = "for developing"]
    #[test]
    fn resize_images() {
        let input_dir = "../data/images/raw"; // Путь к входной папке с изображениями
        let output_dir = "../data/images/resized_images"; // Путь к выходной папке для сохранения измененных изображений
        let width = 200;
        let height = 200;

        assert!(process_images(input_dir, output_dir, width, height).is_ok());

        // let images = load_images_from_directory(input_dir).unwrap();

        // let images_data = convert_images_to_image_pixel_data(images);

        // let image_dataset: InMemDataset<images::ImagePixelData> = InMemDataset::new(images);

        // crate::inference::infer::<MyBackend>(artifact_dir, device.clone(), images_data[0].clone());
        // crate::inference::infer::<MyBackend>(artifact_dir, device.clone(), images_data[1].clone());
    }

    #[ignore = "for developing"]
    #[test]
    fn work_with_csv() {
        let _ = csv_processing::load_keys_from_directory("../data/keys");
    }
}
