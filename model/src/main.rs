mod data;
mod images;
mod model;
mod training;

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
    use burn::data::dataset::InMemDataset;
    use images::{convert_images_to_image_pixel_data, load_images_from_directory};

    use super::*;

    #[ignore = "for developing"]
    #[test]
    fn dev_test() {
        // let input_dir = "../data/images/raw"; // Путь к входной папке с изображениями
        let output_dir = "../data/resized_images"; // Путь к выходной папке для сохранения измененных изображений
                                                   // let width = 200;
                                                   // let height = 200;

        // process_images(input_dir, output_dir, width, height).unwrap();

        let images = load_images_from_directory(output_dir).unwrap();

        let images = convert_images_to_image_pixel_data(images);

        let image_dataset: InMemDataset<images::ImagePixelData> = InMemDataset::new(images);
    }
}
