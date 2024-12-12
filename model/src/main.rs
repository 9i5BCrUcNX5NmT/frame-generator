use std::{
    fs, io,
    path::{Path, PathBuf},
};

use burn::{
    backend::Wgpu,
    data::dataset::{self, vision::MnistDataset, Dataset},
};
use images::{create_dataset, load_images_from_directory, process_images};
use model::ModelConfig;

mod data;
mod images;
mod model;

use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Record {
    pub key: String,
    pub time: String,
    // Добавьте другие поля в зависимости от вашего CSV
}

fn main() {
    // type MyBackend = Wgpu<f32, i32>;

    // let device = Default::default();
    // let model = ModelConfig::new().init::<MyBackend>(&device);

    // println!("{}", model);

    // let file_path = "../keyboard parser/key_log.csv";
    // // let records = read_csv(file_path).unwrap();

    // let rdr = csv::ReaderBuilder::new();

    // Здесь вы можете использовать records для настройки вашего dataset в burn
    // for record in records {
    //     println!("{:?}", record);
    // }

    // let dataset = dataset::InMemDataset::<Record>::from_csv(file_path, &rdr);

    // for i in dataset.unwrap().iter() {
    //     println!("{:?}", i);
    // }

    // let input_dir = "../data/images"; // Путь к входной папке с изображениями
    // let output_dir = "../data/resized_images"; // Путь к выходной папке для сохранения измененных изображений
    // let width = 200;
    // let height = 200;

    // process_images(input_dir, output_dir, width, height).unwrap();

    let image_dataset = load_images_from_directory(output_dir).unwrap();

    create_dataset(image_dataset);
}
