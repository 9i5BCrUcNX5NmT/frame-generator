use std::{path::PathBuf, str::FromStr};

use converter::write_data;
use csv_processing::load_records_from_directory;
use image::DynamicImage;
use images::{process_images, MyImage};
use model_training::{HEIGHT, WIDTH};
use types::MyConstData;
use videos::process_videos;

mod converter;
mod csv_processing;
mod images;
mod types;
mod videos;

pub fn run() {
    let output_dir = "data/images/raw"; // Путь к выходной папке с изображениями
    std::fs::create_dir_all("data/images/raw").unwrap();

    process_videos("data/videos/video.mp4", "data/images/raw/");

    let input_dir = output_dir; // Путь к входной папке с изображениями

    let output_dir = "data/images/resized_images"; // Путь к выходной папке для сохранения измененных изображений
    std::fs::create_dir_all("data/images/resized_images").unwrap();

    process_images(input_dir, output_dir, WIDTH as u32, HEIGHT as u32).unwrap();
}

pub fn write_my_data() {
    let data_path = &PathBuf::from_str("data").unwrap();
    let mydatas = load_records_from_directory(&data_path.join("keys")).unwrap();
    let my_data = MyConstData {
        image: MyImage::from_image(&DynamicImage::new(
            WIDTH as u32,
            HEIGHT as u32,
            image::ColorType::Rgba8,
        )),
        keys_record: mydatas[0].clone(),
    };
    write_data(&data_path.join("preprocessor"), my_data).unwrap();
}
