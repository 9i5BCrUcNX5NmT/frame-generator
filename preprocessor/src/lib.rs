use std::{path::PathBuf, str::FromStr};

use csv_processing::load_records_from_directory;
use hdf5_processing::{read_data, write_data};
use image::DynamicImage;
use images::{process_images, MyImage};
use model_training::{HEIGHT, WIDTH};
use types::MyConstData;
use videos::process_videos;

mod csv_processing;
mod hdf5_processing;
mod images;
mod types;
mod videos;

pub fn process_my_videos() {
    std::fs::create_dir_all("data/images/raw").unwrap();

    process_videos("data/videos/video.mp4", "data/images/raw/");
}

pub fn process_my_images() {
    let input_dir = "data/images/raw"; // Путь к входной папке с изображениями

    let output_dir = "data/images/resized_images"; // Путь к выходной папке для сохранения измененных изображений
    std::fs::create_dir_all("data/images/resized_images").unwrap();

    process_images(input_dir, output_dir, WIDTH as u32, HEIGHT as u32).unwrap();
}

pub fn write_my_data() {
    let data_path = PathBuf::from_str("data").unwrap();

    let mydatas = load_records_from_directory(&data_path.join("keys")).unwrap();
    let my_data = MyConstData {
        image: MyImage::from_image(&DynamicImage::new(
            WIDTH as u32,
            HEIGHT as u32,
            image::ColorType::Rgba8,
        )),
        keys_record: mydatas[0].clone(),
    };

    let data_path = &data_path.join("preprocessor");
    std::fs::create_dir_all(data_path).unwrap();

    write_data(&data_path, my_data).unwrap();
}

pub fn read_my_data() {
    let data_path = PathBuf::from_str("data").unwrap();
    let data_path = &data_path.join("preprocessor");

    let a = read_data(data_path).unwrap();
    println!("{:?}\n\n", a);
}
