use std::{path::PathBuf, str::FromStr};

use csv_processing::load_records_from_directory;

use common::*;
use hdf5_processing::{read_all_hdf5_files, write_data_to_hdf5_files};
use images::{load_images_from_directory, process_images, MyImage};
use types::MyConstData;
use videos::process_videos;

pub mod csv_processing;
pub mod hdf5_processing;
pub mod images;
pub mod types;
mod videos;

pub fn process_my_videos() {
    std::fs::create_dir_all("data/images/raw").unwrap();

    process_videos("data/videos/video.mp4", "data/images/raw/");
}

pub fn process_my_images() {
    let input_dir = &PathBuf::from_str("data/images/raw").unwrap(); // Путь к входной папке с изображениями

    std::fs::create_dir_all("data/images/resized_images").unwrap();
    let output_dir = &PathBuf::from_str("data/images/resized_images").unwrap(); // Путь к выходной папке для сохранения измененных изображений

    process_images(input_dir, output_dir, WIDTH as u32, HEIGHT as u32).unwrap();
}

pub fn write_my_data() {
    let data_path = PathBuf::from_str("data").unwrap();

    let keys_records = load_records_from_directory(&data_path.join("keys")).unwrap();
    let images = load_images_from_directory(&data_path.join("images/resized_images")).unwrap();

    if images.len() == 0 {
        panic!("Отсутствуют изображения для обработки")
    }

    let my_data: Vec<MyConstData> = keys_records
        .iter()
        .zip(images.iter())
        .map(|(keys_record, image_data)| MyConstData {
            image: MyImage::from_image_data(image_data),
            keys_record: keys_record.clone(),
        })
        .collect();

    write_data_to_hdf5_files(&data_path.join("hdf5_files"), &my_data);
}

pub fn read_my_data() {
    let data_path = PathBuf::from_str("data").unwrap();
    let data_path = &data_path.join("hdf5_files");

    let a = read_all_hdf5_files(data_path).unwrap();
    println!("{:?}\n\n", a.len());
}
