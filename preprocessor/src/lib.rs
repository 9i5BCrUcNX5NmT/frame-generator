use std::{path::PathBuf, str::FromStr};

use csv_processing::load_records_from_directory;
use hdf5_processing::{read_data, write_data};
use image::DynamicImage;
use images::{load_images_from_directory, process_images, MyImage};
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

    let data_path = &data_path.join("preprocessor");
    std::fs::create_dir_all(data_path).unwrap();

    let my_data: Vec<MyConstData> = keys_records
        .iter()
        .zip(images.iter())
        .map(|(keys_record, image_data)| MyConstData {
            image: MyImage::from_image_data(image_data),
            keys_record: keys_record.clone(),
        })
        .collect();

    dbg!(my_data.len());

    let my_data = &[my_data[0].clone(), my_data[1].clone(), my_data[2].clone()];

    write_data(&data_path).unwrap();
}

pub fn read_my_data() {
    let data_path = PathBuf::from_str("data").unwrap();
    let data_path = &data_path.join("preprocessor");

    let a = read_data(data_path).unwrap();
    println!("{:?}\n\n", a);
}
