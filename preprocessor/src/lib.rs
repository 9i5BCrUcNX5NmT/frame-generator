use images::process_images;
use model_training::{HEIGHT, WIDTH};
use videos::process_videos;

mod images;
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
