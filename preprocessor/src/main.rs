use images::process_images;

mod images;
mod videos;

fn main() {
    let output_dir = "data/images/raw"; // Путь к выходной папке с изображениями
    std::fs::create_dir_all("data/images/raw").unwrap();

    // process_videos("data/videos/video.mp4", "data/images/raw");

    let input_dir = output_dir; // Путь к входной папке с изображениями

    let output_dir = "data/images/resized_images"; // Путь к выходной папке для сохранения измененных изображений
    std::fs::create_dir_all("data/images/resized_images").unwrap();

    let width = 200;
    let height = 200;

    process_images(input_dir, output_dir, width, height).unwrap();
}
