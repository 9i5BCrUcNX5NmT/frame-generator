use std::{
    fs, io,
    path::{Path, PathBuf},
};

use image::DynamicImage;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

pub struct ImageData {
    image_path: PathBuf,
}

fn load_image(image_data: &ImageData) -> DynamicImage {
    image::open(&image_data.image_path).expect("Failed to open image")
}

fn resize_image(image: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    image.resize_exact(width, height, image::imageops::FilterType::Lanczos3)
}

pub fn save_image(image: &DynamicImage, output_path: &Path) {
    image.save(output_path).expect("Failed to save image");
}

pub fn load_images_from_directory(dir: &str) -> io::Result<Vec<ImageData>> {
    let mut dataset = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            // Здесь вы можете определить, как извлекать метки из имени файла или структуры директорий
            dataset.push(ImageData { image_path: path });
        }
    }

    Ok(dataset)
}

pub fn process_images(
    input_dir: &str,
    output_dir: &str,
    width: u32,
    height: u32,
) -> io::Result<()> {
    let dataset = load_images_from_directory(input_dir)?;

    // Создаем выходную директорию, если она не существует
    fs::create_dir_all(output_dir)?;

    let finish = dataset.len();
    let counter = std::sync::Arc::new(std::sync::Mutex::new(0));

    dataset.par_iter().for_each(|data| {
        // Создаем путь для сохранения измененного изображения
        let output_path = Path::new(output_dir).join(data.image_path.file_name().unwrap());

        if output_path.metadata().is_err() {
            let image = load_image(&data);
            let resized_image = resize_image(&image, width, height); // Изменяем размер до заданных параметров

            save_image(&resized_image, &output_path);

            let mut count = counter.lock().unwrap();
            *count += 1;

            if *count % 100 == 0 {
                println!(
                    "Progress: {}%",
                    (*count as f64 / finish as f64 * 100.0).round(),
                );
            }
        }
    });

    let total_processed = *counter.lock().unwrap();
    println!("Было преобразвано {} изображений", total_processed);

    Ok(())
}
