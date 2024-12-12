use std::{
    fs, io,
    path::{Path, PathBuf},
};

use image::DynamicImage;

pub struct ImageData {
    image_path: PathBuf,
    label: String,
}

pub fn load_images_from_directory(dir: &str) -> io::Result<Vec<ImageData>> {
    let mut dataset = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            // Здесь вы можете определить, как извлекать метки из имени файла или структуры директорий
            let label = path.file_stem().unwrap().to_string_lossy().to_string();
            dataset.push(ImageData {
                image_path: path,
                label,
            });
        }
    }

    Ok(dataset)
}

fn load_image(image_data: &ImageData) -> DynamicImage {
    image::open(&image_data.image_path).expect("Failed to open image")
}

fn resize_image(image: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    image.resize_exact(width, height, image::imageops::FilterType::Lanczos3)
}

fn save_image(image: &DynamicImage, output_path: &Path) {
    image.save(output_path).expect("Failed to save image");
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

    for data in dataset {
        let image = load_image(&data);
        let resized_image = resize_image(&image, width, height); // Изменяем размер до 200x200

        // Создаем путь для сохранения измененного изображения
        let output_path = Path::new(output_dir).join(data.image_path.file_name().unwrap());
        save_image(&resized_image, &output_path);
    }

    Ok(())
}

pub fn create_dataset(dataset: Vec<ImageData>) {
    for data in dataset {
        let image = load_image(&data);

        println!("{:?}, {:?}", image.height(), image.width());
    }
}
