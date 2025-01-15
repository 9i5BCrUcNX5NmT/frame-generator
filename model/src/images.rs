use std::{
    fs, io,
    path::{Path, PathBuf},
};

use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};

pub struct ImageData {
    image_path: PathBuf,
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

#[derive(Debug, Clone)]
pub struct ImagePixelData {
    pub pixels: [[[u8; 4]; 200]; 200],
}

impl ImagePixelData {
    pub fn from_image(image: &DynamicImage) -> Self {
        let mut pixels: [[[u8; 4]; 200]; 200] = [[[0; 4]; 200]; 200];

        for pixel in image.pixels() {
            pixels[pixel.0 as usize][pixel.1 as usize] = pixel.2 .0;
        }

        ImagePixelData { pixels }
    }

    pub fn to_image(&self) -> DynamicImage {
        let mut img = RgbaImage::new(200, 200);

        for (height, i) in self.pixels.iter().enumerate() {
            for (width, j) in i.iter().enumerate() {
                let pixel = Rgba(*j);

                img.put_pixel(width as u32, height as u32, pixel);
            }
        }

        image::DynamicImage::ImageRgba8(img)
    }
}

pub fn convert_images_to_image_pixel_data(images: Vec<ImageData>) -> Vec<ImagePixelData> {
    images
        .iter()
        .map(|image_data| load_image(image_data))
        .map(|image| ImagePixelData::from_image(&image))
        .collect()
}

pub fn convert_image_pixel_data_to_images(images_data: Vec<ImagePixelData>) -> Vec<DynamicImage> {
    images_data
        .iter()
        .map(|image_data| image_data.to_image())
        .collect()
}
