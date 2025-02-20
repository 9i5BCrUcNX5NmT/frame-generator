use std::{
    fs, io,
    path::{Path, PathBuf},
};

use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};

pub struct ImageData {
    pub image_path: PathBuf,
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

pub fn save_image(image: &DynamicImage, output_path: &Path) {
    image.save(output_path).expect("Failed to save image");
}

#[derive(Debug, Clone, Copy)]
pub struct ImagePixelData {
    pub pixels: [[[u8; 200]; 200]; 4],
}

impl ImagePixelData {
    pub fn from_image(image: &DynamicImage) -> Self {
        let mut pixels: [[[u8; 200]; 200]; 4] = [[[0; 200]; 200]; 4];

        for (height, width, colors) in image.pixels() {
            for (i, color) in colors.0.iter().enumerate() {
                pixels[i][height as usize][width as usize] = *color;
            }
        }

        ImagePixelData { pixels }
    }

    pub fn to_image(&self) -> DynamicImage {
        let mut img = RgbaImage::new(200, 200);

        for i in 0..200 {
            for j in 0..200 {
                let pixel = Rgba([
                    self.pixels[0][i][j],
                    self.pixels[1][i][j],
                    self.pixels[2][i][j],
                    self.pixels[3][i][j],
                ]);

                img.put_pixel(i as u32, j as u32, pixel);
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
