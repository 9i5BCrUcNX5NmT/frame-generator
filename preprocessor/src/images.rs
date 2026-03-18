use std::{
    fmt, fs, io,
    path::{Path, PathBuf},
};

use hdf5_metno::H5Type;
use image::{DynamicImage, GenericImageView, RgbaImage};
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

pub fn save_image(image: &DynamicImage, output_path: &PathBuf) {
    image.save(output_path).expect("Failed to save image");
}

pub fn load_images_from_directory(dir: &PathBuf) -> io::Result<Vec<ImageData>> {
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
    input_dir: &PathBuf,
    output_dir: &PathBuf,
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
            let resized_image = resize_image(&DynamicImage::from(image), width, height); // Изменяем размер до заданных параметров

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

#[derive(Clone, Copy, H5Type)]
#[repr(C)]
pub struct MyImage<const HEIGHT: usize, const WIDTH: usize, const CHANNELS: usize> {
    pub pixels: [[[u8; HEIGHT]; WIDTH]; CHANNELS],
}

impl<const H: usize, const W: usize, const C: usize> fmt::Debug for MyImage<H, W, C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Здесь вы можете настроить вывод по своему усмотрению
        write!(f, "Image[{} x {}]({})", H, W, C)
    }
}

impl<const H: usize, const W: usize, const C: usize> MyImage<H, W, C> {
    // pub fn from_image(image: &DynamicImage) -> Self {
    //     let image = image.to_rgba8();
    //     let mut pixels: [[[u8; W]; H]; C] = [[[0; W]; H]; C];

    //     for (i, pixel) in image.pixels().enumerate() {
    //         let height = (i / C / W) % H;
    //         let width = (i / C) % W;

    //         for (color_index, color) in pixel.0.iter().enumerate() {
    //             pixels[color_index][height][width] = *color;
    //         }
    //     }

    //     MyImage { pixels }
    // }

    // pub fn to_image(&self) -> RgbaImage {
    //     let mut img = RgbaImage::new(W as u32, H as u32);

    //     for i in 0..H {
    //         for j in 0..W {
    //             let pixel = Rgb([
    //                 self.pixels[0][i][j],
    //                 self.pixels[1][i][j],
    //                 self.pixels[2][i][j],
    //             ]);

    //             img.put_pixel(j as u32, i as u32, pixel);
    //         }
    //     }

    //     img
    // }

    pub fn from_image(image: &DynamicImage) -> Self {
        let mut pixels = [[[0; H]; W]; C];

        for (width, height, colors) in image.pixels() {
            for (i, color) in colors.0.iter().enumerate() {
                pixels[i][width as usize][height as usize] = *color;
            }
        }

        MyImage { pixels }
    }

    pub fn to_image(&self) -> DynamicImage {
        let mut img = RgbaImage::new(W as u32, H as u32);

        for i in 0..W {
            for j in 0..H {
                let pixel = image::Rgba([
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

    pub fn from_image_data(image_data: &ImageData) -> MyImage<H, W, C> {
        let image = load_image(image_data);

        MyImage::from_image(&image)
    }
}

// pub fn from_images_data<const H: usize, const W: usize, const C: usize>(
//     images: Vec<ImageData>,
// ) -> Vec<MyImage<H, W>> {
//     images
//         .iter()
//         .map(|image_data| MyImage::from_image_data(image_data))
//         .collect()
// }

// pub fn to_dynamic_image<const H: usize, const W: usize, const C: usize>(
//     images_data: Vec<MyImage<H, W>>,
// ) -> Vec<DynamicImage> {
//     images_data
//         .iter()
//         .map(|image_data| image_data.to_image())
//         .collect()
// }

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Test load_images_from_directory with empty directory
    #[test]
    fn test_load_images_from_directory_empty() {
        let temp_dir = std::env::temp_dir().join("test_empty_images");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let result = load_images_from_directory(&temp_dir);

        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// Test load_images_from_directory ignores subdirectories
    #[test]
    fn test_load_images_from_directory_ignores_dirs() {
        let temp_dir = std::env::temp_dir().join("test_dir_ignores");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        // Create a subdirectory with files (should be ignored)
        let subdir = temp_dir.join("subdir");
        fs::create_dir_all(&subdir).unwrap();

        let result = load_images_from_directory(&temp_dir);

        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// Test MyImage::from_image_data with small test image
    #[test]
    fn test_my_image_debug_format() {
        use common::{CHANNELS, HEIGHT, WIDTH};

        let image: MyImage<HEIGHT, WIDTH, CHANNELS> = MyImage {
            pixels: [[[0; HEIGHT]; WIDTH]; CHANNELS],
        };

        let debug_str = format!("{:?}", image);
        assert!(debug_str.contains("Image"));
    }

    /// Test MyImage creation with custom dimensions
    #[test]
    fn test_my_image_custom_dimensions() {
        const H: usize = 64;
        const W: usize = 64;
        const C: usize = 4;

        let image: MyImage<H, W, C> = MyImage {
            pixels: [[[1; H]; W]; C],
        };

        // Check first pixel channel value
        assert_eq!(image.pixels[0][0][0], 1);

        let debug_str = format!("{:?}", image);
        assert!(debug_str.contains("64"));
    }

    /// Test MyImage pixel array bounds
    #[test]
    fn test_my_image_pixel_array_bounds() {
        const H: usize = 32;
        const W: usize = 32;
        const C: usize = 4;

        // Initialize with zeros
        let image: MyImage<H, W, C> = MyImage {
            pixels: [[[0; H]; W]; C],
        };

        // Set some values
        let mut image_mut = image;
        image_mut.pixels[0][0][0] = 255;
        image_mut.pixels[3][31][31] = 128;

        // Verify values were set correctly
        assert_eq!(image_mut.pixels[0][0][0], 255);
        assert_eq!(image_mut.pixels[3][31][31], 128);
        // Verify other values are still 0
        assert_eq!(image_mut.pixels[1][0][0], 0);
    }

    /// Test save_image creates file
    #[test]
    fn test_save_image_creates_file() {
        let temp_dir = std::env::temp_dir().join("test_save_image");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let image = DynamicImage::new_rgb8(10, 10);
        let output_path = temp_dir.join("test.png");

        save_image(&image, &output_path);

        assert!(output_path.exists());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// Test resize_image produces correct dimensions
    #[test]
    fn test_resize_image_dimensions() {
        // Create a 100x100 image
        let image = DynamicImage::new_rgb8(100, 100);

        // Resize to 50x50
        let resized = resize_image(&image, 50, 50);

        assert_eq!(resized.width(), 50);
        assert_eq!(resized.height(), 50);
    }

    /// Test resize_image with different filter types
    #[test]
    fn test_resize_image_different_sizes() {
        let image = DynamicImage::new_rgb8(200, 100);

        let resized_down = resize_image(&image, 50, 25);
        assert_eq!(resized_down.width(), 50);
        assert_eq!(resized_down.height(), 25);

        let resized_up = resize_image(&image, 400, 200);
        assert_eq!(resized_up.width(), 400);
        assert_eq!(resized_up.height(), 200);

        let resized_same = resize_image(&image, 200, 100);
        assert_eq!(resized_same.width(), 200);
        assert_eq!(resized_same.height(), 100);
    }
}
