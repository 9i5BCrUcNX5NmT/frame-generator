use std::{
    fmt, fs, io,
    path::{Path, PathBuf},
};

use hdf5_metno::H5Type;
use image::{DynamicImage, RgbaImage};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

pub struct ImageData {
    image_path: PathBuf,
}

fn load_image(image_data: &ImageData) -> RgbaImage {
    image::open(&image_data.image_path)
        .expect("Failed to open image")
        .to_rgba8()
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
    pub pixels: [[[u8; WIDTH]; HEIGHT]; CHANNELS],
}

impl<const H: usize, const W: usize, const C: usize> fmt::Debug for MyImage<H, W, C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Здесь вы можете настроить вывод по своему усмотрению
        write!(f, "Image[{} x {}]({})", H, W, C)
    }
}

impl<const H: usize, const W: usize, const C: usize> MyImage<H, W, C> {
    pub fn from_image(image: &RgbaImage) -> Self {
        let mut pixels: [[[u8; W]; H]; C] = [[[0; W]; H]; C];

        for (i, pixel) in image.pixels().enumerate() {
            let height = (i / C / W) % H;
            let width = (i / C) % W;

            for (color_index, color) in pixel.0.iter().enumerate() {
                pixels[color_index][height][width] = *color;
            }
        }

        MyImage { pixels }
    }

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

    // pub fn from_image(image: &DynamicImage) -> Self {
    //     let mut pixels: [[[f32; W]; H]; CHANNELS] = [[[0.0; W]; H]; CHANNELS];

    //     for (height, width, colors) in image.pixels() {
    //         for (i, color) in colors.0.iter().enumerate() {
    //             pixels[i][height as usize][width as usize] = *color;
    //         }
    //     }

    //     MyImage { pixels }
    // }

    // pub fn to_image(&self) -> DynamicImage {
    //     let mut img = RgbaImage::new(W as u32, H as u32);

    //     for i in 0..H {
    //         for j in 0..W {
    //             let pixel = Rgba([
    //                 self.pixels[0][i][j],
    //                 self.pixels[1][i][j],
    //                 self.pixels[2][i][j],
    // self.pixels[3][i][j],
    //             ]);

    //             img.put_pixel(j as u32, i as u32, pixel);
    //         }
    //     }

    //     image::DynamicImage::ImageRgba8(img)
    // }

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
