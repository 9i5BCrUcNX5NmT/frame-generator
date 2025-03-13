use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use fs_extra::dir;
use rdev::{Event, EventType};
use xcap::Monitor;

#[derive(Clone)]
pub struct VideoRecorder {
    should_stop: Arc<Mutex<bool>>,
    path_to_images: PathBuf,
}

impl VideoRecorder {
    pub fn new(path_to_images: PathBuf) -> Self {
        Self {
            should_stop: Arc::new(Mutex::new(false)),
            path_to_images,
        }
    }

    pub fn start(&self) -> JoinHandle<()> {
        let path_to_images = self.path_to_images.clone();
        let should_stop = self.should_stop.clone();

        dir::create_all(path_to_images.clone(), true).unwrap();

        // Запускаем поток для захвата изображений
        thread::spawn(move || {
            let monitors = Monitor::all().unwrap();
            let monitor = &monitors[0];

            let mut count = 0;

            while !*should_stop.lock().unwrap() {
                let image = monitor.capture_image().unwrap();
                let file_path = path_to_images.join(format!("image-{}.png", count));
                image.save(file_path).unwrap();
                count += 1;

                // Ждем 1/20 секунды
                thread::sleep(Duration::from_millis(50));
            }
        })
    }

    pub fn check_keys(&mut self, event: &Event) {
        match event.event_type {
            EventType::KeyPress(key) => match key {
                rdev::Key::KeyQ => {
                    println!("Video recorded");

                    *self.should_stop.lock().unwrap() = true;
                }
                _ => {}
            },
            _ => {}
        }
    }

    pub fn is_finished(&self) -> bool {
        *self.should_stop.lock().unwrap()
    }
}
