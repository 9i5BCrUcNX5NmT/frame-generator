use csv::Writer;
use mp4::{Mp4Config, Mp4Sample, Mp4Writer};
use rdev::{listen, EventType};
use scap::capturer::{Area, Capturer, Options, Point, Size};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde::Serialize;

#[derive(Debug, Serialize)]
struct CsvRecord {
    keys: String,
    mouse: String,
}

// TODO: Сделать нормальное использование структуры
struct KeysRecorder {}

impl KeysRecorder {
    fn record() -> io::Result<()> {
        let key_states = Arc::new(Mutex::new(HashSet::new()));
        let key_states_clone = Arc::clone(&key_states);

        let mouse_states = Arc::new(Mutex::new(HashSet::new()));
        let mouse_states_clone = Arc::clone(&mouse_states);

        // Запуск потока для прослушивания нажатий клавиш
        thread::spawn(move || {
            listen(move |event| {
                let mut k_states = key_states_clone.lock().unwrap();
                let mut m_states = mouse_states_clone.lock().unwrap();

                match event.event_type {
                    EventType::KeyPress(key) => {
                        k_states.insert(format!("{:?}", key)); // Сохраняем нажатие клавиши
                    }
                    EventType::KeyRelease(key) => {
                        k_states.remove(&format!("{:?}", key)); // Удаляем отпускание клавиши
                    }
                    EventType::MouseMove { x, y } => {
                        m_states.insert(format!("{},{}", x, y)); // Сохраняем координаты мыши
                    }
                    EventType::ButtonPress(button) => {
                        k_states.insert(format!("{:?}", button)); // Сохраняем нажатие клавиши
                    }
                    EventType::ButtonRelease(button) => {
                        k_states.remove(&format!("{:?}", button)); // Удаляем отпускание клавиши
                    }
                    EventType::Wheel { delta_x, delta_y } => {
                        m_states.insert(format!("{},{}", delta_x, delta_y)); // Удаляем отпускание клавиши
                    }
                }
            })
            .unwrap();
        });

        // Запись в CSV файл каждые 50 мс (или по вашему усмотрению)
        let file_path = "../data/keys/key_events.csv";
        let mut writer = Writer::from_writer(
            OpenOptions::new()
                .append(true)
                .create(true)
                .open(file_path)
                .unwrap(),
        );

        loop {
            thread::sleep(Duration::from_millis(50));

            let k_states = key_states.lock().unwrap();
            let mut m_states = mouse_states.lock().unwrap();

            if !k_states.is_empty() || !m_states.is_empty() {
                let mut record = CsvRecord {
                    keys: "".to_string(),
                    mouse: "".to_string(),
                };

                if !k_states.is_empty() {
                    // Объединяем все нажатия клавиш в одну строку
                    let keys_pressed = k_states.iter().cloned().collect::<Vec<String>>().join(", "); // Объединяем нажатия в одну строку
                    record.keys = keys_pressed;
                }

                if !m_states.is_empty() {
                    // Объединяем все нажатия клавиш в одну строку
                    let mouse_action = m_states.iter().cloned().collect::<Vec<String>>().join(", "); // Объединяем нажатия в одну строку
                    record.mouse = mouse_action;

                    m_states.clear();
                }

                writer.serialize(record).unwrap();
                writer.flush().unwrap();
            }
        }
    }
}

struct VideoRecorder {}

impl VideoRecorder {
    fn record() {
        // Check if the platform is supported
        if !scap::is_supported() {
            println!("❌ Platform not supported");
            return;
        }

        // Check if we have permission to capture screen
        // If we don't, request it.
        if !scap::has_permission() {
            println!("❌ Permission not granted. Requesting permission...");
            if !scap::request_permission() {
                println!("❌ Permission denied");
                return;
            }
        }

        // Get recording targets
        let targets = scap::get_all_targets();
        println!("Targets: {:?}", targets);

        // All your displays and windows are targets
        // You can filter this and capture the one you need.

        // Create Options
        let options = Options {
            fps: 60,
            target: None, // None captures the primary display
            show_cursor: true,
            show_highlight: true,
            excluded_targets: None,
            output_type: scap::frame::FrameType::BGRAFrame,
            output_resolution: scap::capturer::Resolution::_720p,
            crop_area: Some(Area {
                origin: Point { x: 0.0, y: 0.0 },
                size: Size {
                    width: 2000.0,
                    height: 1000.0,
                },
            }),
            ..Default::default()
        };

        // Create Capturer
        let mut capturer = Capturer::build(options).unwrap();

        // Start Capture
        capturer.start_capture();

        // Stop Capture
        capturer.stop_capture();
    }
}

fn main() {
    // Открываем файл с помощью OpenOptions
    let file_path = "../data/videos/test.mp4";
    let mut file = OpenOptions::new()
        .write(true) // Указываем, что мы хотим записывать в файл
        .create(true) // Создаем файл, если он не существует
        .append(true) // Добавляем данные в конец файла
        .open(file_path)
        .unwrap(); // Открываем файл

    let writer = io::BufWriter::new(file);

    let config = Mp4Config {
        major_brand: str::parse("isom").unwrap(),
        minor_version: 512,
        compatible_brands: vec![
            str::parse("isom").unwrap(),
            str::parse("iso2").unwrap(),
            str::parse("avc1").unwrap(),
            str::parse("mp41").unwrap(),
        ],
        timescale: 1000,
    };

    let mut mp4_writer = Mp4Writer::write_start(writer, &config).unwrap();

    let sample = Mp4Sample {
        start_time: todo!(),
        duration: todo!(),
        rendering_offset: todo!(),
        is_sync: todo!(),
        bytes: todo!(),
    }; // Длительность 1000 (1 секунда)
    mp4_writer.write_sample(0, &sample).unwrap();

    mp4_writer.write_end().unwrap();

    // KeysRecorder::record().unwrap();
    // VideoRecorder::record();
}
