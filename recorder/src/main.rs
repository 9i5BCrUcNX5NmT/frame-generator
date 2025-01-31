use csv::Writer;
use mp4::{Mp4Config, Mp4Sample, Mp4Writer, TrackConfig};
use rdev::{listen, EventType};
use scap::capturer::{Area, Capturer, Options, Point, Size};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader};
use std::sync::{Arc, Mutex};
use std::thread::{self, sleep};
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

        // // Get recording targets
        // let targets = scap::get_all_targets();
        // println!("Targets: {:?}", targets);

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

        sleep(Duration::from_secs(1));

        // Открываем файл с помощью OpenOptions
        let file_path = "../data/videos/test_2.mp4";
        let mut file = OpenOptions::new()
            .write(true) // Указываем, что мы хотим записывать в файл
            .create(true) // Создаем файл, если он не существует
            .append(true) // Добавляем данные в конец файла
            .open(file_path)
            .unwrap(); // Открываем файл

        let writer = io::BufWriter::new(file);

        let mp4_config = Mp4Config {
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

        let mut mp4_writer = Mp4Writer::write_start(writer, &mp4_config).unwrap();

        let sample = Mp4Sample {
            start_time: 0,
            duration: 1000 / 60,
            rendering_offset: 0, // TODO: хз для чего
            is_sync: false,      // TODO: хз для чего
            bytes: match capturer.get_next_frame().unwrap() {
                scap::frame::Frame::YUVFrame(yuvframe) => todo!(),
                scap::frame::Frame::RGB(rgbframe) => todo!(),
                scap::frame::Frame::RGBx(rgbx_frame) => todo!(),
                scap::frame::Frame::XBGR(xbgrframe) => todo!(),
                scap::frame::Frame::BGRx(bgrx_frame) => todo!(),
                scap::frame::Frame::BGR0(bgrframe) => todo!(),
                scap::frame::Frame::BGRA(bgraframe) => bgraframe.data.into(),
            },
        }; // Длительность 1000 (1 секунда)

        let track_config = TrackConfig {
            track_type: mp4::TrackType::Video,
            timescale: 1000,
            language: "ru".to_string(),
            media_conf: mp4::MediaConfig::HevcConfig(mp4::HevcConfig {
                width: 2000,
                height: 1000,
            }),
        };
        mp4_writer.add_track(&track_config).unwrap();
        mp4_writer.write_sample(1, &sample).unwrap();

        mp4_writer.write_end().unwrap();

        // Stop Capture
        capturer.stop_capture();
    }
}

fn main() {
    // let f = File::open("../data/videos/test.mp4").unwrap();
    // let size = f.metadata().unwrap().len();
    // let reader = BufReader::new(f);

    // let mp4 = mp4::Mp4Reader::read_header(reader, size).unwrap();

    // // Track info.
    // for track in mp4.tracks().values() {
    //     println!(
    //         "track: #{}({}) {} : {}",
    //         track.track_id(),
    //         track.language(),
    //         track.track_type().unwrap(),
    //         track.box_type().unwrap(),
    //     );
    // }
    // KeysRecorder::record().unwrap();
    VideoRecorder::record();
}
