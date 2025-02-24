use csv::Writer;
use rdev::{listen, Event, EventType};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde::Serialize;

use crate::DATA_DIR;

#[derive(Debug, Serialize)]
struct CsvRecord {
    keys: String,
    mouse: String,
}

pub struct KeysRecorder {
    key_states: HashSet<String>,
    mouse_states: HashSet<String>,
}

impl KeysRecorder {
    // pub fn get_key_states(&self) -> Arc<Mutex<HashSet<String>>> {
    //     Arc::clone(&self.key_states)
    // }

    // pub fn get_mouse_states(&self) -> Arc<Mutex<HashSet<String>>> {
    //     Arc::clone(&self.mouse_states)
    // }

    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            key_states: HashSet::new(),
            mouse_states: HashSet::new(),
        }))
    }

    pub fn insert_event(&mut self, event: Event) {
        match event.event_type {
            EventType::KeyPress(key) => {
                self.key_states.insert(format!("{:?}", key)); // Сохраняем нажатие клавиши
            }
            EventType::KeyRelease(key) => {
                self.key_states.remove(&format!("{:?}", key)); // Удаляем отпускание клавиши
            }
            EventType::MouseMove { x, y } => {
                self.mouse_states.insert(format!("{},{}", x, y)); // Сохраняем координаты мыши
            }
            EventType::ButtonPress(button) => {
                self.key_states.insert(format!("{:?}", button)); // Сохраняем нажатие клавиши
            }
            EventType::ButtonRelease(button) => {
                self.key_states.remove(&format!("{:?}", button)); // Удаляем отпускание клавиши
            }
            EventType::Wheel { delta_x, delta_y } => {
                self.mouse_states.insert(format!("{},{}", delta_x, delta_y)); // Удаляем отпускание клавиши
            }
        }
    }

    pub fn write_keys(&mut self, data_dir: &str) {
        let file_path = data_dir.to_owned() + "keys/key_events.csv";

        let mut writer = Writer::from_writer(
            OpenOptions::new()
                .append(true)
                .create(true)
                .open(file_path)
                .unwrap(),
        );

        if !self.key_states.is_empty() || !self.mouse_states.is_empty() {
            let mut record = CsvRecord {
                keys: "".to_string(),
                mouse: "".to_string(),
            };

            if !self.key_states.is_empty() {
                // Объединяем все нажатия клавиш в одну строку
                let keys_pressed = self
                    .key_states
                    .iter()
                    .cloned()
                    .collect::<Vec<String>>()
                    .join(", "); // Объединяем нажатия в одну строку
                record.keys = keys_pressed;
            }

            if !self.mouse_states.is_empty() {
                // Объединяем все нажатия клавиш в одну строку
                let mouse_action = self
                    .mouse_states
                    .iter()
                    .cloned()
                    .collect::<Vec<String>>()
                    .join(", "); // Объединяем нажатия в одну строку
                record.mouse = mouse_action;

                self.mouse_states.clear();
            }

            writer.serialize(record).unwrap();
            writer.flush().unwrap();
        }
    }
}

pub fn record() -> io::Result<()> {
    let recorder = KeysRecorder::new();

    let recorder_clone = Arc::clone(&recorder);

    // Запуск потока для прослушивания нажатий клавиш
    thread::spawn(move || {
        listen(move |event| {
            recorder_clone.lock().unwrap().insert_event(event);
        })
        .unwrap();
    });

    loop {
        thread::sleep(Duration::from_millis(50));

        recorder.lock().unwrap().write_keys(DATA_DIR);
    }
}
