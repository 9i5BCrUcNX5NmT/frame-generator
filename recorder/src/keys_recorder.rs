use csv::Writer;
use rdev::{Event, EventType};
use std::collections::HashSet;
use std::fs::File;

use serde::Serialize;

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
    pub fn new() -> Self {
        Self {
            key_states: HashSet::new(),
            mouse_states: HashSet::new(),
        }
    }

    pub fn insert_key(&mut self, event: &Event) {
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

    pub fn write_keys(&mut self, writer: &mut Writer<File>) {
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
