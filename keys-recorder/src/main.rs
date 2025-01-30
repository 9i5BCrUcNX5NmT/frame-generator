use csv::Writer;
use rdev::{listen, EventType};
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde::Serialize;

#[derive(Debug, Serialize)]
struct CsvRecord {
    keys: String,
    mouse: String,
    buttons: String,
}

fn main() -> io::Result<()> {
    let key_states = Arc::new(Mutex::new(HashSet::new()));
    let key_states_clone = Arc::clone(&key_states);

    let mouse_states = Arc::new(Mutex::new(HashSet::new()));
    let mouse_states_clone = Arc::clone(&mouse_states);

    let button_states = Arc::new(Mutex::new(HashSet::new()));
    let button_states_clone = Arc::clone(&button_states);

    // Запуск потока для прослушивания нажатий клавиш
    thread::spawn(move || {
        listen(move |event| {
            let mut k_states = key_states_clone.lock().unwrap();
            let mut m_states = mouse_states_clone.lock().unwrap();
            let mut b_states = button_states_clone.lock().unwrap();

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
                    b_states.insert(format!("{:?}", button)); // Сохраняем нажатие клавиши
                }
                EventType::ButtonRelease(button) => {
                    b_states.remove(&format!("{:?}", button)); // Удаляем отпускание клавиши
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

        let b_states = button_states.lock().unwrap();

        if !k_states.is_empty() || !m_states.is_empty() || !b_states.is_empty() {
            let mut record = CsvRecord {
                keys: "".to_string(),
                mouse: "".to_string(),
                buttons: "".to_string(),
            };

            if !k_states.is_empty() {
                // Объединяем все нажатия клавиш в одну строку
                let keys_pressed = k_states.iter().cloned().collect::<Vec<String>>().join(", "); // Объединяем нажатия в одну строку
                record.keys = keys_pressed;
            }

            if !b_states.is_empty() {
                // Объединяем все нажатия клавиш в одну строку
                let buttons_pressed = b_states.iter().cloned().collect::<Vec<String>>().join(", "); // Объединяем нажатия в одну строку
                record.buttons = buttons_pressed;
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
