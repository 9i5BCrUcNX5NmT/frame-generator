use csv::Writer;
use rdev::{listen, EventType, Key};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

fn main() -> io::Result<()> {
    let key_states = Arc::new(Mutex::new(HashSet::new()));
    let key_states_clone = Arc::clone(&key_states);

    // Запуск потока для прослушивания нажатий клавиш
    thread::spawn(move || {
        listen(move |event| {
            let mut states = key_states_clone.lock().unwrap();
            match event.event_type {
                EventType::KeyPress(key) => {
                    states.insert(format!("{:?}", key)); // Сохраняем нажатие
                }
                EventType::KeyRelease(key) => {
                    states.remove(&format!("{:?}", key)); // Удаляем отпускание
                }
                _ => {}
            }
        })
        .unwrap();
    });

    // Запись в CSV файл каждые 50 мс (или по вашему усмотрению)
    let file_path = "../data/keys/key_events.csv";
    let file_exists = std::path::Path::new(file_path).exists();
    let mut writer = Writer::from_writer(
        OpenOptions::new()
            .append(true)
            .create(true)
            .open(file_path)
            .unwrap(),
    );

    // Записываем заголовки только если файл пуст
    if !file_exists
        || File::open(file_path)
            .map(BufReader::new)?
            .fill_buf()?
            .is_empty()
    {
        writer.write_record(&["Keys"]).unwrap();
    }

    loop {
        thread::sleep(Duration::from_millis(50));
        let states = key_states.lock().unwrap();

        if !states.is_empty() {
            // Объединяем все нажатия клавиш в одну строку
            let keys_pressed = states.iter().cloned().collect::<Vec<String>>().join(", "); // Объединяем нажатия в одну строку
            writer.write_record(&[keys_pressed]).unwrap();
            writer.flush().unwrap();
        }
    }
}
