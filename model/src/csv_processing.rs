use std::{fs, io};

use serde::Serialize;

#[derive(Debug, Serialize)]
struct CsvRecord {
    keys: String,
    mouse: String,
}

struct KeysRecord {
    keys: Vec<u8>,   // преобразованные названия клавиш в числа
    mouse: Vec<f64>, // Движения и скроллинг мыши
}

pub fn load_keys_from_directory(dir: &str) -> io::Result<Vec<KeysRecord>> {
    let mut dataset = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            let values = todo!("парсинг");

            dataset.push(values);
        }
    }

    Ok(dataset)
}
