use std::{fs, io};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct CsvRecord {
    keys: String,
    mouse: String,
    buttons: String,
}

pub struct KeysRecord {
    keys: Vec<u8>,        // преобразованные названия клавиш в числа
    mouse: Vec<[i32; 2]>, // Движения и скроллинг мыши
    buttons: Vec<u8>,
}

fn xy_to_int(s: &str) -> [i32; 2] {
    let values: Vec<i32> = s.split(",").map(|s| s.parse::<i32>().unwrap()).collect();

    [values[0], values[1]]
}

fn key_to_int(key: &str) -> u8 {
    let v: rdev::Key = serde_json::from_str(key).unwrap();
    todo!("Превращение key и button в число")
}

fn button_to_int(key: &str) -> u8 {
    let v: rdev::Key = serde_json::from_str(key).unwrap();
    todo!("Превращение key и button в число")
}

pub fn load_keys_from_directory(dir: &str) -> io::Result<Vec<KeysRecord>> {
    let mut dataset = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            let mut reader = csv::Reader::from_path(path).unwrap();
            for result in reader.deserialize() {
                let record: CsvRecord = result.unwrap();

                // dbg!(&record);

                let keys: Vec<&str> = record.keys.split(", ").collect();
                let buttons: Vec<&str> = record.buttons.split(", ").collect();
                let mouse: Vec<&str> = record.mouse.split(", ").collect();

                let keys = if keys.len() > 1 {
                    keys.iter().map(|&s| key_to_int(s)).collect()
                } else {
                    vec![key_to_int(&record.keys)]
                };

                let mouse = if mouse.len() > 1 {
                    mouse.iter().map(|&s| xy_to_int(s)).collect()
                } else {
                    vec![xy_to_int(&record.mouse)]
                };

                let buttons = if buttons.len() > 1 {
                    buttons.iter().map(|&s| button_to_int(s)).collect()
                } else {
                    vec![button_to_int(&record.buttons)]
                };

                todo!();

                dataset.push(KeysRecord {
                    keys,
                    mouse,
                    buttons,
                });
            }
        }
    }

    Ok(dataset)
}
