// TODO: перенести в предобработку
use std::{fs, io};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct CsvRecord {
    keys: String,
    mouse: String,
}

#[derive(Clone, Debug)]
pub struct KeysRecord {
    pub keys: Vec<u8>,        // преобразованные названия клавиш в числа
    pub mouse: Vec<[i32; 2]>, // Движения и скроллинг мыши
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
                let keys_record = parse_csv_record(record);

                dataset.push(keys_record);
            }
        }
    }

    Ok(dataset)
}

fn parse_csv_record(record: CsvRecord) -> KeysRecord {
    let keys: Vec<&str> = record.keys.split(", ").collect();
    let mouse: Vec<&str> = record.mouse.split(", ").collect();

    let keys = if keys.len() > 1 {
        keys.iter().map(|&s| key_to_num(s)).collect()
    } else {
        if record.keys == "" {
            vec![]
        } else {
            vec![key_to_num(&record.keys)]
        }
    };

    let mouse = if mouse.len() > 1 {
        mouse.iter().map(|&s| mouse_to_num(s)).collect()
    } else {
        if record.mouse == "" {
            vec![]
        } else {
            vec![mouse_to_num(&record.mouse)]
        }
    };

    KeysRecord { keys, mouse }
}

fn mouse_to_num(s: &str) -> [i32; 2] {
    let values: Vec<i32> = s.split(",").map(|s| s.parse::<i32>().unwrap()).collect();

    [values[0], values[1]]
}

fn key_to_num(key: &str) -> u8 {
    match key {
        "Alt" => 0,
        "AltGr" => 1,
        "Backspace" => 2,
        "CapsLock" => 3,
        "ControlLeft" => 4,
        "ControlRight" => 5,
        "Delete" => 6,
        "DownArrow" => 7,
        "End" => 8,
        "Escape" => 9,
        "F1" => 10,
        "F10" => 11,
        "F11" => 12,
        "F12" => 13,
        "F2" => 14,
        "F3" => 15,
        "F4" => 16,
        "F5" => 17,
        "F6" => 18,
        "F7" => 19,
        "F8" => 20,
        "F9" => 21,
        "Home" => 22,
        "LeftArrow" => 23,
        "MetaLeft" => 24,
        "MetaRight" => 25,
        "PageDown" => 26,
        "PageUp" => 27,
        "Return" => 28,
        "RightArrow" => 29,
        "ShiftLeft" => 30,
        "ShiftRight" => 31,
        "Space" => 32,
        "Tab" => 33,
        "UpArrow" => 34,
        "PrintScreen" => 35,
        "ScrollLock" => 36,
        "Pause" => 37,
        "NumLock" => 38,
        "BackQuote" => 39,
        "Num1" => 40,
        "Num2" => 41,
        "Num3" => 42,
        "Num4" => 43,
        "Num5" => 44,
        "Num6" => 45,
        "Num7" => 46,
        "Num8" => 47,
        "Num9" => 48,
        "Num0" => 49,
        "Minus" => 50,
        "Equal" => 51,
        "KeyQ" => 52,
        "KeyW" => 53,
        "KeyE" => 54,
        "KeyR" => 55,
        "KeyT" => 56,
        "KeyY" => 57,
        "KeyU" => 58,
        "KeyI" => 59,
        "KeyO" => 60,
        "KeyP" => 61,
        "LeftBracket" => 62,
        "RightBracket" => 63,
        "KeyA" => 64,
        "KeyS" => 65,
        "KeyD" => 66,
        "KeyF" => 67,
        "KeyG" => 68,
        "KeyH" => 69,
        "KeyJ" => 70,
        "KeyK" => 71,
        "KeyL" => 72,
        "SemiColon" => 73,
        "Quote" => 74,
        "BackSlash" => 75,
        "IntlBackslash" => 76,
        "KeyZ" => 77,
        "KeyX" => 78,
        "KeyC" => 79,
        "KeyV" => 80,
        "KeyB" => 81,
        "KeyN" => 82,
        "KeyM" => 83,
        "Comma" => 84,
        "Dot" => 85,
        "Slash" => 86,
        "Insert" => 87,
        "KpReturn" => 88,
        "KpMinus" => 89,
        "KpPlus" => 90,
        "KpMultiply" => 91,
        "KpDivide" => 92,
        "Kp0" => 93,
        "Kp1" => 94,
        "Kp2" => 95,
        "Kp3" => 96,
        "Kp4" => 97,
        "Kp5" => 98,
        "Kp6" => 99,
        "Kp7" => 100,
        "Kp8" => 101,
        "Kp9" => 102,
        "KpDelete" => 103,
        "Function" => 104,
        "Unknown(u32)" => 105, // TODO: Рассмотреть вариант
        "Left" => 106,
        "Right" => 107,
        "Middle" => 108,
        "Unknown(u8)" => 109,
        _ => 110,
    }
}
