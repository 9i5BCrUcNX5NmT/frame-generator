// TODO: перенести в предобработку
use std::{fs, io, path::PathBuf};

use hdf5_metno::H5Type;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct CsvRecord {
    keys: String,
    mouse: String,
}

// #[derive(Clone, Debug)]
// pub struct KeysRecord {
//     pub keys: Vec<u8>,        // преобразованные названия клавиш в числа
//     pub mouse: Vec<[i32; 2]>, // Движения и скроллинг мыши
// }

#[derive(Clone, Debug, H5Type)]
#[repr(C)]
pub struct KeysRecordConst {
    pub keys: [u8; 200],        // преобразованные названия клавиш в числа
    pub mouse: [[i32; 2]; 200], // Движения и скроллинг мыши
}

/// Получение всех записей из всех файлов в директории
pub fn load_records_from_directory(dir: &PathBuf) -> io::Result<Vec<KeysRecordConst>> {
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

fn parse_csv_record(record: CsvRecord) -> KeysRecordConst {
    let keys: Vec<&str> = record.keys.split(", ").collect();
    let mouse: Vec<&str> = record.mouse.split(", ").collect();

    if keys.len() > 200 || mouse.len() > 200 {
        panic!("Несоответствие размеров")
    }

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

    let mut keys_const = [0; 200];
    let mut mouse_const = [[0; 2]; 200];

    for (i, value) in keys.iter().enumerate() {
        keys_const[i] = *value;
    }

    for (i, value) in mouse.iter().enumerate() {
        mouse_const[i] = *value;
    }

    KeysRecordConst {
        keys: keys_const,
        mouse: mouse_const,
    }
}

fn mouse_to_num(s: &str) -> [i32; 2] {
    let values: Vec<i32> = s.split(",").map(|s| s.parse::<i32>().unwrap()).collect();

    [values[0], values[1]]
}

fn key_to_num(key: &str) -> u8 {
    let key = key.to_lowercase();

    match key.as_str() {
        "alt" => 0,
        "altgr" => 1,
        "backspace" => 2,
        "capslock" => 3,
        "controlleft" | "control" => 4,
        "controlright" => 5,
        "delete" => 6,
        "downarrow" => 7,
        "end" => 8,
        "escape" => 9,
        "f1" => 10,
        "f10" => 11,
        "f11" => 12,
        "f12" => 13,
        "f2" => 14,
        "f3" => 15,
        "f4" => 16,
        "f5" => 17,
        "f6" => 18,
        "f7" => 19,
        "f8" => 20,
        "f9" => 21,
        "home" => 22,
        "leftarrow" => 23,
        "metaleft" => 24,
        "metaright" => 25,
        "pagedown" => 26,
        "pageup" => 27,
        "return" => 28,
        "rightarrow" => 29,
        "shiftleft" | "shift" => 30,
        "shiftright" => 31,
        "space" => 32,
        "tab" => 33,
        "uparrow" => 34,
        "printscreen" => 35,
        "scrolllock" => 36,
        "pause" => 37,
        "numlock" => 38,
        "backquote" => 39,
        "num1" | "1" => 40,
        "num2" | "2" => 41,
        "num3" | "3" => 42,
        "num4" | "4" => 43,
        "num5" | "5" => 44,
        "num6" | "6" => 45,
        "num7" | "7" => 46,
        "num8" | "8" => 47,
        "num9" | "9" => 48,
        "num0" | "0" => 49,
        "minus" | "-" => 50,
        "equal" | "=" => 51,
        "keyq" | "q" => 52,
        "keyw" | "w" => 53,
        "keye" | "e" => 54,
        "keyr" | "r" => 55,
        "keyt" | "t" => 56,
        "keyy" | "y" => 57,
        "keyu" | "u" => 58,
        "keyi" | "i" => 59,
        "keyo" | "o" => 60,
        "keyp" | "p" => 61,
        "leftbracket" => 62,
        "rightbracket" => 63,
        "keya" | "a" => 64,
        "keys" | "s" => 65,
        "keyd" | "d" => 66,
        "keyf" | "f" => 67,
        "keyg" | "g" => 68,
        "keyh" | "h" => 69,
        "keyj" | "j" => 70,
        "keyk" | "k" => 71,
        "keyl" | "l" => 72,
        "semicolon" => 73,
        "quote" => 74,
        "backslash" => 75,
        "intlbackslash" => 76,
        "keyz" | "z" => 77,
        "keyx" | "x" => 78,
        "keyc" | "c" => 79,
        "keyv" | "v" => 80,
        "keyb" | "b" => 81,
        "keyn" | "n" => 82,
        "keym" | "m" => 83,
        "comma" | "," => 84,
        "dot" | "." => 85,
        "slash" | "/" => 86,
        "insert" => 87,
        "kpretun" => 88,
        "kpminus" => 89,
        "kpplus" => 90,
        "kpmultiply" => 91,
        "kpdivide" => 92,
        "kp0" => 93,
        "kp1" => 94,
        "kp2" => 95,
        "kp3" => 96,
        "kp4" => 97,
        "kp5" => 98,
        "kp6" => 99,
        "kp7" => 100,
        "kp8" => 101,
        "kp9" => 102,
        "kpdelete" => 103,
        "function" => 104,
        "unknown(u32)" => 105, // todo: рассмотреть вариант
        "left" => 106,
        "right" => 107,
        "middle" => 108,
        "unknown(u8)" => 109,
        _ => 110,
    }
}
