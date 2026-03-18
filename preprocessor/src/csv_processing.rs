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

pub fn key_to_num(key: &str) -> u8 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // === key_to_num tests ===

    #[test]
    fn test_key_to_num_modifiers() {
        assert_eq!(key_to_num("alt"), 0);
        assert_eq!(key_to_num("altgr"), 1);
        assert_eq!(key_to_num("backspace"), 2);
        assert_eq!(key_to_num("capslock"), 3);
        assert_eq!(key_to_num("controlleft"), 4);
        assert_eq!(key_to_num("control"), 4); // alias
        assert_eq!(key_to_num("controlright"), 5);
    }

    #[test]
    fn test_key_to_num_navigation() {
        assert_eq!(key_to_num("delete"), 6);
        assert_eq!(key_to_num("downarrow"), 7);
        assert_eq!(key_to_num("end"), 8);
        assert_eq!(key_to_num("escape"), 9);
        assert_eq!(key_to_num("home"), 22);
        assert_eq!(key_to_num("leftarrow"), 23);
        assert_eq!(key_to_num("pagedown"), 26);
        assert_eq!(key_to_num("pageup"), 27);
        assert_eq!(key_to_num("return"), 28);
        assert_eq!(key_to_num("rightarrow"), 29);
        assert_eq!(key_to_num("uparrow"), 34);
        assert_eq!(key_to_num("insert"), 87);
    }

    #[test]
    fn test_key_to_num_function_keys() {
        assert_eq!(key_to_num("f1"), 10);
        assert_eq!(key_to_num("f2"), 14);
        assert_eq!(key_to_num("f3"), 15);
        assert_eq!(key_to_num("f4"), 16);
        assert_eq!(key_to_num("f5"), 17);
        assert_eq!(key_to_num("f6"), 18);
        assert_eq!(key_to_num("f7"), 19);
        assert_eq!(key_to_num("f8"), 20);
        assert_eq!(key_to_num("f9"), 21);
        assert_eq!(key_to_num("f10"), 11);
        assert_eq!(key_to_num("f11"), 12);
        assert_eq!(key_to_num("f12"), 13);
    }

    #[test]
    fn test_key_to_num_modifier_keys() {
        assert_eq!(key_to_num("shiftleft"), 30);
        assert_eq!(key_to_num("shift"), 30); // alias
        assert_eq!(key_to_num("shiftright"), 31);
        assert_eq!(key_to_num("metaleft"), 24);
        assert_eq!(key_to_num("metaright"), 25);
        assert_eq!(key_to_num("space"), 32);
        assert_eq!(key_to_num("tab"), 33);
    }

    #[test]
    fn test_key_to_num_number_row() {
        assert_eq!(key_to_num("num1"), 40);
        assert_eq!(key_to_num("1"), 40); // alias
        assert_eq!(key_to_num("num2"), 41);
        assert_eq!(key_to_num("2"), 41);
        assert_eq!(key_to_num("num3"), 42);
        assert_eq!(key_to_num("3"), 42);
        assert_eq!(key_to_num("num4"), 43);
        assert_eq!(key_to_num("4"), 43);
        assert_eq!(key_to_num("num5"), 44);
        assert_eq!(key_to_num("5"), 44);
        assert_eq!(key_to_num("num6"), 45);
        assert_eq!(key_to_num("6"), 45);
        assert_eq!(key_to_num("num7"), 46);
        assert_eq!(key_to_num("7"), 46);
        assert_eq!(key_to_num("num8"), 47);
        assert_eq!(key_to_num("8"), 47);
        assert_eq!(key_to_num("num9"), 48);
        assert_eq!(key_to_num("9"), 48);
        assert_eq!(key_to_num("num0"), 49);
        assert_eq!(key_to_num("0"), 49);
    }

    #[test]
    fn test_key_to_num_special_keys() {
        assert_eq!(key_to_num("minus"), 50);
        assert_eq!(key_to_num("-"), 50);
        assert_eq!(key_to_num("equal"), 51);
        assert_eq!(key_to_num("="), 51);
        assert_eq!(key_to_num("backquote"), 39);
        assert_eq!(key_to_num("leftbracket"), 62);
        assert_eq!(key_to_num("rightbracket"), 63);
        assert_eq!(key_to_num("semicolon"), 73);
        assert_eq!(key_to_num("quote"), 74);
        assert_eq!(key_to_num("backslash"), 75);
        assert_eq!(key_to_num("intlbackslash"), 76);
        assert_eq!(key_to_num("comma"), 84);
        assert_eq!(key_to_num(","), 84);
        assert_eq!(key_to_num("dot"), 85);
        assert_eq!(key_to_num("."), 85);
        assert_eq!(key_to_num("slash"), 86);
        assert_eq!(key_to_num("/"), 86);
    }

    #[test]
    fn test_key_to_num_letter_keys_q_to_u() {
        assert_eq!(key_to_num("keyq"), 52);
        assert_eq!(key_to_num("q"), 52);
        assert_eq!(key_to_num("keyw"), 53);
        assert_eq!(key_to_num("w"), 53);
        assert_eq!(key_to_num("keye"), 54);
        assert_eq!(key_to_num("e"), 54);
        assert_eq!(key_to_num("keyr"), 55);
        assert_eq!(key_to_num("r"), 55);
        assert_eq!(key_to_num("keyt"), 56);
        assert_eq!(key_to_num("t"), 56);
        assert_eq!(key_to_num("keyy"), 57);
        assert_eq!(key_to_num("y"), 57);
        assert_eq!(key_to_num("keyu"), 58);
        assert_eq!(key_to_num("u"), 58);
    }

    #[test]
    fn test_key_to_num_letter_keys_i_to_p() {
        assert_eq!(key_to_num("keyi"), 59);
        assert_eq!(key_to_num("i"), 59);
        assert_eq!(key_to_num("keyo"), 60);
        assert_eq!(key_to_num("o"), 60);
        assert_eq!(key_to_num("keyp"), 61);
        assert_eq!(key_to_num("p"), 61);
    }

    #[test]
    fn test_key_to_num_letter_keys_a_to_l() {
        assert_eq!(key_to_num("keya"), 64);
        assert_eq!(key_to_num("a"), 64);
        assert_eq!(key_to_num("keys"), 65);
        assert_eq!(key_to_num("s"), 65);
        assert_eq!(key_to_num("keyd"), 66);
        assert_eq!(key_to_num("d"), 66);
        assert_eq!(key_to_num("keyf"), 67);
        assert_eq!(key_to_num("f"), 67);
        assert_eq!(key_to_num("keyg"), 68);
        assert_eq!(key_to_num("g"), 68);
        assert_eq!(key_to_num("keyh"), 69);
        assert_eq!(key_to_num("h"), 69);
        assert_eq!(key_to_num("keyj"), 70);
        assert_eq!(key_to_num("j"), 70);
        assert_eq!(key_to_num("keyk"), 71);
        assert_eq!(key_to_num("k"), 71);
        assert_eq!(key_to_num("keyl"), 72);
        assert_eq!(key_to_num("l"), 72);
    }

    #[test]
    fn test_key_to_num_letter_keys_z_to_m() {
        assert_eq!(key_to_num("keyz"), 77);
        assert_eq!(key_to_num("z"), 77);
        assert_eq!(key_to_num("keyx"), 78);
        assert_eq!(key_to_num("x"), 78);
        assert_eq!(key_to_num("keyc"), 79);
        assert_eq!(key_to_num("c"), 79);
        assert_eq!(key_to_num("keyv"), 80);
        assert_eq!(key_to_num("v"), 80);
        assert_eq!(key_to_num("keyb"), 81);
        assert_eq!(key_to_num("b"), 81);
        assert_eq!(key_to_num("keyn"), 82);
        assert_eq!(key_to_num("n"), 82);
        assert_eq!(key_to_num("keym"), 83);
        assert_eq!(key_to_num("m"), 83);
    }

    #[test]
    fn test_key_to_num_keypad() {
        assert_eq!(key_to_num("kp0"), 93);
        assert_eq!(key_to_num("kp1"), 94);
        assert_eq!(key_to_num("kp2"), 95);
        assert_eq!(key_to_num("kp3"), 96);
        assert_eq!(key_to_num("kp4"), 97);
        assert_eq!(key_to_num("kp5"), 98);
        assert_eq!(key_to_num("kp6"), 99);
        assert_eq!(key_to_num("kp7"), 100);
        assert_eq!(key_to_num("kp8"), 101);
        assert_eq!(key_to_num("kp9"), 102);
        assert_eq!(key_to_num("kpminus"), 89);
        assert_eq!(key_to_num("kpplus"), 90);
        assert_eq!(key_to_num("kpmultiply"), 91);
        assert_eq!(key_to_num("kpdivide"), 92);
        assert_eq!(key_to_num("kpdelete"), 103);
    }

    #[test]
    fn test_key_to_num_special_buttons() {
        assert_eq!(key_to_num("printscreen"), 35);
        assert_eq!(key_to_num("scrolllock"), 36);
        assert_eq!(key_to_num("pause"), 37);
        assert_eq!(key_to_num("numlock"), 38);
        assert_eq!(key_to_num("function"), 104);
    }

    #[test]
    fn test_key_to_num_mouse_buttons() {
        assert_eq!(key_to_num("left"), 106);
        assert_eq!(key_to_num("right"), 107);
        assert_eq!(key_to_num("middle"), 108);
    }

    #[test]
    fn test_key_to_num_case_insensitive() {
        assert_eq!(key_to_num("ESCAPE"), 9);
        assert_eq!(key_to_num("Escape"), 9);
        assert_eq!(key_to_num("escape"), 9);
        assert_eq!(key_to_num("KEYQ"), 52);
        assert_eq!(key_to_num("KeyQ"), 52);
    }

    #[test]
    fn test_key_to_num_unknown_key() {
        // Unknown keys should return 110
        assert_eq!(key_to_num("unknownkey"), 110);
        assert_eq!(key_to_num("someweirdkey"), 110);
        assert_eq!(key_to_num(""), 110);
    }

    #[test]
    fn test_mouse_to_num_valid() {
        assert_eq!(mouse_to_num("100,200"), [100, 200]);
        assert_eq!(mouse_to_num("0,0"), [0, 0]);
        assert_eq!(mouse_to_num("-50,-100"), [-50, -100]);
    }

    // === load_records_from_directory tests ===

    /// Test load_records_from_directory with empty directory
    #[test]
    fn test_load_records_empty_directory() {
        let temp_dir = std::env::temp_dir().join("test_empty_csv");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let result = load_records_from_directory(&temp_dir);

        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// Test load_records_from_directory with valid CSV
    #[test]
    fn test_load_records_valid_csv() {
        let temp_dir = std::env::temp_dir().join("test_valid_csv");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        // Create a valid CSV file using csv::Writer
        let csv_path = temp_dir.join("test.csv");
        {
            let file = fs::File::create(&csv_path).unwrap();
            let mut writer = csv::Writer::from_writer(file);
            writer.serialize(("keys", "mouse")).unwrap();
            writer.serialize(("KeyQ", "100,200")).unwrap();
            writer.serialize(("KeyW", "150,250")).unwrap();
            writer.flush().unwrap();
        }

        let result = load_records_from_directory(&temp_dir);

        assert!(result.is_ok());
        let records = result.unwrap();
        assert_eq!(records.len(), 2);

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// Test load_records_from_directory with empty CSV
    #[test]
    fn test_load_records_empty_csv() {
        let temp_dir = std::env::temp_dir().join("test_empty_csv_file");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        // Create an empty CSV file
        let csv_path = temp_dir.join("empty.csv");
        let mut file = fs::File::create(&csv_path).unwrap();
        writeln!(file, "keys,mouse").unwrap();
        // No data rows

        let result = load_records_from_directory(&temp_dir);

        assert!(result.is_ok());
        let records = result.unwrap();
        assert!(records.is_empty());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// Test parse_csv_record with single key
    #[test]
    fn test_parse_csv_record_single_key() {
        let record = CsvRecord {
            keys: "q".to_string(),
            mouse: "100,200".to_string(),
        };

        let result = parse_csv_record(record);

        assert_eq!(result.keys[0], 52); // 'q' -> 52
        assert_eq!(result.mouse[0], [100, 200]);
    }

    /// Test parse_csv_record with multiple keys
    #[test]
    fn test_parse_csv_record_multiple_keys() {
        let record = CsvRecord {
            keys: "q, w, e".to_string(),
            mouse: "100,200, 300,400".to_string(),
        };

        let result = parse_csv_record(record);

        assert_eq!(result.keys[0], 52); // 'q' -> 52
        assert_eq!(result.keys[1], 53); // 'w' -> 53
        assert_eq!(result.keys[2], 54); // 'e' -> 54
    }

    /// Test parse_csv_record with empty keys
    #[test]
    fn test_parse_csv_record_empty_keys() {
        let record = CsvRecord {
            keys: "".to_string(),
            mouse: "100,200".to_string(),
        };

        let result = parse_csv_record(record);

        assert_eq!(result.keys[0], 0); // Empty becomes 0
        assert_eq!(result.mouse[0], [100, 200]);
    }

    /// Test parse_csv_record fills rest with zeros
    #[test]
    fn test_parse_csv_record_zeros_remainder() {
        let record = CsvRecord {
            keys: "q".to_string(),
            mouse: "100,200".to_string(),
        };

        let result = parse_csv_record(record);

        // First position should have values
        assert_eq!(result.keys[0], 52);
        // Rest should be zeros
        assert_eq!(result.keys[1], 0);
        assert_eq!(result.keys[100], 0);
        assert_eq!(result.keys[199], 0);
    }

    /// Test KeysRecordConst has correct size
    #[test]
    fn test_keys_record_const_size() {
        let record = KeysRecordConst {
            keys: [0; 200],
            mouse: [[0; 2]; 200],
        };

        // Just verify it can be created
        assert_eq!(record.keys.len(), 200);
        assert_eq!(record.mouse.len(), 200);
    }
}
