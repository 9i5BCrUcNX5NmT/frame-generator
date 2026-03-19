use std::{fs, path::PathBuf, str::FromStr};

use image::DynamicImage;

use crate::State;

pub fn key_to_string(named: iced::keyboard::key::Named) -> String {
    match named {
        iced::keyboard::key::Named::Escape => "Escape".to_string(),
        iced::keyboard::key::Named::Enter => "Enter".to_string(),
        iced::keyboard::key::Named::Tab => "Tab".to_string(),
        iced::keyboard::key::Named::Space => "Space".to_string(),
        iced::keyboard::key::Named::Backspace => "Backspace".to_string(),
        iced::keyboard::key::Named::ArrowUp => "ArrowUp".to_string(),
        iced::keyboard::key::Named::ArrowDown => "ArrowDown".to_string(),
        iced::keyboard::key::Named::ArrowLeft => "ArrowLeft".to_string(),
        iced::keyboard::key::Named::ArrowRight => "ArrowRight".to_string(),
        iced::keyboard::key::Named::Home => "Home".to_string(),
        iced::keyboard::key::Named::End => "End".to_string(),
        iced::keyboard::key::Named::PageUp => "PageUp".to_string(),
        iced::keyboard::key::Named::PageDown => "PageDown".to_string(),
        iced::keyboard::key::Named::Insert => "Insert".to_string(),
        iced::keyboard::key::Named::Delete => "Delete".to_string(),
        iced::keyboard::key::Named::F1 => "F1".to_string(),
        iced::keyboard::key::Named::F2 => "F2".to_string(),
        iced::keyboard::key::Named::F3 => "F3".to_string(),
        iced::keyboard::key::Named::F4 => "F4".to_string(),
        iced::keyboard::key::Named::F5 => "F5".to_string(),
        iced::keyboard::key::Named::F6 => "F6".to_string(),
        iced::keyboard::key::Named::F7 => "F7".to_string(),
        iced::keyboard::key::Named::F8 => "F8".to_string(),
        iced::keyboard::key::Named::F9 => "F9".to_string(),
        iced::keyboard::key::Named::F10 => "F10".to_string(),
        iced::keyboard::key::Named::F11 => "F11".to_string(),
        iced::keyboard::key::Named::F12 => "F12".to_string(),
        iced::keyboard::key::Named::Shift => "Shift".to_string(),
        iced::keyboard::key::Named::Control => "Control".to_string(),
        iced::keyboard::key::Named::Alt => "Alt".to_string(),
        iced::keyboard::key::Named::Meta => "Meta".to_string(),
        _ => format!("{:?}", named),
    }
}

#[derive(Default)]
pub struct DataStatus {
    pub images_from_frames: bool,
    pub resized_images: bool,
    pub hdf5_files: bool,
    pub keys: bool,
}

pub fn check_data(state: &mut State) {
    let data_path = PathBuf::from_str("data").unwrap();

    state.data_status.hdf5_files = check_dir_not_empty(&data_path.join("hdf5_files"));
    state.data_status.keys = check_dir_not_empty(&data_path.join("keys"));

    let images_path = data_path.join("images");
    state.data_status.images_from_frames = check_dir_not_empty(&images_path.join("raw"));
    state.data_status.resized_images = check_dir_not_empty(&images_path.join("resized_images"));
}

/// Convert a DynamicImage to an iced image Handle (RGBA8)
pub fn dynamic_image_to_handle(img: &DynamicImage) -> iced::widget::image::Handle {
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    iced::widget::image::Handle::from_rgba(w, h, rgba.into_raw())
}

/// Load the first image from data/images/resized_images/ as the initial image
pub fn load_initial_image() -> (Option<DynamicImage>, Option<iced::widget::image::Handle>) {
    let image_dir = PathBuf::from_str("data/images/resized_images").unwrap();
    match get_first_file_in_directory(&image_dir) {
        Some(path) => match image::open(&path) {
            Ok(img) => {
                let handle = dynamic_image_to_handle(&img);
                (Some(img), Some(handle))
            }
            Err(_) => (None, None),
        },
        None => (None, None),
    }
}

/// Get the first file in a directory (sorted by name)
fn get_first_file_in_directory(dir: &PathBuf) -> Option<PathBuf> {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
        .map(|e| e.path())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

fn check_dir_not_empty(dir: &PathBuf) -> bool {
    fs::create_dir_all(dir).unwrap();

    match fs::read_dir(dir) {
        Ok(entries) => {
            for entry in entries {
                if let Ok(entry) = entry {
                    if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                        return true;
                    }
                }
            }
            false
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced::keyboard::key::Named;

    /// Test key_to_string for all named keys
    #[test]
    fn test_key_to_string_escape() {
        assert_eq!(key_to_string(Named::Escape), "Escape");
    }

    #[test]
    fn test_key_to_string_enter() {
        assert_eq!(key_to_string(Named::Enter), "Enter");
    }

    #[test]
    fn test_key_to_string_tab() {
        assert_eq!(key_to_string(Named::Tab), "Tab");
    }

    #[test]
    fn test_key_to_string_space() {
        assert_eq!(key_to_string(Named::Space), "Space");
    }

    #[test]
    fn test_key_to_string_backspace() {
        assert_eq!(key_to_string(Named::Backspace), "Backspace");
    }

    #[test]
    fn test_key_to_string_arrow_keys() {
        assert_eq!(key_to_string(Named::ArrowUp), "ArrowUp");
        assert_eq!(key_to_string(Named::ArrowDown), "ArrowDown");
        assert_eq!(key_to_string(Named::ArrowLeft), "ArrowLeft");
        assert_eq!(key_to_string(Named::ArrowRight), "ArrowRight");
    }

    #[test]
    fn test_key_to_string_home_end() {
        assert_eq!(key_to_string(Named::Home), "Home");
        assert_eq!(key_to_string(Named::End), "End");
    }

    #[test]
    fn test_key_to_string_page_keys() {
        assert_eq!(key_to_string(Named::PageUp), "PageUp");
        assert_eq!(key_to_string(Named::PageDown), "PageDown");
    }

    #[test]
    fn test_key_to_string_insert_delete() {
        assert_eq!(key_to_string(Named::Insert), "Insert");
        assert_eq!(key_to_string(Named::Delete), "Delete");
    }

    #[test]
    fn test_key_to_string_function_keys() {
        assert_eq!(key_to_string(Named::F1), "F1");
        assert_eq!(key_to_string(Named::F2), "F2");
        assert_eq!(key_to_string(Named::F3), "F3");
        assert_eq!(key_to_string(Named::F4), "F4");
        assert_eq!(key_to_string(Named::F5), "F5");
        assert_eq!(key_to_string(Named::F6), "F6");
        assert_eq!(key_to_string(Named::F7), "F7");
        assert_eq!(key_to_string(Named::F8), "F8");
        assert_eq!(key_to_string(Named::F9), "F9");
        assert_eq!(key_to_string(Named::F10), "F10");
        assert_eq!(key_to_string(Named::F11), "F11");
        assert_eq!(key_to_string(Named::F12), "F12");
    }

    #[test]
    fn test_key_to_string_modifier_keys() {
        assert_eq!(key_to_string(Named::Shift), "Shift");
        assert_eq!(key_to_string(Named::Control), "Control");
        assert_eq!(key_to_string(Named::Alt), "Alt");
        assert_eq!(key_to_string(Named::Meta), "Meta");
    }

    #[test]
    fn test_key_to_string_unknown_key() {
        // Unknown key should return debug format
        let result = key_to_string(Named::AudioVolumeDown);
        assert!(!result.is_empty());
    }

    /// Test DataStatus default
    #[test]
    fn test_data_status_default() {
        let status = DataStatus::default();
        assert!(!status.images_from_frames);
        assert!(!status.resized_images);
        assert!(!status.hdf5_files);
        assert!(!status.keys);
    }

    /// Test check_dir_not_empty with empty directory
    #[test]
    fn test_check_dir_not_empty_empty_dir() {
        let temp_dir = std::env::temp_dir().join("test_empty_dir_check");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let result = check_dir_not_empty(&temp_dir);

        assert!(!result);

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// Test check_dir_not_empty with file present
    #[test]
    fn test_check_dir_not_empty_with_file() {
        let temp_dir = std::env::temp_dir().join("test_file_dir_check");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        // Create a test file
        std::fs::write(temp_dir.join("test.txt"), "test").unwrap();

        let result = check_dir_not_empty(&temp_dir);

        assert!(result);

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// Test check_dir_not_empty creates directory if not exists
    #[test]
    fn test_check_dir_not_empty_creates_dir() {
        let temp_dir = std::env::temp_dir().join("nonexistent_test_dir_check_12345");
        let _ = fs::remove_dir_all(&temp_dir);

        let result = check_dir_not_empty(&temp_dir);

        assert!(!result);
        assert!(temp_dir.exists());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
