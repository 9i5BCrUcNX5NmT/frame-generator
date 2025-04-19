pub fn get_first_file_in_directory(dir_path: &str) -> std::io::Result<Option<std::path::PathBuf>> {
    let path = std::path::Path::new(dir_path);
    let entries = std::fs::read_dir(path)?;

    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;

        if file_type.is_file() {
            return Ok(Some(entry.path()));
        }
    }

    Ok(None) // Если файлов нет, возвращаем None
}
