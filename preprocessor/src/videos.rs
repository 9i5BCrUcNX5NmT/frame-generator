use std::process::Command;

pub fn process_videos(video_dir: &str, result_images_dir: &str) {
    // Создаем команду
    // ffmpeg -i input_video.mp4 -vf "fps=20" output_%04d.png
    let output = Command::new("ffmpeg")
        .args([
            "-i",
            video_dir,
            "-vf",
            "fps=20",
            &format!("{result_images_dir}out_%04d.png"),
        ])
        .output() // Выполняем команду и получаем вывод
        .expect("Не удалось выполнить команду");

    // Проверяем статус выполнения
    if output.status.success() {
        // Преобразуем вывод в строку
        let stdout = String::from_utf8_lossy(&output.stdout);
        println!("Вывод: {}", stdout);
    } else {
        // Если команда завершилась с ошибкой, выводим сообщение об ошибке
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("Ошибка: {}", stderr);
    }
}
