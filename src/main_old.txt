use std::io;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{self, ClearType},
};
use image::RgbaImage;
use tui::widgets;
use tui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph, Row, Table},
};
use utils::check_data;
use utils::generate_frame;

mod utils;

#[derive(Debug, Clone)]
enum Message {
    Key(String),
    Mouse((u16, u16)), // Mouse position can be represented as (x, y)
    GenerateImage,
    ReloadImage,
    ModelTraining,
    Record,
    FramesProcessing,
    WriteData,
    CheckData,
}

#[derive(Default)]
struct State {
    pub pressed_key: String,
    pub mouse_position: (u16, u16),
    pub image: Option<RgbaImage>, // Placeholder for image path or data
    pub data_status: utils::DataStatus,
    pub message_to_user: String,

    image_path: Option<PathBuf>,
}
fn draw_ui<B: tui::backend::Backend>(f: &mut tui::Frame<B>, state: &State) {
    let size = f.size();
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints::<Vec<Constraint>>(vec![
            Constraint::Percentage(20),
            Constraint::Percentage(60),
            Constraint::Percentage(20),
        ]);

    f.render_widget(Block::default().title("Status").borders(Borders::ALL), size);

    let chunks = layout.split(size);
    let status_text = format!(
        "Mouse Position: {:?}\nKey: {}\nStatus: {}\nHDF5 Files: {}\nKeys: {}",
        state.mouse_position,
        state.pressed_key,
        state.data_status.resized_images,
        state.data_status.hdf5_files,
        state.data_status.keys,
    );

    let status_paragraph =
        Paragraph::new(status_text).block(Block::default().borders(Borders::ALL).title("Status"));
    f.render_widget(status_paragraph, chunks[0]);

    widgets

    // Placeholder for image display
    let image_paragraph = Paragraph::new(
        state
            .image
            .clone()
            .unwrap_or_else(|| "No Image".to_string()),
    )
    .block(Block::default().borders(Borders::ALL).title("Image"));
    f.render_widget(image_paragraph, chunks[1]);

    // Buttons (as text for simplicity)
    let buttons = vec![
        "Generate Image",
        "Reload Image",
        "Check Data",
        "Record",
        "Process Frames",
        "Write Data",
        "Model Training",
    ];
    let button_rows: Vec<Row> = buttons
        .iter()
        .map(|&btn| Row::new(vec![btn]).style(Style::default().fg(Color::White)))
        .collect();

    let button_table =
        Table::new(button_rows).block(Block::default().borders(Borders::ALL).title("Actions"));
    f.render_widget(button_table, chunks[2]);
}

fn capture_key(key: KeyEvent) -> Option<Message> {
    let key_string = match key.code {
        KeyCode::Char(c) => c.to_string(),
        KeyCode::Esc => "Esc".to_string(),
        _ => "".to_string(),
    };

    Some(Message::Key(key_string))
}
fn update(state: &mut State, message: Message) {
    match message {
        Message::Key(key) => {
            state.pressed_key = key;
            // Дополнительная логика, если необходимо
        }
        Message::Mouse(point) => {
            state.mouse_position = point;
            // Дополнительная логика, если необходимо
        }
        Message::GenerateImage => {
            // Логика генерации изображения
            let keys = vec![state.pressed_key.clone()];
            let mouse = vec![state.mouse_position.clone()]; // Убедитесь, что mouse_position имеет тип Point

            // Здесь вы можете вызвать вашу функцию генерации изображения
            if let Some(image_path) = &state.image_path {
                // Предполагаем, что у вас есть поле image_path в State
                state.image = Some(generate_frame(image_path, keys, mouse));
                state.message_to_user = "Изображение сгенерировано.".to_string();
            } else {
                state.message_to_user = "Ошибка: путь к изображению не задан.".to_string();
            }
        }
        Message::ModelTraining => {
            thread::spawn(|| {
                // Логика тренировки модели
                model_training::training::run();
            });
            state.message_to_user = "Начата тренировка модели.".to_string();
        }
        Message::Record => {
            thread::spawn(|| {
                // Логика записи
                recorder::run();
            });
            state.message_to_user = "Начата запись.".to_string();
        }
        Message::FramesProcessing => {
            thread::spawn(|| {
                // Логика обработки кадров
                preprocessor::process_my_images();
            });
            state.message_to_user = "Начата обработка кадров.".to_string();
        }
        Message::ReloadImage => {
            // Логика перезагрузки изображения
            state.image_path = Some(PathBuf::from("data/images/resized_images/image-0.png")); // Предполагаем, что у вас есть поле image_path в State
            state.message_to_user = "Изображение перезагружено.".to_string();
        }
        Message::WriteData => {
            // Логика записи данных
            preprocessor::write_my_data();
            state.message_to_user = "Данные записаны.".to_string();
        }
        Message::CheckData => {
            // Логика проверки данных
            check_data(state);
            state.message_to_user = "Проверка данных завершена.".to_string();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    terminal.clear()?;

    let mut state = State::default();
    let mut last_tick = Instant::now();
    loop {
        // Обработка событий
        if event::poll(Duration::from_millis(100))? {
            if let event::Event::Key(key_event) = event::read()? {
                if let Some(message) = capture_key(key_event) {
                    update(&mut state, message);
                }
                // Выход из приложения при нажатии Esc
                if key_event.code == KeyCode::Esc {
                    break;
                }
            }
        }

        // Отрисовка интерфейса
        terminal.draw(|f| draw_ui(f, &state))?;

        // Обновление состояния (например, можно добавить таймер или другие обновления)
        if last_tick.elapsed() >= Duration::from_secs(1) {
            // Здесь можно обновить состояние, если это необходимо
            last_tick = Instant::now();
        }
    }

    // Завершение работы
    terminal.clear()?;
    Ok(())
}
