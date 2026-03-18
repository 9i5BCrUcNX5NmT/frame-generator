use std::thread;

use iced::keyboard::{on_key_press, Key, Modifiers};
use iced::widget::{button, column, container, mouse_area, row, text};
use iced::{Alignment, Element, Length, Size, Subscription, Theme};

mod utils;

#[derive(Debug, Clone)]
enum Message {
    Key(String),
    Mouse(iced::Point),
    GenerateImage,
    ReloadImage,
    ModelTraining,
    Record,
    StopRecord,
    Postprocess,
    CheckData,
}

#[derive(Default)]
pub struct State {
    pub pressed_key: String,
    pub mouse_position: iced::Point,
    pub data_status: utils::DataStatus,
    pub message_to_user: String,
}

fn main() -> iced::Result {
    iced::application("Генерация |>_<|", update, view)
        .window_size(Size::new(800.0, 600.0))
        .theme(|_state| Theme::GruvboxDark)
        .subscription(keyboard_subscription)
        .run()
}

fn update(state: &mut State, message: Message) {
    match message {
        Message::Key(key) => state.pressed_key = key,
        Message::Mouse(point) => state.mouse_position = point,
        Message::GenerateImage => {
            state.message_to_user = "GenerateImage clicked".to_string();
        }
        Message::ReloadImage => {
            state.message_to_user = "ReloadImage clicked".to_string();
        }
        Message::ModelTraining => {
            state.message_to_user = "Starting training...".to_string();
            thread::spawn(|| model_training::training::run());
        }
        Message::Record => {
            state.message_to_user = "Starting recording...".to_string();
            thread::spawn(|| recorder::run());
        }
        Message::StopRecord => {
            state.message_to_user = "Stop recording clicked".to_string();
        }
        Message::Postprocess => {
            state.message_to_user = "Postprocessing...".to_string();
            thread::spawn(|| {
                preprocessor::process_my_images();
                preprocessor::write_my_data();
            });
        }
        Message::CheckData => {
            utils::check_data(state);
            state.message_to_user = "Data status updated".to_string();
        }
    };
}

fn view(state: &State) -> Element<'_, Message> {
    let content = column![
        // Top row: mouse, keys, status, log
        row![
            column![
                text(format!("{:?}", state.mouse_position)),
                row![text("Key: "), text(&state.pressed_key)],
            ]
            .spacing(10),
            column![
                text("Status"),
                text(format!(
                    "Преобразование кадров: {}",
                    state.data_status.resized_images
                )),
                text(format!("hdf5 файлы: {}", state.data_status.hdf5_files)),
                text(format!("Keys: {}", state.data_status.keys)),
            ]
            .spacing(10),
            column![text("Log"), text(&state.message_to_user),].spacing(10)
        ]
        .spacing(10),
        // Image placeholder
        container(text("Изображение (нажмите Генерация)"))
            .width(Length::Fill)
            .height(Length::Fill),
        // Buttons - matching original UI
        column![
            row![
                button(text("Генерация")).on_press(Message::GenerateImage),
                button(text("Сбросить изображение")).on_press(Message::ReloadImage),
                button(text("Перезагрузить статус")).on_press(Message::CheckData),
            ],
            row![
                button(text("Запись")).on_press(Message::Record),
                button(text("Стоп запись")).on_press(Message::StopRecord),
                button(text("Постобработка")).on_press(Message::Postprocess),
            ],
            row![button(text("Тренировка")).on_press(Message::ModelTraining),]
        ]
        .spacing(10)
    ]
    .align_x(Alignment::End)
    .spacing(10)
    .padding(10);

    mouse_area(
        container(content)
            .center_x(Length::Fill)
            .center_y(Length::Fill),
    )
    .on_move(|point| Message::Mouse(point))
    .into()
}

fn capture_key(key: Key, _modifiers: Modifiers) -> Option<Message> {
    let key = match key {
        Key::Named(named) => utils::key_to_string(named),
        Key::Character(character) => character.to_string(),
        Key::Unidentified => "".to_string(),
    };

    Some(Message::Key(key))
}

fn keyboard_subscription(_state: &State) -> Subscription<Message> {
    on_key_press(capture_key)
}
