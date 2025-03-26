use std::thread;

use iced::keyboard::{Key, Modifiers, on_key_press};
use iced::widget::{button, column, container, image, mouse_area, row, text};
use iced::{Alignment, Length, Size};
use iced::{Element, Point, Subscription, Theme};
use utils::{DataStatus, check_data, generate_frame, key_to_string};

mod utils;

#[derive(Debug, Clone)]
enum Message {
    Key(String),
    Mouse(Point<f32>),
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
    pub mouse_position: Point<f32>,
    pub image: Option<image::Handle>,
    pub data_status: DataStatus,
    pub message_to_user: String,
}

fn view(state: &State) -> Element<Message> {
    let content = column![
        row![
            column![
                text(format!("{}", state.mouse_position.clone())),
                row![text("Key: "), text(state.pressed_key.clone())],
            ]
            .spacing(10),
            column![
                text("Status"),
                // text(format!("Записанное видео: {}", state.data_status.video)),
                text(format!(
                    "Преобразование кадров: {}",
                    state.data_status.resized_images
                )),
                text(format!("hdf5 файлы: {}", state.data_status.hdf5_files)),
                text(format!("Keys: {}", state.data_status.keys)),
            ]
            .spacing(10),
            column![text("Log"), text(state.message_to_user.clone()),].spacing(10)
        ]
        .spacing(10),
        match &state.image {
            Some(image_handle) => image(image_handle),
            None => image(""),
        }
        .width(Length::Fill)
        .height(Length::Fill),
        column![
            row![
                button(text("Генерация")).on_press(Message::GenerateImage),
                button(text("Сбросить изображение")).on_press(Message::ReloadImage),
                button(text("Перезагрузить статус")).on_press(Message::CheckData),
            ],
            row![
                button(text("Запись")).on_press(Message::Record),
                button(text("Обработать кадры")).on_press(Message::FramesProcessing),
                button(text("Записать в hdf5")).on_press(Message::WriteData),
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
        Key::Named(named) => key_to_string(named),
        Key::Character(character) => character.to_string(),
        Key::Unidentified => "".to_string(),
    };

    Some(Message::Key(key))
}

fn keyboard_subscription(_state: &State) -> Subscription<Message> {
    on_key_press(capture_key)
}

fn update(state: &mut State, message: Message) {
    match message {
        Message::Key(key) => state.pressed_key = key,
        Message::Mouse(point) => state.mouse_position = point,
        Message::GenerateImage => {
            let keys = vec![state.pressed_key.clone()];
            let mouse = vec![state.mouse_position];

            state.image = Some(match &state.image {
                Some(image_handle) => generate_frame(image_handle, keys, mouse),
                None => generate_frame(
                    &image::Handle::from_path("data/images/resized_images/image-0.png"),
                    keys,
                    mouse,
                ),
            })
        }
        Message::ModelTraining => {
            thread::spawn(|| model_training::training::run());
        }
        Message::Record => {
            thread::spawn(|| recorder::run());
        }
        // Message::VideoProcessing => {
        //     thread::spawn(|| preprocessor::process_my_videos());
        // }
        Message::FramesProcessing => {
            thread::spawn(|| preprocessor::process_my_images());
        }
        Message::ReloadImage => {
            state.image = Some(image::Handle::from_path(
                "data/images/resized_images/image-0.png",
            ))
        }
        Message::WriteData => preprocessor::write_my_data(),
        Message::CheckData => check_data(state),
    };
}

fn theme(_state: &State) -> Theme {
    Theme::CatppuccinLatte
}

pub fn main() -> iced::Result {
    iced::application("Генерация |>_<|", update, view)
        .window_size(Size::new(800.0, 600.0))
        .theme(theme)
        .subscription(keyboard_subscription)
        .run()
}
