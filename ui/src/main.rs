use std::thread;

use iced::keyboard::{on_key_press, Key, Modifiers};
use iced::widget::{button, column, container, image, mouse_area, row, text};
use iced::Length;
use iced::{Element, Point, Subscription, Theme};
use utils::{generate_frame, key_to_string};

mod utils;

#[derive(Debug, Clone)]
enum Message {
    Key(String),
    Mouse(Point<f32>),
    GenerateImage,
    ReloadImage,
    ModelTraining,
    Record,
    VideoProcessing,
    FramesProcessing,
    WriteData,
    ReadData,
}

#[derive(Default)]
struct State {
    pub pressed_key: String,
    pub mouse_position: Point<f32>,
    pub image: Option<image::Handle>,
}

fn view(state: &State) -> Element<Message> {
    let content = column![
        button(text(state.pressed_key.clone())),
        button(text(format!("{}", state.mouse_position.clone()))),
        match &state.image {
            Some(image_handle) => image(image_handle),
            None => image(""),
        }
        // .content_fit(iced::ContentFit::Fill),
        .width(Length::Fill)
        .height(Length::Fill),
        column![
            row![
                button(text("Генерация")).on_press(Message::GenerateImage),
                button(text("Сбросить изображение")).on_press(Message::ReloadImage),
            ],
            row![
                button(text("Запись")).on_press(Message::Record),
                button(text("Извлечь кадры из видео")).on_press(Message::VideoProcessing),
                button(text("Обработать кадры")).on_press(Message::FramesProcessing),
            ],
            row![
                button(text("Тренировка")).on_press(Message::ModelTraining),
                button(text("WriteData")).on_press(Message::WriteData),
                button(text("ReadData")).on_press(Message::ReadData),
            ]
        ]
        .spacing(20)
    ]
    .spacing(20);

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
                    &image::Handle::from_path("data/images/test/out_0001.png"),
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
        Message::VideoProcessing => {
            thread::spawn(|| preprocessor::process_my_videos());
        }
        Message::FramesProcessing => {
            thread::spawn(|| preprocessor::process_my_images());
        }
        Message::ReloadImage => {
            state.image = Some(image::Handle::from_path("data/images/test/out_0001.png"))
        }
        Message::WriteData => preprocessor::write_my_data(),
        Message::ReadData => preprocessor::read_my_data(),
    };
}

fn theme(_state: &State) -> Theme {
    Theme::Moonfly
}

pub fn main() -> iced::Result {
    iced::application("Генерация |>_<|", update, view)
        .theme(theme)
        .subscription(keyboard_subscription)
        .run()
}
