use iced::keyboard::{on_key_press, Key, Modifiers};
use iced::widget::{button, column, container, image, mouse_area, text};
use iced::Length;
use iced::{Element, Point, Subscription, Theme};
use utils::{generate_frame, key_to_string};

mod utils;

#[derive(Debug, Clone)]
enum Message {
    Key(String),
    Mouse(Point<f32>),
    ReloadImage,
}

#[derive(Default)]
struct State {
    pub pressed_key: String,
    pub mouse_position: String,
    pub image_path: String,
}

fn view(state: &State) -> Element<Message> {
    let content = column![
        button(text(state.pressed_key.clone())),
        button(text(state.mouse_position.clone())),
        image(&state.image_path),
        button(text("Генерация")).on_press(Message::ReloadImage)
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
        Message::Mouse(point) => state.mouse_position = point.to_string(),
        Message::ReloadImage => {
            generate_frame("tmp/test/output");
            state.image_path = "tmp/test/output/image.png".to_string();
        }
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
