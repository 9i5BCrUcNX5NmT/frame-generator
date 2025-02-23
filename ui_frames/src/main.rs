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
        },
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
        Message::Mouse(point) => state.mouse_position = point,
        Message::ReloadImage => {
            let keys = vec![state.pressed_key.clone()];
            let mouse = vec![state.mouse_position];

            state.image = Some(match &state.image {
                Some(image_handle) => generate_frame(image_handle, keys, mouse),
                None => generate_frame(
                    &image::Handle::from_path("tmp/test/output/image.png"),
                    keys,
                    mouse,
                ),
            })
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
