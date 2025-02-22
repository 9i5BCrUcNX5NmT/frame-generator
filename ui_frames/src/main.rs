use iced::keyboard::{on_key_press, Key, Modifiers};
use iced::widget::{button, column, container, mouse_area, row, text};
use iced::Length::{self, Fill};
use iced::{Element, Point, Subscription, Theme};
use utils::key_to_string;

mod utils;

#[derive(Debug, Clone)]
enum Message {
    Key(String),
    Mouse(Point<f32>),
}

#[derive(Default)]
struct State {
    pub pressed_key: String,
    pub mouse_position: String,
}

fn view(state: &State) -> Element<Message> {
    let content = column![
        button(text(state.pressed_key.clone())),
        button(text(state.mouse_position.clone())),
    ]
    .spacing(20);

    let mouse_area = mouse_area(
        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill),
    )
    .on_move(|point| Message::Mouse(point));

    container(mouse_area)
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill)
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
    };
}

fn theme(state: &State) -> Theme {
    Theme::Moonfly
}

pub fn main() -> iced::Result {
    iced::application("A cool application", update, view)
        .theme(theme)
        .subscription(keyboard_subscription)
        .run()
}
