use std::io;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use image::DynamicImage;
use ratatui::{
    DefaultTerminal, Frame,
    buffer::Buffer,
    layout::Rect,
    style::Stylize,
    symbols::border,
    text::Line,
    widgets::{Block, Borders, Widget},
};
use ratatui_image::{StatefulImage, picker::Picker, protocol::StatefulProtocol};
use utils::get_first_file_in_directory;

mod utils;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let mut terminal = ratatui::init();

    let image_path = match get_first_file_in_directory("data/images/resized_images").unwrap() {
        Some(image_path) => image_path,
        None => panic!("Отсутствуют файлы в resized_images"),
    };

    let first_image = image::open(image_path).unwrap();
    let picker = Picker::from_query_stdio()
        .expect("Поддержка протокола считывания размера шрифта из терминала(среды запуска)");
    let image = picker.new_resize_protocol(first_image.clone());

    let _app_result = App {
        exit: false,
        image,
        picker,
        dynamic_image: first_image,
    }
    .run(&mut terminal)?;

    ratatui::restore();

    Ok(())
}

pub struct App {
    exit: bool,
    image: StatefulProtocol,
    picker: Picker,
    dynamic_image: DynamicImage,
}

impl App {
    /// runs the application's main loop until the user quits
    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        while !self.exit {
            terminal.draw(|frame| {
                self.draw(frame);

                let inner_area = Block::default()
                    .borders(Borders::ALL)
                    .title("image")
                    .inner(frame.area());
                frame.render_stateful_widget(StatefulImage::default(), inner_area, &mut self.image);
            })?;
            self.handle_events()?;
        }
        Ok(())
    }

    fn draw(&mut self, frame: &mut Frame) {
        frame.render_widget(self, frame.area());
    }

    /// updates the application's state based on user input
    fn handle_events(&mut self) -> io::Result<()> {
        match event::read()? {
            // it's important to check that the event is a key press event as
            // crossterm also emits key release and repeat events on Windows.
            Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
                self.handle_key_event(key_event)
            }
            _ => {}
        };
        Ok(())
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) {
        match key_event.code {
            KeyCode::Char('q') => self.exit(),
            KeyCode::Char('i') => self.inference(),
            KeyCode::Char('t') => self.train(),
            _ => {}
        }
    }

    fn exit(&mut self) {
        self.exit = true;
    }

    fn inference(&mut self) {
        let generated_image =
            model_training::inference::generate(&self.dynamic_image.clone(), vec![], vec![]); // TODO: считывание клавиш и мыши
        self.image = self.picker.new_resize_protocol(generated_image);
    }

    fn train(&mut self) {
        model_training::training::run();
    }
}

impl Widget for &mut App {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let title = Line::from(" Diffusion Learner ".bold());
        let instructions = Line::from(vec![
            " Train ".into(),
            "<T>".blue().bold(),
            " Inference ".into(),
            "<I>".blue().bold(),
            " Quit ".into(),
            "<Q> ".blue().bold(),
        ]);
        let block = Block::bordered()
            .title(title.centered())
            .title_bottom(instructions.centered())
            .border_set(border::THICK);

        block.render(area, buf);
    }
}
