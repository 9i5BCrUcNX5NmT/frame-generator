use std::io;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::{
    DefaultTerminal, Frame,
    buffer::Buffer,
    layout::Rect,
    style::Stylize,
    symbols::border,
    text::{Line, Text},
    widgets::{Block, Paragraph, Widget},
};
use ratatui_image::{StatefulImage, picker::Picker, protocol::StatefulProtocol};

mod utils;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let mut terminal = ratatui::init();

    let _app_result = App::default().run(&mut terminal)?;

    ratatui::restore();

    Ok(())
}

#[derive(Debug, Default)]
pub struct App {
    exit: bool,
}

impl App {
    /// runs the application's main loop until the user quits
    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        while !self.exit {
            terminal.draw(|frame| self.draw(frame))?;
            self.handle_events()?;
        }
        Ok(())
    }

    fn draw(&self, frame: &mut Frame) {
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
            // KeyCode::Left => self.decrement_counter(),
            // KeyCode::Right => self.increment_counter(),
            _ => {}
        }
    }

    fn exit(&mut self) {
        self.exit = true;
    }

    fn inference(&mut self) {
        todo!()
    }

    // fn increment_counter(&mut self) {
    //     self.counter += 1;
    // }

    // fn decrement_counter(&mut self) {
    //     self.counter -= 1;
    // }
}

impl Widget for &App {
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

        // let test_image: image::DynamicImage = image::open("screenshots/image-1.png").unwrap();

        // let picker = Picker::from_query_stdio().unwrap();

        // picker.new_resize_protocol(test_image).render(area, buf);

        // Paragraph::new(self.ascii_image.clone())
        //     .centered()
        //     .block(block)
        //     .render(area, buf);

        // let counter_text = Text::from(vec![Line::from(vec![
        //     "Value: ".into(),
        //     self.counter.to_string().yellow(),
        // ])]);

        // Paragraph::new(counter_text)
        //     .centered()
        //     .block(block)
        //     .render(area, buf);

        block.render(area, buf);
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use ratatui::style::Style;

//     #[test]
//     fn render() {
//         let app = App::default();
//         let mut buf = Buffer::empty(Rect::new(0, 0, 50, 4));

//         app.render(buf.area, &mut buf);

//         let mut expected = Buffer::with_lines(vec![
//             "┏━━━━━━━━━━━━━━ Diffusion Learner ━━━━━━━━━━━━━━━┓",
//             "┃                                                ┃",
//             "┃                                                ┃",
//             "┗━━━━━━━ Train <T> Inference <I> Quit <Q> ━━━━━━━┛",
//         ]);
//         let title_style = Style::new().bold();
//         let counter_style = Style::new().yellow();
//         let key_style = Style::new().blue().bold();
//         expected.set_style(Rect::new(15, 0, 17, 1), title_style);
//         // expected.set_style(Rect::new(28, 1, 1, 1), counter_style);
//         expected.set_style(Rect::new(13, 3, 6, 1), key_style);
//         expected.set_style(Rect::new(30, 3, 7, 1), key_style);
//         expected.set_style(Rect::new(43, 3, 4, 1), key_style);

//         assert_eq!(buf, expected);
//     }

//     #[test]
//     fn handle_key_event() -> io::Result<()> {
//         let mut app = App::default();
//         app.handle_key_event(KeyCode::Right.into());
//         assert_eq!(app.counter, 1);

//         app.handle_key_event(KeyCode::Left.into());
//         assert_eq!(app.counter, 0);

//         let mut app = App::default();
//         app.handle_key_event(KeyCode::Char('q').into());
//         assert!(app.exit);

//         Ok(())
//     }
// }
