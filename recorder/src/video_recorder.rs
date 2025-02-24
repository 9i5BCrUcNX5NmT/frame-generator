use std::io::{self, Write};
use std::thread::{self, sleep};
use std::time::{Duration, Instant};

use rdev::{listen, EventType};
use windows_capture::capture::{Context, GraphicsCaptureApiHandler};
use windows_capture::encoder::{
    AudioSettingsBuilder, ContainerSettingsBuilder, VideoEncoder, VideoSettingsBuilder,
};
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::InternalCaptureControl;
use windows_capture::monitor::Monitor;
use windows_capture::settings::{ColorFormat, CursorCaptureSettings, DrawBorderSettings, Settings};

// Handles capture events.
struct Capture {
    // The video encoder that will be used to encode the frames.
    encoder: Option<VideoEncoder>,
    // To measure the time the capture has been running
    start: Instant,
    should_stop: bool, // Состояние записи
}

impl GraphicsCaptureApiHandler for Capture {
    // The type of flags used to get the values from the settings.
    type Flags = String;

    // The type of error that can be returned from `CaptureControl` and `start` functions.
    type Error = Box<dyn std::error::Error + Send + Sync>;

    // Function that will be called to create a new instance. The flags can be passed from settings.
    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        println!("Created with Flags: {}", ctx.flags);

        let encoder = VideoEncoder::new(
            VideoSettingsBuilder::new(2560, 1440),
            AudioSettingsBuilder::default().disabled(true),
            ContainerSettingsBuilder::default(),
            "data/videos/video.mp4",
        )?;

        Ok(Self {
            encoder: Some(encoder),
            start: Instant::now(),
            should_stop: false, // Инициализация состояния записи
        })
    }

    // Called every time a new frame is available.
    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        capture_control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        print!(
            "\rRecording for: {} seconds",
            self.start.elapsed().as_secs()
        );
        io::stdout().flush()?;

        // Send the frame to the video encoder
        self.encoder.as_mut().unwrap().send_frame(frame)?;

        // Note: The frame has other uses too, for example, you can save a single frame to a file, like this:
        // frame.save_as_image("frame.png", ImageFormat::Png)?;
        // Or get the raw data like this so you have full control:
        // let data = frame.buffer()?;

        // Stop the capture after 6 seconds
        // if self.start.elapsed().as_secs() >= 6 {
        //     // Finish the encoder and save the video.
        //     self.encoder.take().unwrap().finish()?;

        //     capture_control.stop();

        //     // Because there wasn't any new lines in previous prints
        //     println!();
        // }

        // Проверяем состояние записи
        if self.should_stop {
            // Завершите кодировщик и сохраните видео.
            self.encoder.take().unwrap().finish()?;
            capture_control.stop();
            println!();
        }

        Ok(())
    }

    // Optional handler called when the capture item (usually a window) closes.
    fn on_closed(&mut self) -> Result<(), Self::Error> {
        println!("Capture session ended");

        Ok(())
    }
}

pub fn record() {
    // Gets the foreground window, refer to the docs for other capture items
    let primary_monitor = Monitor::primary().expect("There is no primary monitor");

    let settings = Settings::new(
        // Item to capture
        primary_monitor,
        // Capture cursor settings
        CursorCaptureSettings::Default,
        // Draw border settings
        DrawBorderSettings::Default,
        // The desired color format for the captured frame.
        ColorFormat::Bgra8,
        // Additional flags for the capture settings that will be passed to user defined `new` function.
        "Yea this works".to_string(),
    );

    // Starts the capture and takes control of the current thread.
    // The errors from handler trait will end up here
    let capture = Capture::start_free_threaded(settings).expect("Screen capture failed");
    let capture_callback = capture.callback();

    thread::spawn(move || {
        listen(move |event| match event.event_type {
            EventType::KeyPress(key) => match key {
                rdev::Key::KeyQ => {
                    capture_callback.lock().should_stop = true;
                }
                _ => {}
            },
            _ => {}
        })
        .unwrap();
    });

    while !capture.is_finished() {
        sleep(Duration::from_micros(1));
    }
}
