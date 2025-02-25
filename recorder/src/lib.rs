use std::{
    fs::OpenOptions,
    thread::{self, sleep},
    time::Duration,
};

use csv::Writer;
use keys_recorder::KeysRecorder;
use rdev::listen;
use video_recorder::VideoRecorder;

mod keys_recorder;
mod video_recorder;

pub(crate) const DATA_DIR: &str = "data/";

pub fn run() {
    std::fs::create_dir_all(DATA_DIR.to_owned() + "keys").unwrap();
    std::fs::create_dir_all(DATA_DIR.to_owned() + "videos").unwrap();

    let file_path = DATA_DIR.to_owned() + "keys/key_events.csv";

    let mut writer = Writer::from_writer(
        OpenOptions::new()
            .append(true)
            .create(true)
            .open(file_path)
            .unwrap(),
    );

    let keys_recorder = KeysRecorder::new();
    let keys_recorder_clone = keys_recorder.clone();

    let vider_recorder = VideoRecorder::start_recording();
    let vider_recorder_clone = vider_recorder.clone();

    thread::spawn(|| {
        listen(move |event| {
            keys_recorder_clone.lock().unwrap().insert_key(&event);
            vider_recorder_clone.lock().unwrap().control_capture(&event);
        })
        .unwrap();
    });

    while !vider_recorder.lock().unwrap().is_finished() {
        sleep(Duration::from_micros(50));

        keys_recorder.lock().unwrap().write_keys(&mut writer);
    }
}
