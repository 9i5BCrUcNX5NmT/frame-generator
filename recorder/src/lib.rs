use std::{
    fs::OpenOptions,
    path::PathBuf,
    str::FromStr,
    sync::{Arc, Mutex},
    thread::{self, sleep},
    time::Duration,
};

use common::DATA_DIR;
use csv::Writer;
use fs_extra::dir;
use keys_recorder::KeysRecorder;
use rdev::listen;
use video_recorder::VideoRecorder;

mod keys_recorder;
mod video_recorder;

pub fn run() {
    // TODO: не лучшее решение
    dir::create(DATA_DIR, true).unwrap();

    let images_path = PathBuf::from_str(DATA_DIR).unwrap().join("images");
    let keys_path = PathBuf::from_str(DATA_DIR).unwrap().join("keys");

    dir::create_all(&images_path, true).unwrap();
    dir::create_all(&keys_path, true).unwrap();

    let mut writer = Writer::from_writer(
        OpenOptions::new()
            .append(true)
            .create(true)
            .open(keys_path.join("key_events.csv"))
            .unwrap(),
    );

    let keys_recorder = Arc::new(Mutex::new(KeysRecorder::new()));
    let keys_recorder_clone_1 = keys_recorder.clone();
    let keys_recorder_clone_2 = keys_recorder.clone();

    let vider_recorder = Arc::new(Mutex::new(VideoRecorder::new(images_path.join("raw"))));
    let vider_recorder_clone_1 = vider_recorder.clone();
    let vider_recorder_clone_2 = vider_recorder.clone();

    vider_recorder.lock().unwrap().start();

    println!("Record started");
    thread::spawn(|| {
        listen(move |event| {
            keys_recorder_clone_1.lock().unwrap().insert_key(&event);
            vider_recorder_clone_1.lock().unwrap().check_keys(&event);
        })
        .unwrap();
    });

    while !vider_recorder_clone_2.lock().unwrap().is_finished() {
        sleep(Duration::from_micros(50));

        keys_recorder_clone_2
            .lock()
            .unwrap()
            .write_keys(&mut writer);
    }
    println!("Record was end");
}
