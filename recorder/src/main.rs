use std::{
    thread::{self, sleep},
    time::Duration,
};

mod keys_recorder;
mod video_recorder;

fn main() {
    std::fs::create_dir_all("../data/keys").unwrap();
    std::fs::create_dir_all("../data/videos").unwrap();

    let videos_handle = thread::spawn(|| video_recorder::record());
    let _keys_handle = thread::spawn(|| keys_recorder::record().unwrap());

    while !videos_handle.is_finished() {
        sleep(Duration::from_millis(1));
    }
}
