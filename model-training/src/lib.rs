mod blocks;
mod csv_processing;
mod data;
mod images;
pub mod inference;
mod model;
pub mod training;
mod types;

pub const WIDTH: usize = 192;
pub const HEIGHT: usize = 108;
pub const MOUSE_VECTOR_LENGTH: usize = 20;
