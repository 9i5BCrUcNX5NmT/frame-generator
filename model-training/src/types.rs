use crate::{csv_processing::KeysRecord, images::MyImage, HEIGHT, WIDTH};

#[derive(Clone, Debug)]
pub struct MyData {
    pub image: MyImage<WIDTH, HEIGHT>,
    pub keys: KeysRecord,
}
