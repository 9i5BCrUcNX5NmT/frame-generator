use crate::{csv_processing::KeysRecord, images::ImagePixelData};

#[derive(Clone, Debug)]
pub struct MyData {
    pub image: ImagePixelData,
    pub keys: KeysRecord,
}
