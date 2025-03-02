use crate::{csv_processing::KeysRecord, images::MyImage};
use common::*;

#[derive(Clone, Debug)]
pub struct MyData {
    pub image: MyImage<WIDTH, HEIGHT>,
    pub keys: KeysRecord,
}
