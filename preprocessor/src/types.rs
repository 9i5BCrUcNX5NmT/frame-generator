use hdf5_metno::H5Type;
use model_training::{HEIGHT, WIDTH};

use crate::{csv_processing::KeysRecordConst, images::MyImage};

#[derive(Clone, Debug, H5Type)]
#[repr(C)]
pub struct MyConstData {
    pub image: MyImage<WIDTH, HEIGHT>,
    pub keys_record: KeysRecordConst,
}
