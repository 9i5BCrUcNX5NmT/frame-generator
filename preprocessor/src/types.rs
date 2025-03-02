use common::*;
use hdf5_metno::H5Type;

use crate::{csv_processing::KeysRecordConst, images::MyImage};

#[derive(Clone, Debug, H5Type)]
#[repr(C)]
pub struct MyConstData {
    pub image: MyImage<WIDTH, HEIGHT>,
    pub keys_record: KeysRecordConst,
}
