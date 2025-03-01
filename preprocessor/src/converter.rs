use std::path::PathBuf;

use burn::tensor::TensorData;
use hdf5_metno::{File, Result};

use crate::types::MyConstData;

pub fn write_data(data_path: &PathBuf, my_data: MyConstData) -> Result<()> {
    let file_path = data_path.join("my_daya.h5");

    let file = File::create(file_path)?; // open for writing
    let group = file.create_group("dir")?; // create a group
    #[cfg(feature = "blosc")]
    blosc_set_nthreads(2); // set number of blosc threads
    let builder = group.new_dataset_builder();
    #[cfg(feature = "blosc")]
    let builder = builder.blosc_zstd(9, true); // zstd + shuffle

    // let my_pixels = TensorData::from(my_data.image.pixels);
    // let my_keys = TensorData::from(my_data.keys_record.keys);
    // let my_mouse = TensorData::from(my_data.keys_record.mouse);

    let ds = builder
        .with_data(&[my_data])
        // finalize and write the dataset
        .create("data")?;
    // // create an attr with fixed shape but don't write the data
    // let attr = ds.new_attr::<Color>().shape([3]).create("colors")?;
    // // write the attr data
    // attr.write(&[R, G, B])?;
    Ok(())
}
