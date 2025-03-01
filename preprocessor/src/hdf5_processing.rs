use std::{fs, io, path::PathBuf};

use burn::tensor::TensorData;
use hdf5_metno::{File, Result};

use crate::types::MyConstData;

pub fn write_data(data_path: &PathBuf, my_data: MyConstData) -> Result<()> {
    let file_path = data_path.join("my_daya.h5");

    if file_path.exists() {
        let file = File::open_rw(file_path)?;

        let group = file.group("dir")?;
        let ds = group.dataset("data")?;
        ds.write(&[my_data])?;
    } else {
        let file = File::create(file_path)?; // open for writing
        let group = file.create_group("dir")?; // create a group
        #[cfg(feature = "blosc")]
        blosc_set_nthreads(2); // set number of blosc threads
        let builder = group.new_dataset_builder();
        #[cfg(feature = "blosc")]
        let builder = builder.blosc_zstd(9, true); // zstd + shuffle

        let ds = builder
            .with_data(&[my_data])
            // finalize and write the dataset
            .create("data")?;
    }

    Ok(())
}

pub fn read_data(data_path: &PathBuf) -> io::Result<Vec<MyConstData>> {
    let mut dataset = Vec::new();

    for entry in fs::read_dir(data_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            let file = File::open(path)?; // open for reading
            let ds = file.dataset("dir/data")?; // open the dataset

            let a = ds.read_1d::<MyConstData>().unwrap();

            dataset.push(a[0].clone());
        }
    }

    Ok(dataset)
}
