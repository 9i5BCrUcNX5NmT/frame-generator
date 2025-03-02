use std::{fs, io, path::PathBuf};

use hdf5_metno::{File, Result};
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};

use crate::types::MyConstData;

fn write_hdf5_file(
    file_path: &PathBuf,
    my_data: &ArrayBase<OwnedRepr<MyConstData>, Dim<[usize; 1]>>,
) -> Result<()> {
    let file = File::create(file_path)?; // open for writing
    let group = file.create_group("dir")?; // create a group
                                           // #[cfg(feature = "blosc")]
                                           // blosc_set_nthreads(2); // set number of blosc threads
    let builder = group.new_dataset_builder();
    // #[cfg(feature = "blosc")]
    // let builder = builder.blosc_zstd(9, true); // zstd + shuffle

    let _ = builder
        .with_data(my_data)
        // finalize and write the dataset
        .create("data")?;

    Ok(())
}

pub fn write_data_to_hdf5_files(data_path: &PathBuf, my_data: &Vec<MyConstData>) {
    std::fs::create_dir_all(data_path).unwrap();

    for (i, data) in my_data.chunks(1).enumerate() {
        let array_data = Array::from_vec(data.to_vec());

        let file_path = data_path.join(format!("my_data_{}.h5", i));

        write_hdf5_file(&file_path, &array_data).unwrap();
    }
}

pub fn read_all_hdf5_files(data_path: &PathBuf) -> io::Result<Vec<MyConstData>> {
    let mut dataset = Vec::new();

    for entry in fs::read_dir(data_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            let file = File::open(path)?; // open for reading
            let ds = file.dataset("dir/data")?; // open the dataset

            let my_data = ds.read_raw::<MyConstData>().unwrap();

            dataset.extend(my_data);
        }
    }

    Ok(dataset)
}
