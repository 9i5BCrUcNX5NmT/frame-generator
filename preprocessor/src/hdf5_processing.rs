use std::{fs, io, path::PathBuf};

use hdf5_metno::{File, Result, Selection};

use crate::types::MyConstData;

pub fn write_data(
    data_path: &PathBuf,
    // my_data: &[MyConstData; 3]
) -> Result<()> {
    let file_path = data_path.join("my_data.h5");

    let file = File::create(file_path)?; // open for writing
    let group = file.create_group("dir")?; // create a group
                                           // #[cfg(feature = "blosc")]
                                           // blosc_set_nthreads(2); // set number of blosc threads
                                           // let builder = group.new_dataset_builder();
                                           // #[cfg(feature = "blosc")]
                                           // let builder = builder.blosc_zstd(9, true); // zstd + shuffle

    // let ds = builder
    //     .with_data(my_data)
    //     // finalize and write the dataset
    //     .create("data")?;

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

            let a = ds.read_raw::<MyConstData>().unwrap();

            println!("{:?}", a);

            dataset.push(a[0].clone());
        }
    }

    Ok(dataset)
}
