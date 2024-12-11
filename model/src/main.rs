use burn::{
    backend::Wgpu,
    data::dataset::{self, Dataset},
};
use model::ModelConfig;

mod data;
mod model;

use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Record {
    pub key: String,
    pub time: String,
    // Добавьте другие поля в зависимости от вашего CSV
}

fn main() {
    // type MyBackend = Wgpu<f32, i32>;

    // let device = Default::default();
    // let model = ModelConfig::new().init::<MyBackend>(&device);

    // println!("{}", model);

    let file_path = "../keyboard parser/key_log.csv";
    // let records = read_csv(file_path).unwrap();

    let rdr = csv::ReaderBuilder::new();

    // Здесь вы можете использовать records для настройки вашего dataset в burn
    // for record in records {
    //     println!("{:?}", record);
    // }

    let dataset = dataset::InMemDataset::<Record>::from_csv(file_path, &rdr);

    for i in dataset.unwrap().iter() {
        println!("{:?}", i);
    }
}
