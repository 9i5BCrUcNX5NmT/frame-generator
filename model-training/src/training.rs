use std::{path::PathBuf, str::FromStr};

use crate::{data::FrameBatcher, models::model_v1::model::ModelV1Config};

use burn::{
    backend::{self, Autodiff},
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, metric::LossMetric},
};

use preprocessor::{hdf5_processing::read_all_hdf5_files, types::MyConstData};

#[derive(Config)]
pub(crate) struct TrainingConfig {
    pub model: ModelV1Config,
    pub optimizer: AdamConfig,
    #[config(default = 25)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 16)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let data_path = PathBuf::from_str("data").unwrap();
    let data_path = &data_path.join("hdf5_files");

    let my_data = read_all_hdf5_files(data_path).expect("Чтение всех файлов hdf5");

    let train_percintil = 0.8;
    let train_len = (my_data.len() as f64 * train_percintil) as usize;

    let train_data = my_data[..train_len].to_vec();
    let test_data = my_data[train_len..].to_vec();

    let dataset_train: InMemDataset<MyConstData> = InMemDataset::new(train_data);
    let dataset_test: InMemDataset<MyConstData> = InMemDataset::new(test_data);

    let batcher_train = FrameBatcher::<B>::new(device.clone());
    let batcher_valid = FrameBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_test);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

// /// Зашумление
// pub fn diffuse<B: AutodiffBackend>(
//     input: Tensor<B, 4>,
//     // The alpha value at timepoint t
//     alpha_t: Tensor<B, 4>,
//     // The sigma value at timepoint t
//     sigma_t: Tensor<B, 4>,
// ) -> Tensor<B, 4> {
//     let noise = input.random_like(burn::tensor::Distribution::Default);
//     let diffused_input = input * alpha_t + noise * sigma_t;

//     diffused_input
// }

pub fn run() {
    let artifact_dir = "tmp/test";

    #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
    type MyBackend = backend::NdArray<f32>;
    #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
    let device = backend::ndarray::NdArrayDevice::default();

    #[cfg(feature = "wgpu")]
    type MyBackend = backend::Wgpu<f32, i32>;
    #[cfg(feature = "wgpu")]
    let device = backend::wgpu::WgpuDevice::default();

    #[cfg(feature = "cuda")]
    type MyBackend = backend::Cuda<f32, i32>;
    #[cfg(feature = "cuda")]
    let device = backend::cuda::CudaDevice::default();

    type MyAutodiffBackend = Autodiff<MyBackend>;

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelV1Config::new(), AdamConfig::new()),
        device.clone(),
    );
}
