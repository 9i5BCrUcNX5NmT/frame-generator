use std::{path::PathBuf, str::FromStr};

use crate::{
    data::{FrameBatch, FrameBatcher},
    models::unet::model::{UNet, UNetConfig},
};
use burn::{
    backend::{self, Autodiff},
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    nn::loss::MseLoss,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep, metric::LossMetric,
    },
};

use burn::{prelude::Backend, tensor::Tensor};
use preprocessor::{hdf5_processing::read_all_hdf5_files, types::MyConstData};

impl<B: Backend> UNet<B> {
    pub fn forward_generation(
        &self,
        inputs: Tensor<B, 4>,
        keys: Tensor<B, 2>,
        mouse: Tensor<B, 3>,
        targets: Tensor<B, 4>,
    ) -> RegressionOutput<B> {
        let output = self.forward(inputs, keys, mouse);

        let loss =
            MseLoss::new().forward(output.clone(), targets.clone(), nn::loss::Reduction::Auto);

        let [batch_size, color, height, width] = output.dims();
        let output_2d = output.reshape([batch_size, color * height * width]);

        let [batch_size, color, height, width] = targets.dims();
        let targets_2d = targets.reshape([batch_size, color * height * width]);

        RegressionOutput {
            loss,
            output: output_2d,
            targets: targets_2d,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<FrameBatch<B>, RegressionOutput<B>> for UNet<B> {
    fn step(&self, batch: FrameBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_generation(batch.images, batch.keys, batch.mouse, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<FrameBatch<B>, RegressionOutput<B>> for UNet<B> {
    fn step(&self, batch: FrameBatch<B>) -> RegressionOutput<B> {
        self.forward_generation(batch.images, batch.keys, batch.mouse, batch.targets)
    }
}

#[derive(Config)]
pub(crate) struct TrainingConfig {
    pub model: UNetConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 8)]
    pub batch_size: usize,
    #[config(default = 4)]
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

    // let train_dir = "data/images/train";
    // let test_dir = "data/images/test";

    // let train_images =
    //     convert_images_to_image_pixel_data(load_images_from_directory(train_dir).unwrap());
    // let test_images =
    //     convert_images_to_image_pixel_data(load_images_from_directory(test_dir).unwrap());

    // let keys = load_keys_from_directory("data/keys").unwrap();

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

pub fn run() {
    let artifact_dir = "tmp/test";

    // type MyBackend = backend::NdArray<f32>;
    // let device = backend::ndarray::NdArrayDevice::default();
    // type MyBackend = backend::Wgpu<f32, i32>;
    // let device = backend::wgpu::WgpuDevice::default();
    type MyBackend = backend::CudaJit<f32, i32>;
    let device = backend::cuda_jit::CudaDevice::default();

    type MyAutodiffBackend = Autodiff<MyBackend>;

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(UNetConfig::new(), AdamConfig::new()),
        device.clone(),
    );
}
