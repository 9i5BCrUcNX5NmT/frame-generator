use std::{path::PathBuf, str::FromStr};

use crate::{
    data::FrameBatcher,
    models::{baseline::model::BaselineConfig, unet::model::UNetConfig},
};
use burn::{
    backend::{self, Autodiff},
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    nn::loss::{MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::{activation::sigmoid, backend::AutodiffBackend},
    train::{LearnerBuilder, metric::LossMetric},
};

use preprocessor::{hdf5_processing::read_all_hdf5_files, types::MyConstData};

#[derive(Config)]
pub(crate) struct TrainingConfig {
    pub model: BaselineConfig,
    pub optimizer: AdamConfig,
    #[config(default = 15)]
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

    // // My
    // let mut model = config.model.init(&device);

    // let mut optimizer = config.optimizer.init();

    // for epoch in 1..config.num_epochs + 1 {
    //     for (iteration, batch) in dataloader_train.iter().enumerate() {
    //         // base diffusion train
    //         let random_timestamps = Tensor::<B, 1>::random(
    //             [batch.images.dims()[0]],
    //             burn::tensor::Distribution::Normal(0.0, 1.0),
    //             &device,
    //         );

    //         let alpha_timestamps = sigmoid(random_timestamps.clone())
    //             .sqrt()
    //             .reshape([-1, 1, 1, 1]);
    //         let sigma_timestamps = sigmoid(-random_timestamps.clone())
    //             .sqrt()
    //             .reshape([-1, 1, 1, 1]);

    //         let noise = batch
    //             .images
    //             .random_like(burn::tensor::Distribution::Default);
    //         let noised_images = batch.images * alpha_timestamps + noise.clone() * sigma_timestamps;
    //         // let noised_images = diffuse(batch.images.clone(), alpha_timestamps, sigma_timestamps);

    //         let predict = model.forward(
    //             noised_images.clone(),
    //             batch.keys.clone(),
    //             batch.mouse.clone(),
    //             random_timestamps.clone(),
    //         );

    //         // let snr = random_timestamps.exp().clamp_max(5);
    //         // let weight = snr.recip().reshape([-1, 1, 1, 1]);

    //         let loss = MseLoss::new().forward(predict, noise, Reduction::Auto); // TODO: Mean or Sum?

    //         println!(
    //             "[Train - Epoch {} - Iteration {}] Loss {:.3}",
    //             epoch,
    //             iteration,
    //             loss.clone().into_scalar(),
    //         );

    //         let grads = loss.backward();
    //         let grads = GradientsParams::from_grads(grads, &model);
    //         model = optimizer.step(config.learning_rate, model, grads);
    //     }

    //     println!("\nEpoch: {}\n", epoch);
    // }

    // model
    //     .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
    //     .expect("Trained model should be saved successfully");
    // println!("Model is save");

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

    // type MyBackend = backend::NdArray<f32>;
    // let device = backend::ndarray::NdArrayDevice::default();
    // type MyBackend = backend::Wgpu<f32, i32>;
    // let device = backend::wgpu::WgpuDevice::default();
    type MyBackend = backend::CudaJit<f32, i32>;
    let device = backend::cuda_jit::CudaDevice::default();

    type MyAutodiffBackend = Autodiff<MyBackend>;

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(BaselineConfig::new(), AdamConfig::new()),
        device.clone(),
    );
}
