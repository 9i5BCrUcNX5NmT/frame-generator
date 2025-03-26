use std::{path::PathBuf, str::FromStr};

use crate::{
    data::FrameBatcher,
    models::{
        baseline_conv::model::BaselineConfig, baseline_diffusion::model::DiffusionConfig,
        unet::model::UNetConfig,
    },
};
use burn::{
    backend::{self, Autodiff},
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    nn::loss::{MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::{Distribution, backend::AutodiffBackend},
    train::{LearnerBuilder, metric::LossMetric},
};

use common::{CHANNELS, HEIGHT, WIDTH};
use preprocessor::{hdf5_processing::read_all_hdf5_files, types::MyConstData};

#[derive(Config)]
pub(crate) struct TrainingConfig {
    pub model: UNetConfig,
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

    // My
    let mut model = config.model.init(&device);

    let mut optimizer = config.optimizer.init();

    // Iterate over our training for X epochs
    for epoch in 1..config.num_epochs + 1 {
        // Implement our training loop
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            // // Generate a batch of fake images from noise (standarded normal distribution)
            // let noise = Tensor::<B, 4>::random(
            //     [config.batch_size, config.model.latent_dim],
            //     Distribution::Normal(0.0, 1.0),
            //     &device,
            // );
            // datach: do not update gerenator, only discriminator is updated

            let mut loss: Tensor<B, 1, Float> = Tensor::zeros([1], &device);

            // base diffusion train
            // TODO: upgrade to one step train
            for timestamp in 0..config.model.num_timestamps {
                let noise_level = timestamp as f32 / config.model.num_timestamps as f32;
                let noised_images: Tensor<B, 4> = add_noise(batch.images.clone(), noise_level); // TODO: Пока не знаю, как лучше
                let model_predict = model.forward(
                    noised_images.clone(),
                    batch.keys.clone(),
                    batch.mouse.clone(),
                );

                loss = loss.add(MseLoss::new().forward(
                    model_predict,
                    batch.targets.clone(),
                    Reduction::Mean,
                )); // TODO: Mean or Sum?
            }

            loss = loss.div_scalar(config.model.num_timestamps as f32);

            // println!("Loss: {}", loss.to_data().to_vec::<f32>().unwrap()[0]);
            println!(
                "[Train - Epoch {} - Iteration {}] Loss {:.3}",
                epoch,
                iteration,
                loss.clone().into_scalar(),
            );

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(config.learning_rate, model, grads);
        }

        println!("\nEpoch: {}\n", epoch);
    }

    // let learner = LearnerBuilder::new(artifact_dir)
    //     .metric_train_numeric(LossMetric::new())
    //     .metric_valid_numeric(LossMetric::new())
    //     .with_file_checkpointer(CompactRecorder::new())
    //     .devices(vec![device.clone()])
    //     .num_epochs(config.num_epochs)
    //     .summary()
    //     .build(
    //         config.model.init::<B>(&device),
    //         config.optimizer.init(),
    //         config.learning_rate,
    //     );

    // let model_trained = learner.fit(dataloader_train, dataloader_test);

    // model_trained
    //     .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
    //     .expect("Trained model should be saved successfully");

    model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
    println!("Model is save");
}

fn add_noise<B: AutodiffBackend>(input: Tensor<B, 4>, noise_level: f32) -> Tensor<B, 4> {
    // Добавление шума к входным данным
    let noise = input.random_like(burn::tensor::Distribution::Normal(0.0, 1.0));
    input * (1.0 - noise_level) + noise * (noise_level)
}

pub fn run() {
    let artifact_dir = "tmp/test";

    // type MyBackend = backend::NdArray<f32>;
    // let device = backend::ndarray::NdArrayDevice::default();
    type MyBackend = backend::Wgpu<f32, i32>;
    let device = backend::wgpu::WgpuDevice::default();
    // type MyBackend = backend::CudaJit<f32, i32>;
    // let device = backend::cuda_jit::CudaDevice::default();

    type MyAutodiffBackend = Autodiff<MyBackend>;

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(UNetConfig::new(), AdamConfig::new()),
        device.clone(),
    );
}
