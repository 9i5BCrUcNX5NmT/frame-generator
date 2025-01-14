use crate::{
    data::{FrameBatch, FrameBatcher},
    images::{self, convert_images_to_image_pixel_data, load_images_from_directory},
    model::{Model, ModelConfig},
};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};
use nn::loss::HuberLossConfig;

use burn::{
    prelude::Backend,
    tensor::Tensor,
    train::metric::{AccuracyInput, Adaptor, LossInput},
};

impl<B: Backend> Model<B> {
    pub fn forward_generation(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 4>,
    ) -> RegressionOutput<B> {
        let output = self.forward(images);
        // Какую дельту ставить? Я хз
        let loss = HuberLossConfig::new(0.5)
            .init()
            // Так же неизвестно какую Reduction ставить
            .forward(output.clone(), targets.clone(), nn::loss::Reduction::Auto);

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

impl<B: AutodiffBackend> TrainStep<FrameBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: FrameBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_generation(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<FrameBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: FrameBatch<B>) -> RegressionOutput<B> {
        self.forward_generation(batch.images, batch.targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
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

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let train_dir = "../data/images/train";
    let test_dir = "../data/images/test";

    let train_images = load_images_from_directory(train_dir).unwrap();
    let test_images = load_images_from_directory(test_dir).unwrap();

    let dataset_train: InMemDataset<images::ImagePixelData> =
        InMemDataset::new(convert_images_to_image_pixel_data(train_images));
    let dataset_test: InMemDataset<images::ImagePixelData> =
        InMemDataset::new(convert_images_to_image_pixel_data(test_images));

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
