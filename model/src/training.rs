use crate::model::{Model, ModelConfig};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};
use nn::loss::HuberLossConfig;

impl<B: Backend> Model<B> {
    pub fn forward_generation(&self, images: Tensor<B, 4>, targets: Tensor<B, 4>) {
        let output = self.forward(images);
        // Какую дельту ставить? Я хз
        let loss = HuberLossConfig::new(0.5)
            .init()
            // Так же неизвестно какую Reduction ставить
            .forward(output.clone(), targets.clone(), nn::loss::Reduction::Auto);
        // let loss = CrossEntropyLossConfig::new()
        //     .init(&output.device())
        //     .forward(output.clone(), targets.clone());

        // ClassificationOutput::new(loss, output, targets)

        todo!("Возврат Output")
    }
}
