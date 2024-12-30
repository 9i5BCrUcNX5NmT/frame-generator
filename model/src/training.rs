use crate::{
    data::FrameBatch,
    model::{Model, ModelConfig},
};
use burn::{
    data::dataloader::DataLoaderBuilder,
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

use burn::{
    prelude::Backend,
    tensor::Tensor,
    train::metric::{AccuracyInput, Adaptor, LossInput},
};

/// Simple classification output adapted for multiple metrics.
// #[derive(new)]
pub struct GeneraionOutput<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,

    /// The output.
    pub output: Tensor<B, 4>,

    /// The targets.
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> Adaptor<AccuracyInput<B>> for GeneraionOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for GeneraionOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> Model<B> {
    pub fn forward_generation(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 4>,
    ) -> GeneraionOutput<B> {
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
        GeneraionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<FrameBatch<B>, GeneraionOutput<B>> for Model<B> {
    fn step(&self, batch: FrameBatch<B>) -> TrainOutput<GeneraionOutput<B>> {
        let item = self.forward_generation(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<FrameBatch<B>, GeneraionOutput<B>> for Model<B> {
    fn step(&self, batch: FrameBatch<B>) -> GeneraionOutput<B> {
        self.forward_generation(batch.images, batch.targets)
    }
}
