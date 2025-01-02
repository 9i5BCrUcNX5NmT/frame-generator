use crate::{data::FrameBatch, model::Model};
use burn::{
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
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
        images: Tensor<B, 3>,
        targets: Tensor<B, 3>,
    ) -> RegressionOutput<B> {
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
        RegressionOutput {
            loss,
            output,
            targets,
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
