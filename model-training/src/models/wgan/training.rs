use burn::{
    nn::loss::{MseLoss, Reduction},
    prelude::Backend,
    tensor::{Tensor, backend::AutodiffBackend},
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data::FrameBatch;

use super::model::WganDecoder;

impl<B: Backend> WganDecoder<B> {
    pub fn forward_generation(
        &self,
        inputs: Tensor<B, 4>,
        keys: Tensor<B, 2>,
        mouse: Tensor<B, 3>,
        targets: Tensor<B, 4>,
    ) -> RegressionOutput<B> {
        const P_STD: f32 = 1.2;
        const P_MEAN: f32 = -1.2;

        let random_normal = Tensor::random(
            [inputs.dims()[0], 1, 1, 1],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &inputs.device(),
        );
        let sigma = (random_normal * P_STD + P_MEAN).exp();
        let noise = inputs.random_like(burn::tensor::Distribution::Normal(0.0, 1.0)) * sigma;

        let noised_images = inputs + noise;

        let output = self.forward(noised_images, keys, mouse);

        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Auto);

        let output_2d = output.flatten(1, 3);
        let targets_2d = targets.flatten(1, 3);

        RegressionOutput {
            loss,
            output: output_2d,
            targets: targets_2d,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<FrameBatch<B>, RegressionOutput<B>> for WganDecoder<B> {
    fn step(&self, batch: FrameBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_generation(batch.images, batch.keys, batch.mouse, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<FrameBatch<B>, RegressionOutput<B>> for WganDecoder<B> {
    fn step(&self, batch: FrameBatch<B>) -> RegressionOutput<B> {
        self.forward_generation(batch.images, batch.keys, batch.mouse, batch.targets)
    }
}
