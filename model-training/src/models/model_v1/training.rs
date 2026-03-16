use burn::{
    nn::loss::{MseLoss, Reduction},
    prelude::Backend,
    tensor::{Tensor, backend::AutodiffBackend},
    train::{InferenceStep, RegressionOutput, TrainOutput, TrainStep},
};

use crate::data::FrameBatch;

use super::model::ModelV1;

impl<B: Backend> ModelV1<B> {
    pub fn forward_generation(
        &self,
        inputs: Tensor<B, 4>,
        keys: Tensor<B, 2>,
        mouse: Tensor<B, 3>,
        targets: Tensor<B, 4>,
    ) -> RegressionOutput<B> {
        const P_STD: f32 = 1.2;
        const P_MEAN: f32 = -1.2;

        let batch_size = inputs.dims()[0];
        let device = inputs.device();

        // Sample random timestep for each sample in batch
        let random_timestep = Tensor::random(
            [batch_size],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );

        let random_normal = Tensor::random(
            [batch_size, 1, 1, 1],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let sigma = (random_normal * P_STD + P_MEAN).exp();
        let noise = inputs.random_like(burn::tensor::Distribution::Normal(0.0, 1.0)) * sigma;

        let noised_targets = targets.clone() + noise;

        let output = self.forward(inputs, keys, mouse, noised_targets, random_timestep);

        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Auto);

        let output_2d = output.flatten(1, 3);
        let targets_2d = targets.flatten(1, 3);

        RegressionOutput::new(loss, output_2d, targets_2d)
    }
}

impl<B: AutodiffBackend> TrainStep for ModelV1<B> {
    type Input = FrameBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: FrameBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_generation(batch.images, batch.keys, batch.mouse, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for ModelV1<B> {
    type Input = FrameBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: FrameBatch<B>) -> RegressionOutput<B> {
        self.forward_generation(batch.images, batch.keys, batch.mouse, batch.targets)
    }
}
