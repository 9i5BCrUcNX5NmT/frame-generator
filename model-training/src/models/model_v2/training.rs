use burn::{
    nn::loss::{MseLoss, Reduction},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Distribution, Tensor},
    train::{InferenceStep, RegressionOutput, TrainOutput, TrainStep},
};

use crate::{data::FrameBatch, models::noise_schedule::CosineNoiseSchedule};

use super::model::ModelV2;

/// Number of diffusion timesteps for training
const NUM_TIMESTEPS: usize = 1000;

/// KL divergence weight for VAE regularization (beta-VAE)
const KL_WEIGHT: f32 = 0.001;

impl<B: Backend> ModelV2<B> {
    /// Compute diffusion training loss: MSE(predicted_noise, true_noise) + KL divergence.
    ///
    /// 1. Sample random timestep per batch element
    /// 2. Encode target to latent, add noise at timestep
    /// 3. Predict noise with conditioned U-Net
    /// 4. Loss = MSE(predicted, true_noise) + beta * KL(q(z|x) || p(z))
    pub fn forward_diffusion(
        &self,
        _images: Tensor<B, 4>,
        keys: Tensor<B, 2>,
        mouse: Tensor<B, 3>,
        targets: Tensor<B, 4>,
    ) -> RegressionOutput<B> {
        let batch_size = targets.dims()[0];
        let device = targets.device();
        let schedule = CosineNoiseSchedule::new(NUM_TIMESTEPS);

        // Sample random timestep for each element in batch
        let t_indices: Vec<f32> = Tensor::<B, 1>::random(
            [batch_size],
            Distribution::Uniform(0.0, NUM_TIMESTEPS as f64),
            &device,
        )
        .to_data()
        .to_vec::<f32>()
        .unwrap();

        // Use the mean timestep for noise parameters (simplified batched approach)
        let mean_t = t_indices.iter().sum::<f32>() / batch_size as f32;
        let t_idx = mean_t as usize;
        let (alpha, sigma) = schedule.get(t_idx);

        // Create normalized timestep tensor for embedder
        let timestep_data: Vec<f32> = t_indices
            .iter()
            .map(|t| *t / NUM_TIMESTEPS as f32)
            .collect();
        let timestep = Tensor::from_data(
            burn::tensor::TensorData::new(timestep_data, [batch_size]),
            &device,
        );

        // Sample noise in latent space
        let (mu_probe, _) = self.vae.encode(targets.clone());
        let noise_shape = mu_probe.dims();
        let true_noise = Tensor::random(noise_shape, Distribution::Normal(0.0, 1.0), &device);

        // Forward: predict noise
        let (predicted_noise, mu, logvar) = self.forward_train(
            targets,
            keys,
            mouse,
            timestep,
            true_noise.clone(),
            alpha,
            sigma,
        );

        // MSE loss on noise prediction
        let noise_loss =
            MseLoss::new().forward(predicted_noise.clone(), true_noise.clone(), Reduction::Auto);

        // KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        let kl_loss =
            (mu.clone().powf_scalar(2.0) + logvar.clone().exp() - logvar - 1.0).mean() * 0.5;

        // Combined loss
        let loss = noise_loss + kl_loss * KL_WEIGHT;

        // Flatten for RegressionOutput
        let output_2d = predicted_noise.flatten(1, 3);
        let targets_2d = true_noise.flatten(1, 3);

        RegressionOutput::new(loss, output_2d, targets_2d)
    }
}

impl<B: AutodiffBackend> TrainStep for ModelV2<B> {
    type Input = FrameBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: FrameBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_diffusion(batch.images, batch.keys, batch.mouse, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for ModelV2<B> {
    type Input = FrameBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: FrameBatch<B>) -> RegressionOutput<B> {
        self.forward_diffusion(batch.images, batch.keys, batch.mouse, batch.targets)
    }
}
