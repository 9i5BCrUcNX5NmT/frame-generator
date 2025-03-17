use burn::{
    config::Config,
    module::Module,
    prelude::Backend,
    tensor::{Distribution, Float, Shape, Tensor},
};
use serde::Deserialize;

use super::{
    inner_model::{InnerModel, InnerModelConfig},
    utils::EdmBatch,
};

fn add_dims<B: Backend, const D: usize>(input: Tensor<B, 1>, n: usize) -> Tensor<B, D> {
    let mut new_dims = [0; D];

    let ndims = input.dims().len();

    for (i, dim) in input.dims().iter().enumerate() {
        new_dims[i] = dim + (n - ndims);
    }

    input.reshape(new_dims)
}

#[derive(Clone)]
struct Conditioners<B: Backend> {
    c_in: Tensor<B, 4>,
    c_out: Tensor<B, 4>,
    c_skip: Tensor<B, 4>,
    c_noise: Tensor<B, 1>,
}

#[derive(Debug, Deserialize, Clone)]
struct SigmaDistributionConfig {
    loc: f64,
    scale: f64,
    sigma_min: f64,
    sigma_max: f64,
}

#[derive(Module, Debug)]
pub struct Denoiser<B: Backend> {
    sigma_data: f64,
    sigma_offset_noise: f64,

    inner_model: InnerModel<B>,

    // SigmaDistributionConfig
    loc: f64,
    scale: f64,
    sigma_min: f64,
    sigma_max: f64,
}

impl<B: Backend> Denoiser<B> {
    pub fn forward(&self, batch: EdmBatch<B>) -> Tensor<B, 4> {
        let n: usize = todo!("self.cfg.inner_model.num_steps_conditioning");
        let seq_length = batch.obs.dims()[1] - n;

        let all_obs = batch.obs.clone();
        let mut loss: f64 = 0.0;

        for i in 0..seq_length {
            let obs = all_obs.slice([None, Some((i as i64, (n + i) as i64))]);
            let next_obs = all_obs.slice([None, Some(((n + i) as i64, (n + i) as i64))]);

            let [_b, t, c, h, w] = next_obs.dims();
            let next_obs = next_obs.reshape([t, c, h, w]); // b = 1

            let act = batch.act.slice([None, Some((i as i64, (n + i) as i64))]);
            let mask = batch
                .mask_padding
                .slice([None, Some(((n + i) as i64, (n + i) as i64))]);

            let [b, t, c, h, w] = obs.dims();
            let obs = obs.reshape([b, t * c, h, w]);

            let device = &self.devices()[0];

            let sigma = self.sample_sigma(b, device);
            let noisy_next_obs = self.apply_noise(next_obs, sigma, self.sigma_offset_noise, device);

            let cs = self.compute_conditioners(sigma);
            let model_output = self.compute_model_output(noisy_next_obs, obs, act, cs);

            let target = (next_obs - cs.c_skip * noisy_next_obs) / cs.c_out;
            todo!("loss += mse_loss(model_output[mask], target[mask])");

            let denoised = self.wrap_model_output(noisy_next_obs, model_output, cs);
            todo!("Обновление all_obs[:, n + i] = denoised")
        }

        loss /= seq_length as f64;
        todo!()
    }

    fn sample_sigma(&self, n: usize, device: &B::Device) -> Tensor<B, 1> {
        let sigma: Tensor<B, 1, Float> = Tensor::random([n], Distribution::Default, device);
        let sigma = sigma * self.scale + self.loc;
        let sigma = sigma.exp().clamp(self.sigma_min, self.sigma_max);

        sigma
    }

    fn apply_noise(
        &self,
        input: Tensor<B, 4>,
        sigma: Tensor<B, 1>,
        sigma_offset_noise: f64,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let [batch_size, channels, _height, _width] = input.dims();
        let offset_noise = Tensor::random(
            [batch_size, channels, 1, 1],
            burn::tensor::Distribution::Default,
            device,
        )
        .powf_scalar(sigma_offset_noise);

        input.clone()
            + offset_noise
            + input
                .random_like(burn::tensor::Distribution::Default)
                .powf(add_dims(sigma, 4))
    }

    fn compute_conditioners(&self, sigma: Tensor<B, 1>) -> Conditioners<B> {
        let sigma = (sigma.powi_scalar(2) + self.sigma_offset_noise).sqrt();
        let c_in = (sigma.clone().powi_scalar(2) + self.sigma_data.powi(2))
            .sqrt()
            .recip();
        let c_skip = (sigma.clone().powi_scalar(2) + self.sigma_data)
            .recip()
            .powf_scalar(self.sigma_data.powi(2));
        let c_out = sigma.clone() * c_skip.clone().sqrt();
        let c_noise = sigma.log().div_scalar(4);

        Conditioners {
            c_in: add_dims(c_in, 4),
            c_out: add_dims(c_out, 4),
            c_skip: add_dims(c_skip, 4),
            c_noise: add_dims(c_noise, 1),
        }
    }

    fn compute_model_output(
        &self,
        noisy_next_obs: Tensor<B, 4>,
        obs: Tensor<B, 4>,
        act: Tensor<B, 2>,
        cs: Conditioners<B>,
    ) -> Tensor<B, 4> {
        let rescaled_obs = obs.div_scalar(self.sigma_data);
        let rescaled_noise = noisy_next_obs.powf(cs.c_in);

        self.inner_model
            .forward(rescaled_noise, cs.c_noise, rescaled_obs, act)
    }

    fn wrap_model_output(
        &self,
        noisy_next_obs: Tensor<B, 4>,
        model_output: Tensor<B, 4>,
        cs: Conditioners<B>,
    ) -> Tensor<B, 4> {
        let denoised = cs.c_skip * noisy_next_obs + cs.c_out * model_output;

        denoised
            .clamp(-1, 1)
            .add_scalar(1)
            .div_scalar(2)
            .mul_scalar(255)
            .round()
            .div_scalar(255)
            .mul_scalar(2)
            .sub_scalar(1)
    }

    fn denoise(
        &self,
        noisy_next_obs: Tensor<B, 4>,
        sigma: Tensor<B, 1>,
        obs: Tensor<B, 4>,
        act: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        let cs = self.compute_conditioners(sigma);
        let model_output = self.compute_model_output(noisy_next_obs.clone(), obs, act, cs.clone());
        let denoised = self.wrap_model_output(noisy_next_obs, model_output, cs);

        denoised
    }
}

#[derive(Config, Debug)]
pub struct DenoiserConfig {
    inner_model: InnerModelConfig,
    sigma_data: f64,
    sigma_offset_noise: f64,
}

impl DenoiserConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Denoiser<B> {
        let cfg: SigmaDistributionConfig = todo!();
        Denoiser {
            inner_model: self.inner_model.init(device),
            sigma_data: self.sigma_data,
            sigma_offset_noise: self.sigma_offset_noise,
            loc: cfg.loc,
            scale: cfg.scale,
            sigma_min: cfg.sigma_min,
            sigma_max: cfg.sigma_max,
        }
    }

    // fn setup_training(&mut self, cfg: SigmaDistributionConfig) -> &mut Self {
    //     assert!(self.sample_sigma_training.is_none());
    // }
}
