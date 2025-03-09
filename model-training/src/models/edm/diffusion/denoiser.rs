use burn::{
    config::Config,
    module::Module,
    prelude::Backend,
    tensor::{Float, Shape, Tensor},
};
use serde::Deserialize;

use super::inner_model::{InnerModel, InnerModelConfig};

fn add_dims<B: Backend, const D: usize>(input: Tensor<B, D>, n: usize) -> Tensor<B, D> {
    let mut new_dims = [0; D];

    let ndims = input.dims().len();

    for (i, dim) in input.dims().iter().enumerate() {
        new_dims[i] = dim + (n - ndims);
    }

    input.reshape(new_dims)
}

struct Conditioners<B: Backend> {
    c_in: Tensor<B, 4>,
    c_out: Tensor<B, 4>,
    c_skip: Tensor<B, 4>,
    c_noise: Tensor<B, 1>,
}

#[derive(Debug, Deserialize)]
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
    sigma_distribution: SigmaDistribution,
}

impl<B: Backend> Denoiser<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        todo!()
    }

    fn sample_sigma<B: Backend>(&self, n: usize, device: &B::Device) -> Tensor<B, 1> {
        let sigma: Tensor<B, 1, Float> = Tensor::random([n], self.sigma_distribution, device);
        let sigma =
            sigma * self.sigma_distribution_config.scale + self.sigma_distribution_config.loc;
        let sigma = sigma.exp().clamp(
            self.sigma_distribution_config.sigma_min,
            self.sigma_distribution_config.sigma_max,
        );

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

        input
            + offset_noise
            + input
                .random_like(burn::tensor::Distribution::Default)
                .powf(add_dims(sigma, 4))
    }

    fn compute_conditioners(&self, sigma: Tensor<B, 1>) -> Conditioners<B, 4> {
        let sigma = (sigma.powi_scalar(2) + self.sigma_offset_noise).sqrt();
        let c_in = (sigma.powi_scalar(2) + self.sigma_data.powi(2))
            .sqrt()
            .recip();
        let c_skip = (sigma.powi(2) + self.sigma_data)
            .recip()
            .powf_scalar(self.sigma_data.powi(2));
        let c_out = sigma * c_skip.sqrt();
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
        act: Tensor<B, 4>,
        cs: Conditioners<B>,
    ) -> Tensor<B, 4> {
        let rescaled_obs = obs.div_scalar(self.sigma_data);
        let rescaled_noise = noisy_next_obs.powf(cs.c_in);

        self.inner_model_config;

        todo!()
    }
}

#[derive(Config, Debug)]
pub struct DenoiserConfig {
    sigma_data: f64,
    sigma_offset_noise: f64,
}

impl DenoiserConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Denoiser<B> {
        Denoiser {
            inner_model: self.inner_model_config.init(device),
            sigma_data: self.sigma_data,
            sigma_offset_noise: self.sigma_offset_noise,
            sigma_distribution: todo!(),
        }
    }

    // fn setup_training(&mut self, cfg: SigmaDistributionConfig) -> &mut Self {
    //     assert!(self.sample_sigma_training.is_none());
    // }
}
