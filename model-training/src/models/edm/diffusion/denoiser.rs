use burn::{
    prelude::Backend,
    tensor::{Float, Shape, Tensor},
};

use super::inner_model::{InnerModel, InnerModelConfig};

fn add_dims<B: Backend, const D: usize>(input: Tensor<B, D>, n: usize) -> Tensor<B, D> {
    let mut new_dims = [0; D];

    let ndims = input.dims().len();

    for (i, dim) in input.dims().iter().enumerate() {
        new_dims[i] = dim + (n - ndims);
    }

    input.reshape(new_dims)
}

struct Conditioners<B: Backend, const D: usize> {
    c_in: Tensor<B, D>,
    c_out: Tensor<B, D>,
    c_skip: Tensor<B, D>,
    c_noise: Tensor<B, D>,
}

struct SigmaDistributionConfig {
    loc: f64,
    scale: f64,
    sigma_min: f64,
    sigma_max: f64,
}

#[derive(Module, Debug)]
pub struct Denoiser<B: Backend> {
    inner_model: InnerModel<B>,
}

impl<B: Backend> Denoiser<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        todo!()
    }
}

#[derive(Config, Debug)]
pub struct DenoiserConfig {
    sigma_data: f64,
    sigma_offset_noise: f64,
    inner_model_config: InnerModelConfig,
}

impl DenoiserConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Denoiser<B> {
        Denoiser {
            inner_model: self.inner_model_config.init(device),
        }
    }

    fn setup_training(&self, cfg: SigmaDistributionConfig) {
        assert!(self.s)
    }

    fn sample_sigma<B: Backend>(
        n: usize,
        cfg: SigmaDistributionConfig,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let sigma: Tensor<B, 1, Float> =
            Tensor::random([n], burn::tensor::Distribution::Default, device);
        let sigma = sigma * cfg.scale + cfg.loc;
        let sigma = sigma.exp().

        todo!()
    }
}
