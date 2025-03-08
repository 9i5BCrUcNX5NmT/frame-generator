use burn::{
    config::Config,
    module::Module,
    nn::Linear,
    prelude::Backend,
    tensor::{Float, Tensor},
};

#[derive(Module, Debug)]
pub struct DiffusionSampler<B: Backend> {}

impl<B: Backend> DiffusionSampler<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        todo!()
    }
}

#[derive(Config, Debug)]
pub struct DiffusionSamplerConfig {
    in_channels: usize,
}

impl DiffusionSamplerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DiffusionSampler<B> {
        // DiffusionSampler {}

        todo!()
    }
}

fn build_sigmas<B: Backend>(
    num_steps: usize,
    sigma_min: f64,
    sigma_max: f64,
    rho: i32,
    device: &B::Device,
) -> Tensor<B, 4> {
    let min_inv_rho = sigma_min.powf(1.0 / rho as f64);
    let max_inv_rho = sigma_max.powf(1.0 / rho as f64);

    let linspace: Tensor<B, 1> =
        Tensor::arange(0..num_steps as i64, device).float() / num_steps as f64;

    let sigmas = linspace
        .powf_scalar(min_inv_rho - max_inv_rho)
        .add_scalar(max_inv_rho)
        .powi_scalar(rho);

    let zero: Tensor<B, 1, Float> = Tensor::zeros([1], device);

    let sigmas = Tensor::cat(vec![sigmas, zero], 0);

    sigmas
}
