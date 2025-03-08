#[derive(Module, Debug)]
pub struct DiffusionSambpler<B: Backend> {
    conv: ConvTranspose2d<B>,
}

impl<B: Backend> DownBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(input)
    }
}

#[derive(Config, Debug)]
pub struct DownBlockConfig {
    in_channels: usize,
}

impl DownBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DownBlock<B> {
        DownBlock {
            conv: ConvTranspose2dConfig::new([self.in_channels, self.in_channels], [3, 3])
                .init(device),
        }
    }
}

fn build_sigmas(num_steps: usize, sigma_min: f64, sigma_max: f64, rho: i32) {}
