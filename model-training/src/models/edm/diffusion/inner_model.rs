use burn::{module::Module, nn::Embedding, prelude::Backend};

use crate::models::edm::blocks::FourierFeatures;

#[derive(Module, Debug)]
pub struct InnerModel<B: Backend> {
    noise_emb: FourierFeatures<B>,
    act_emb: Embedding<B>,
}

impl<B: Backend> InnerModel<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(input)
    }
}

#[derive(Config, Debug)]
pub struct DownBlockConfig {
    in_channels: usize,
}

impl DownBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> InnerModel<B> {
        InnerModel {
            conv: ConvTranspose2dConfig::new([self.in_channels, self.in_channels], [3, 3])
                .init(device),
        }
    }
}
