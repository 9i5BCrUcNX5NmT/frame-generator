use burn::{
    module::Module,
    nn::{Embedding, Linear, Sigmoid, conv::Conv2d},
    prelude::Backend,
};

use crate::models::edm::blocks::FourierFeatures;

#[derive(Module, Debug)]
pub struct InnerModel<B: Backend> {
    noise_emb: FourierFeatures<B>,
    act_emb: Embedding<B>,
    cond_proj_1: Linear<B>,
    cond_proj_act: Sigmoid,
    cond_proj_1: Linear<B>,
    conv_in: Conv2d<B>,
    unet: Unet,
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
