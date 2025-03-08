use burn::{
    config::{self, Config},
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, GroupNorm, GroupNormConfig, Linear, LinearConfig, Sigmoid,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::Backend,
    tensor::Tensor,
};

use crate::models::edm::blocks::{
    FourierFeatures, FourierFeaturesConfig, GN_GROUP_SIZE, UNet, UNetConfig,
};

#[derive(Module, Debug)]
pub struct InnerModel<B: Backend> {
    noise_emb: FourierFeatures<B>,
    act_emb: Embedding<B>,
    cond_proj_1: Linear<B>,
    cond_proj_activation: Sigmoid,
    cond_proj_2: Linear<B>,
    conv_in: Conv2d<B>,
    unet: UNet<B>,
    norm_out: GroupNorm<B>,
    norm_out_activation: Sigmoid,
    conv_out: Conv2d<B>,
}

impl<B: Backend> InnerModel<B> {
    pub fn forward(
        &self,
        noisy_next_obs: Tensor<B, 4>,
        c_noise: Tensor<B, 1>,
        obs: Tensor<B, 4>,
        act: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        let noise_emb_out = self.noise_emb.forward(c_noise);

        let act_emb_out = self.act_emb.forward(act.int());
        let act_emb_out: Tensor<B, 2> = act_emb_out.flatten(1, 2);

        let cond = noise_emb_out.add(act_emb_out);
        let cond = self.cond_proj_1.forward(cond);
        let cond = self.cond_proj_activation.forward(cond);
        let cond = self.cond_proj_2.forward(cond);

        let x = Tensor::cat(vec![obs, noisy_next_obs], 1);
        let x = self.conv_in.forward(x);

        let (x, _, _) = self.unet.forward(x, cond);

        let x = self.norm_out.forward(x);
        let x = self.norm_out_activation.forward(x);
        let x = self.conv_out.forward(x);

        x
    }
}

#[derive(Config, Debug)]
pub struct InnerModelConfig {
    img_channels: usize,
    num_steps_conditioning: usize,
    cond_channels: usize,
    depths: Vec<usize>,
    channels: Vec<usize>,
    attn_depths: Vec<bool>,
    #[config(default = "None")]
    num_actions: Option<usize>,
}

impl InnerModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> InnerModel<B> {
        let num_action = if let Some(num_act) = self.num_actions {
            num_act
        } else {
            0
        }; // TODO: хз
        InnerModel {
            noise_emb: FourierFeaturesConfig::new(self.cond_channels).init(device),
            act_emb: EmbeddingConfig::new(
                num_action,
                self.cond_channels / self.num_steps_conditioning,
            )
            .init(device),
            cond_proj_1: LinearConfig::new(self.cond_channels, self.cond_channels).init(device),
            cond_proj_activation: Sigmoid,
            cond_proj_2: LinearConfig::new(self.cond_channels, self.cond_channels).init(device),
            conv_in: Conv2dConfig::new(
                [
                    (self.num_steps_conditioning + 1) * self.img_channels,
                    self.channels[0],
                ],
                [3, 3],
            )
            .init(device),
            unet: UNetConfig::new(
                self.cond_channels,
                self.channels.clone(),
                self.depths.clone(),
                self.attn_depths.clone(),
            )
            .init(device),
            norm_out: GroupNormConfig::new(
                1.max(self.channels[0] / GN_GROUP_SIZE),
                self.channels[0],
            )
            .init(device),
            norm_out_activation: Sigmoid,
            conv_out: Conv2dConfig::new([self.channels[0], self.img_channels], [3, 3]).init(device),
        }
    }
}
