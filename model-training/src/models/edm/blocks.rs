use std::f32::consts::PI;

use burn::{
    config::Config,
    module::Module,
    nn::{
        GroupNorm, GroupNormConfig, Sigmoid, SwiGlu,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        interpolate::{Interpolate2d, Interpolate2dConfig},
    },
    prelude::Backend,
    tensor::Tensor,
};

#[derive(Module, Debug)]
pub struct FourierFeatures<B: Backend> {
    weight: Tensor<B, 2>,
}

impl<B: Backend> FourierFeatures<B> {
    pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 2> {
        let x = input.unsqueeze().mul_scalar(2.0 * PI);
        let x = x.matmul(self.weight.clone());
        let x = Tensor::cat(vec![x.clone().cos(), x.clone().sin()], x.dims().len()); // -1 replace to "last dim"

        x
    }
}

#[derive(Config, Debug)]
pub struct FourierFeaturesConfig {
    cond_channels: usize,
}

impl FourierFeaturesConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FourierFeatures<B> {
        assert!(self.cond_channels % 2 == 0); // зачем?

        let weight = Tensor::random(
            [1, self.cond_channels / 2],
            burn::tensor::Distribution::Default,
            device,
        );

        FourierFeatures { weight }
    }
}

#[derive(Module, Debug)]
pub struct DownBlock<B: Backend> {
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

#[derive(Module, Debug)]
pub struct UpBlock<B: Backend> {
    conv: Conv2d<B>,
    interpolate: Interpolate2d,
}

impl<B: Backend> UpBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.interpolate.forward(input);
        let x = self.conv.forward(x);

        x
    }
}

#[derive(Config, Debug)]
pub struct UpBlockConfig {
    in_channels: usize,
}

impl UpBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UpBlock<B> {
        UpBlock {
            conv: Conv2dConfig::new([self.in_channels, self.in_channels], [3, 3]).init(device),
            interpolate: Interpolate2dConfig::new().init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct SmallResBlock<B: Backend> {
    group_norm: GroupNorm<B>,
    conv: Conv2d<B>,
    activation: Sigmoid,
    skip_projection: Option<Conv2d<B>>,
}

impl<B: Backend> SmallResBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        if let Some(skip_conv) = &self.skip_projection {
            skip_conv.forward(input.clone()) + self.conv.forward(input)
        } else {
            self.conv.forward(input)
        }
    }
}

#[derive(Config, Debug)]
pub struct SmallResBlockConfig {
    in_channels: usize,
    out_channels: usize,
}

impl SmallResBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SmallResBlock<B> {
        SmallResBlock {
            group_norm: GroupNormConfig::new(self.in_channels, self.in_channels).init(device),
            conv: Conv2dConfig::new([self.in_channels, self.in_channels], [3, 3]).init(device),
            activation: Sigmoid,
            skip_projection: if self.in_channels == self.out_channels {
                None
            } else {
                Some(Conv2dConfig::new([self.in_channels, self.out_channels], [1, 1]).init(device))
            },
        }
    }
}

#[derive(Module, Debug)]
pub struct UNet<B: Backend> {}

impl<B: Backend> UNet<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {}
}

#[derive(Config, Debug)]
pub struct DownBlockConfig {}

impl DownBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNet<B> {
        UNet {}
    }
}
