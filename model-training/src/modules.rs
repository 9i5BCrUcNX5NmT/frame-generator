use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct ConvFusionModule<B: Backend> {
    conv1: Conv2d<B>,
    activation1: Relu,
    conv2: Conv2d<B>,
    activation2: Relu,
}

#[derive(Config, Debug)]
pub struct ConvFusionModuleConfig {
    in_channels: usize,
    out_channels: usize,
    embed_dim: usize,
}

impl ConvFusionModuleConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvFusionModule<B> {
        ConvFusionModule {
            conv1: Conv2dConfig::new(
                [self.in_channels + self.embed_dim, self.out_channels],
                [3, 3],
            )
            .init(device),
            activation1: Relu,
            conv2: Conv2dConfig::new([self.out_channels, self.out_channels], [3, 3]).init(device),
            activation2: Relu,
        }
    }
}

impl<B: Backend> ConvFusionModule<B> {
    /// Normal method added to a struct.
    pub fn forward(&self, input: Tensor<B, 4>, embed: Tensor<B, 2>) -> Tensor<B, 4> {
        let [batch_size, _channels, height, width] = input.dims();
        let [_, embedding_dim] = embed.dims();

        let embed_map = embed.unsqueeze_dims::<4>(&[2, 3]);
        let embed_map = embed_map.expand([batch_size, embedding_dim, height, width]);

        let x = Tensor::cat(vec![input, embed_map.clone()], 1);

        let x = self.conv1.forward(x);
        let x = self.activation1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.activation2.forward(x);

        x
    }
}

#[derive(Module, Debug)]
pub struct DownModule<B: Backend> {
    pool: MaxPool2d,
    conv: ConvFusionModule<B>,
}

#[derive(Config, Debug)]
pub struct DownModuleConfig {
    in_channels: usize,
    out_channels: usize,
    embed_dim: usize,
}

impl DownModuleConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DownModule<B> {
        DownModule {
            pool: MaxPool2dConfig::new([2, 2]).init(),
            conv: ConvFusionModuleConfig::new(self.in_channels, self.out_channels, self.embed_dim)
                .init(device),
        }
    }
}

impl<B: Backend> DownModule<B> {
    /// Normal method added to a struct.
    pub fn forward(&self, input: Tensor<B, 4>, embed: Tensor<B, 2>) -> Tensor<B, 4> {
        let x = self.pool.forward(input.clone());
        let x = self.conv.forward(x, embed);

        x
    }
}

#[derive(Module, Debug)]
pub struct UpModule<B: Backend> {
    up: ConvTranspose2d<B>,
    conv: ConvFusionModule<B>,
}

#[derive(Config, Debug)]
pub struct UpModuleConfig {
    in_channels: usize,
    out_channels: usize,
    embed_dim: usize,
}

impl UpModuleConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UpModule<B> {
        UpModule {
            up: ConvTranspose2dConfig::new([self.in_channels, self.out_channels], [2, 2])
                .init(device),
            conv: ConvFusionModuleConfig::new(
                self.out_channels * 2,
                self.out_channels,
                self.embed_dim,
            )
            .init(device),
        }
    }
}

impl<B: Backend> UpModule<B> {
    /// Normal method added to a struct.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
        skip: Tensor<B, 4>,
        embed: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        let mut x = self.up.forward(input.clone());

        let [_i1, _i2, i3, i4] = input.dims();
        let [_s1, _s2, s3, s4] = skip.dims();

        if i3 != s3 || i4 != s4 {
            let diff_y = s3 - i3;
            let diff_x = s4 - i4;
            x = x.pad(
                (
                    diff_x / 2,
                    diff_x - diff_x / 2,
                    diff_y / 2,
                    diff_y - diff_y / 2,
                ),
                0.0, // TODO: Другие виды заполнения?
            )
        }

        let x = Tensor::cat(vec![skip, x], 1);
        let x = self.conv.forward(x, embed);

        x
    }
}

#[derive(Module, Debug)]
pub struct MouseEmbedder<B: Backend> {
    linear1: Linear<B>,
    activation: Relu,
    linear2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct MouseEmbedderConfig {
    embed_dim: usize,
}

impl MouseEmbedderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MouseEmbedder<B> {
        MouseEmbedder {
            linear1: LinearConfig::new(2 * 200, 256).init(device),
            activation: Relu,
            linear2: LinearConfig::new(256, self.embed_dim).init(device),
        }
    }
}

impl<B: Backend> MouseEmbedder<B> {
    /// Normal method added to a struct.
    pub fn forward(&self, mouse: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = mouse.flatten(1, 2); // [n, 2, 200] -> [n, 400]
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);

        x
    }
}

#[derive(Module, Debug)]
pub struct KeyboardEmbedder<B: Backend> {
    linear1: Linear<B>,
    activation: Relu,
    linear2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct KeyboardEmbedderConfig {
    embed_dim: usize,
}

impl KeyboardEmbedderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> KeyboardEmbedder<B> {
        KeyboardEmbedder {
            linear1: LinearConfig::new(108, 256).init(device),
            activation: Relu,
            linear2: LinearConfig::new(256, self.embed_dim).init(device),
        }
    }
}

impl<B: Backend> KeyboardEmbedder<B> {
    /// Normal method added to a struct.
    pub fn forward(&self, keys: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(keys);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);

        x
    }
}
