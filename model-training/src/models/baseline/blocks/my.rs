use burn::{
    config::Config,
    module::Module,
    nn::{
        self, BatchNorm,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::Backend,
    tensor::Tensor,
};
use common::{HEIGHT, WIDTH};

/// Layer block of generator model
#[derive(Module, Debug)]
pub struct LayerBlock<B: Backend> {
    fc: nn::Linear<B>,
    bn: nn::BatchNorm<B, 0>,
    leakyrelu: nn::LeakyRelu,
}

impl<B: Backend> LayerBlock<B> {
    pub fn new(input: usize, output: usize, device: &B::Device) -> Self {
        let fc = nn::LinearConfig::new(input, output)
            .with_bias(true)
            .init(device);
        let bn: BatchNorm<B, 0> = nn::BatchNormConfig::new(output)
            .with_epsilon(0.8)
            .init(device);
        let leakyrelu = nn::LeakyReluConfig::new().with_negative_slope(0.2).init();

        Self { fc, bn, leakyrelu }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let output = self.fc.forward(input); // output: [Batch, x]
        let output = self.bn.forward(output); // output: [Batch, x]

        self.leakyrelu.forward(output) // output: [Batch, x]
    }
}

// #[derive(Module, Debug)]
// pub struct Encoder<B: Backend> {
//     conv1: Conv2d<B>,
//     conv2: Conv2d<B>,
//     batch_norm: BatchNorm<B, 4>,
// }

// #[derive(Config, Debug)]
// pub struct EncoderConfig {
//     in_channels: usize,
//     out_channels: usize,
// }

// impl EncoderConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
//         Encoder {
//             conv1: Conv2dConfig::new([HEIGHT / 2, WIDTH], []),
//             conv2: (),
//             batch_norm: (),
//         }
//     }
// }

// impl<B: Backend> Encoder<B> {
//     /// Normal method added to a struct.
//     pub fn forward(&self, input: Tensor<B, 4>, embed: Tensor<B, 2>) -> Tensor<B, 4> {
//         let x = self.pool.forward(input.clone());
//         let x = self.conv.forward(x, embed);

//         x
//     }
// }
