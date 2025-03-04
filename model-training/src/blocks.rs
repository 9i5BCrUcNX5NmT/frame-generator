use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use common::MOUSE_VECTOR_LENGTH;

// #[derive(Module, Debug)]
// pub struct ConvFusionBlock<B: Backend> {
//     conv1: Conv2d<B>,
//     activation1: Relu,
//     conv2: Conv2d<B>,
//     activation2: Relu,
// }

// #[derive(Config, Debug)]
// pub struct ConvFusionBlockConfig {
//     in_channels: usize,
//     out_channels: usize,
//     embed_dim: usize,
// }

// impl ConvFusionBlockConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> ConvFusionBlock<B> {
//         ConvFusionBlock {
//             conv1: Conv2dConfig::new(
//                 [self.in_channels + self.embed_dim, self.out_channels],
//                 [3, 3],
//             )
//             .init(device),
//             activation1: Relu,
//             conv2: Conv2dConfig::new([self.out_channels, self.out_channels], [3, 3]).init(device),
//             activation2: Relu,
//         }
//     }
// }

// impl<B: Backend> ConvFusionBlock<B> {
//     /// Normal method added to a struct.
//     pub fn forward(&self, input: Tensor<B, 4>, embed: Tensor<B, 2>) -> Tensor<B, 4> {
//         let [batch_size, _channels, height, width] = input.dims();
//         let [_, embedding_dim] = embed.dims();

//         let embed_map = embed.unsqueeze_dims::<4>(&[2, 3]); // [embed_dim, 1, 1]
//         let embed_map = embed_map.expand([batch_size, embedding_dim, height, width]); // [embed_dim, height, width]

//         let x = Tensor::cat(vec![input, embed_map.clone()], 1); // [embed_dim + channels, height, width]

//         let x = self.conv1.forward(x); // [channels, height / 3, width / 3]
//         let x = self.activation1.forward(x);
//         let x = self.conv2.forward(x); // [channels, height / 9, width / 9]
//         let x = self.activation2.forward(x);

//         x
//     }
// }

// #[derive(Module, Debug)]
// pub struct DownBlock<B: Backend> {
//     pool: MaxPool2d,
//     conv: ConvFusionBlock<B>,
// }

// #[derive(Config, Debug)]
// pub struct DownBlockConfig {
//     in_channels: usize,
//     out_channels: usize,
//     embed_dim: usize,
// }

// impl DownBlockConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> DownBlock<B> {
//         DownBlock {
//             pool: MaxPool2dConfig::new([2, 2]).init(),
//             conv: ConvFusionBlockConfig::new(self.in_channels, self.out_channels, self.embed_dim)
//                 .init(device),
//         }
//     }
// }

// impl<B: Backend> DownBlock<B> {
//     /// Normal method added to a struct.
//     pub fn forward(&self, input: Tensor<B, 4>, embed: Tensor<B, 2>) -> Tensor<B, 4> {
//         let x = self.pool.forward(input.clone());
//         let x = self.conv.forward(x, embed);

//         x
//     }
// }

// #[derive(Module, Debug)]
// pub struct UpBlock<B: Backend> {
//     up: ConvTranspose2d<B>,
//     conv: ConvFusionBlock<B>,
// }

// #[derive(Config, Debug)]
// pub struct UpBlockConfig {
//     in_channels: usize,
//     out_channels: usize,
//     embed_dim: usize,
// }

// impl UpBlockConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> UpBlock<B> {
//         UpBlock {
//             up: ConvTranspose2dConfig::new([self.in_channels, self.out_channels], [2, 2])
//                 .init(device),
//             conv: ConvFusionBlockConfig::new(
//                 self.out_channels * 2,
//                 self.out_channels,
//                 self.embed_dim,
//             )
//             .init(device),
//         }
//     }
// }

// impl<B: Backend> UpBlock<B> {
//     /// Normal method added to a struct.
//     pub fn forward(
//         &self,
//         input: Tensor<B, 4>,
//         skip: Tensor<B, 4>,
//         embed: Tensor<B, 2>,
//     ) -> Tensor<B, 4> {
//         let mut x = self.up.forward(input.clone());

//         let [_i1, _i2, i3, i4] = input.dims();
//         let [_s1, _s2, s3, s4] = skip.dims();

//         if i3 != s3 || i4 != s4 {
//             let diff_y = s3 - i3;
//             let diff_x = s4 - i4;

//             x = x.pad(
//                 (
//                     diff_x / 2,
//                     diff_x - diff_x / 2 - 1,
//                     diff_y / 2,
//                     diff_y - diff_y / 2 - 1,
//                 ),
//                 0.0, // TODO: Другие виды заполнения?
//             )
//         }

//         let x = Tensor::cat(vec![skip, x], 1);
//         let x = self.conv.forward(x, embed);

//         x
//     }
// }

#[derive(Module, Debug)]
pub struct MouseEmbedder<B: Backend> {
    linear1: Linear<B>,
    activation: Relu,
    linear2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct MouseEmbedderConfig {
    embed_dim: usize,
    hidden_dim: usize,
}

impl MouseEmbedderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MouseEmbedder<B> {
        MouseEmbedder {
            linear1: LinearConfig::new(2 * MOUSE_VECTOR_LENGTH, self.hidden_dim).init(device),
            activation: Relu,
            linear2: LinearConfig::new(self.hidden_dim, self.embed_dim).init(device),
        }
    }
}

impl<B: Backend> MouseEmbedder<B> {
    /// Normal method added to a struct.
    pub fn forward(&self, mouse: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = mouse.flatten(1, 2); // [n, 2, MOUSE_VECTOR_LENGTH] -> [n, MOUSE_VECTOR_LENGTH * 2]
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
    hidden_dim: usize,
}

impl KeyboardEmbedderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> KeyboardEmbedder<B> {
        KeyboardEmbedder {
            linear1: LinearConfig::new(108, self.hidden_dim).init(device),
            activation: Relu,
            linear2: LinearConfig::new(self.hidden_dim, self.embed_dim).init(device),
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

#[derive(Module, Debug)]
struct M {}

impl<B: Backend> M<B> {
    fn a() {}
}
