use std::f32::consts::PI;

use burn::{
    config::Config,
    module::Module,
    nn::{
        GroupNorm, GroupNormConfig, Linear, LinearConfig, Sigmoid,
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        interpolate::{Interpolate2d, Interpolate2dConfig},
    },
    prelude::Backend,
    tensor::{Distribution, Tensor},
};

const GN_GROUP_SIZE: usize = 32;
const GN_EPS: f64 = 1e-5;
const ATTN_HEAD_DIM: usize = 8;

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

        let weight = Tensor::random([1, self.cond_channels / 2], Distribution::Default, device);

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
    channels: [usize; 2],
}

impl SmallResBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SmallResBlock<B> {
        SmallResBlock {
            group_norm: GroupNormConfig::new(self.channels[0], self.channels[0]).init(device),
            conv: Conv2dConfig::new([self.channels[0], self.channels[0]], [3, 3]).init(device),
            activation: Sigmoid,
            skip_projection: if self.channels[0] == self.channels[1] {
                None
            } else {
                Some(Conv2dConfig::new(self.channels, [1, 1]).init(device))
            },
        }
    }
}

#[derive(Module, Debug)]
pub struct AdaGroupNorm<B: Backend> {
    linear: Linear<B>,
    group_norm: GroupNorm<B>,
}

impl<B: Backend> AdaGroupNorm<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.group_norm.forward(input.clone());

        let y = self.linear.forward(input);
        let y = y.chunk(2, 1);
        let (scale, shift) = (y[0].clone(), y[1].clone());

        x * scale.add_scalar(1) + shift
    }
}

#[derive(Config, Debug)]
pub struct AdaGroupNormConfig {
    in_channels: usize,
    cond_channels: usize,
}

impl AdaGroupNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AdaGroupNorm<B> {
        let num_groups = 1.max(self.in_channels / GN_GROUP_SIZE);

        AdaGroupNorm {
            group_norm: GroupNormConfig::new(num_groups, self.in_channels).init(device),
            linear: LinearConfig::new(self.cond_channels, self.in_channels * 2).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    proj: Option<Conv2d<B>>,
    norm1: AdaGroupNorm<B>,
    conv1: Conv2d<B>,
    activation1: Sigmoid,
    norm2: AdaGroupNorm<B>,
    conv2: Conv2d<B>,
    activation2: Sigmoid,
    attn: Option<MultiHeadAttention<B>>,
}

impl<B: Backend> ResBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.norm1.forward(input.clone());
        let x = self.conv1.forward(x);
        let x = self.activation1.forward(x);

        let x = self.norm2.forward(x);
        let x = self.conv2.forward(x);
        let mut x = self.activation2.forward(x);

        if let Some(pr) = &self.proj {
            let r = pr.forward(input);
            x = x.add(r);
        }

        if let Some(att) = &self.attn {
            let mha_input = MhaInput::self_attn(x.flatten(1, 2)); // хз почему не сошлось
            x = att.forward(mha_input).context.unsqueeze(); // хз
        }

        x
    }
}

#[derive(Config, Debug)]
pub struct ResBlockConfig {
    channels: [usize; 2],
    cond_channels: usize,
    attn: bool,
}

impl ResBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResBlock<B> {
        let should_proj = self.channels[0] != self.channels[1];

        ResBlock {
            proj: if should_proj {
                Some(Conv2dConfig::new(self.channels, [1, 1]).init(device))
            } else {
                None
            },
            norm1: AdaGroupNormConfig::new(self.channels[0], self.cond_channels).init(device),
            conv1: Conv2dConfig::new(self.channels, [3, 3]).init(device),
            activation1: Sigmoid,
            norm2: AdaGroupNormConfig::new(self.channels[1], self.cond_channels).init(device),
            conv2: Conv2dConfig::new([self.channels[1], self.channels[1]], [3, 3]).init(device),
            activation2: Sigmoid,
            attn: if self.attn {
                Some(MultiHeadAttentionConfig::new(self.channels[1], ATTN_HEAD_DIM).init(device))
            } else {
                None
            },
        }
    }
}

#[derive(Module, Debug)]
pub struct ResBlocks<B: Backend> {
    res_blocks: Vec<ResBlock<B>>,
}

impl<B: Backend> ResBlocks<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
        to_cat: Option<Vec<Tensor<B, 4>>>,
    ) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let mut x = input;
        let mut outputs = vec![];

        for (i, res_block) in self.res_blocks.iter().enumerate() {
            x = if let Some(to_cat) = &to_cat {
                Tensor::cat(vec![x, to_cat[i].clone()], 1)
            } else {
                x
            };

            x = res_block.forward(x);
            outputs.push(x.clone());
        }

        (x, outputs)
    }
}

#[derive(Config, Debug)]
pub struct ResBlocksConfig {
    vec_channels: Vec<[usize; 2]>,
    cond_channels: usize,
    attn: bool,
}

impl ResBlocksConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResBlocks<B> {
        ResBlocks {
            res_blocks: self
                .vec_channels
                .iter()
                .map(|channels| {
                    ResBlockConfig::new(channels.clone(), self.cond_channels, self.attn)
                        .init(device)
                })
                .collect(),
        }
    }
}

#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    down_blocks: Vec<ResBlocks<B>>,
    up_blocks: Vec<ResBlocks<B>>,
    mid_blocks: ResBlocks<B>,
    downsamples: Vec<Option<DownBlock<B>>>,
    upsamples: Vec<Option<UpBlock<B>>>,
    num_down: usize,
}

impl<B: Backend> UNet<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Vec<Vec<Tensor<B, 4>>>, Vec<Vec<Tensor<B, 4>>>) {
        let [_, c, h, w] = input.dims();
        let n = self.num_down;

        let two_n = 2_usize.pow(n as u32);
        let padding_h = h.div_ceil(two_n) * two_n - h;
        let padding_w = w.div_ceil(two_n) * two_n - w;
        let mut x = input.pad((0, padding_w, 0, padding_h), 0.0);

        let mut down_outputs = vec![];
        for (block, down) in self.down_blocks.iter().zip(&self.downsamples) {
            let x_down = if let Some(down) = down {
                down.forward(x)
            } else {
                x
            };
            let (x_out, block_outputs) = block.forward(x_down, None);

            x = x_out;
            down_outputs.push(block_outputs);
        }

        let (mut x, _) = self.mid_blocks.forward(x, None);

        down_outputs.reverse();

        let mut up_outputs = vec![];
        for ((block, up), skip) in self
            .up_blocks
            .iter()
            .zip(&self.upsamples)
            .zip(&down_outputs)
        {
            let mut skip = skip.clone();
            skip.reverse();

            let x_up = if let Some(up) = up { up.forward(x) } else { x };
            let (x_out, block_outputs) = block.forward(x_up, Some(skip.to_vec()));

            x = x_out;
            up_outputs.push(block_outputs);
        }

        let new_b = x.dims()[0];
        let x = x.slice([0..new_b, 0..c, 0..h, 0..w]);

        (x, down_outputs, up_outputs)
    }
}

#[derive(Config, Debug)]
pub struct UNetConfig {
    cond_channels: usize,
    channels: Vec<usize>,
    depths: Vec<usize>,
    attn_depths: Vec<bool>,
}

impl UNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNet<B> {
        let num_down = self.channels.len() - 1;

        let mut down_blocks = vec![];
        let mut up_blocks = vec![];

        for (i, n) in self.depths.iter().enumerate() {
            let c1 = self.channels[0.max(i - 1)];
            let c2 = self.channels[i];

            let mut channels = vec![];
            channels.push([c1, c2]);
            for _ in 0..(n - 1) {
                channels.push([c1, c2]);
            }

            down_blocks.push(
                ResBlocksConfig::new(channels, self.cond_channels, self.attn_depths[i])
                    .init(device),
            );

            let mut channels = vec![];
            channels.push([c2 * 2, c2]);
            for _ in 0..(n - 1) {
                channels.push([c2 * 2, c2]);
            }
            channels.push([c1 + c2, c1]);

            up_blocks.push(
                ResBlocksConfig::new(channels, self.cond_channels, self.attn_depths[i])
                    .init(device),
            );
        }

        let last_channel = *self.channels.last().unwrap();
        let mid_blocks = ResBlocksConfig::new(
            vec![[last_channel, last_channel], [last_channel, last_channel]],
            self.cond_channels,
            true,
        )
        .init(device);

        let mut channels_without_last = self.channels[..(self.channels.len() - 1)].to_vec();
        let mut downsamples = vec![None];
        for channel in channels_without_last.iter() {
            downsamples.push(Some(DownBlockConfig::new(*channel).init(device)));
        }

        channels_without_last.reverse();

        let mut upsamples = vec![None];
        for channel in channels_without_last.iter() {
            upsamples.push(Some(UpBlockConfig::new(*channel).init(device)));
        }

        UNet {
            down_blocks,
            up_blocks,
            mid_blocks,
            downsamples,
            upsamples,
            num_down,
        }
    }
}
