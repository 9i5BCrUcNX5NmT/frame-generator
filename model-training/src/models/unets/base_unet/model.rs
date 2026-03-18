use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        Relu,
    },
    prelude::*,
};
use common::{CHANNELS, HEIGHT, WIDTH};

/// Project conditional to bottleneck dimensions
#[derive(Module, Debug)]
struct ConditionalProject<B: Backend> {
    conv: Conv2d<B>,
    activation: Relu,
}

impl ConditionalProjectConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConditionalProject<B> {
        ConditionalProject {
            conv: Conv2dConfig::new([self.conditional_dim, self.hidden_dim * 4], [1, 1])
                .init(device),
            activation: Relu,
        }
    }
}

#[derive(Config, Debug)]
struct ConditionalProjectConfig {
    conditional_dim: usize,
    hidden_dim: usize,
}

impl<B: Backend> ConditionalProject<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        self.activation.forward(x)
    }
}

// TODO: Разделить модель диффузии и её саму

#[derive(Module, Debug)]
pub struct BaseUNet<B: Backend> {
    conv1: Conv2d<B>,
    act1: Relu,
    conv2: Conv2d<B>,
    act2: Relu,

    down1: MaxPool2d,

    conv3: Conv2d<B>,
    act3: Relu,
    conv4: Conv2d<B>,
    act4: Relu,

    down2: MaxPool2d,

    conv5: Conv2d<B>,
    act5: Relu,
    conv6: Conv2d<B>,
    act6: Relu,

    // Project conditional to bottleneck dimensions
    cond_project: ConditionalProject<B>,

    up1: ConvTranspose2d<B>,

    conv7: Conv2d<B>,
    act7: Relu,
    conv8: Conv2d<B>,
    act8: Relu,

    up2: ConvTranspose2d<B>,

    conv9: Conv2d<B>,
    act9: Relu,
    conv10: Conv2d<B>,
    act10: Relu,
    // out_conv: AdaptiveAvgPool2d,
}

#[derive(Config, Debug)]
pub struct BaseUNetConfig {
    #[config(default = "16")]
    embed_dim: usize,

    #[config(default = "16")]
    hidden_dim: usize,

    // FIXED: conditional_dim should be embed_dim*2 + embed_dim*3 = 5*embed_dim = 500
    // But we're passing CHANNELS(4) + embed_dim*3(300) = 304
    #[config(default = "304")]
    conditional_dim: usize,
}

impl BaseUNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BaseUNet<B> {
        BaseUNet {
            conv1: Conv2dConfig::new([CHANNELS, self.hidden_dim], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act1: Relu,
            conv2: Conv2dConfig::new([self.hidden_dim, self.hidden_dim], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act2: Relu,

            down1: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),

            conv3: Conv2dConfig::new([self.hidden_dim, self.hidden_dim * 2], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act3: Relu,
            conv4: Conv2dConfig::new([self.hidden_dim * 2, self.hidden_dim * 2], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act4: Relu,

            down2: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),

            conv5: Conv2dConfig::new([self.hidden_dim * 2, self.hidden_dim * 4], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act5: Relu,
            conv6: Conv2dConfig::new([self.hidden_dim * 4, self.hidden_dim * 4], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act6: Relu,

            // Project conditional to bottleneck dimensions
            cond_project: ConditionalProjectConfig {
                conditional_dim: self.conditional_dim,
                hidden_dim: self.hidden_dim,
            }
            .init(device),

            up1: ConvTranspose2dConfig::new([self.hidden_dim * 4, self.hidden_dim * 2], [2, 2])
                .with_stride([2, 2])
                .init(device),

            conv7: Conv2dConfig::new([self.hidden_dim * 4, self.hidden_dim * 2], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act7: Relu,
            conv8: Conv2dConfig::new([self.hidden_dim * 2, self.hidden_dim * 2], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act8: Relu,

            up2: ConvTranspose2dConfig::new([self.hidden_dim * 2, self.hidden_dim], [2, 2])
                .with_stride([2, 2])
                .init(device),

            conv9: Conv2dConfig::new([self.hidden_dim * 2, self.hidden_dim], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act9: Relu,
            conv10: Conv2dConfig::new([self.hidden_dim, CHANNELS], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act10: Relu,
            // out_conv: AdaptiveAvgPool2dConfig::new([HEIGHT, WIDTH]).init(),
        }
    }
}

impl<B: Backend> BaseUNet<B> {
    pub fn forward(
        &self,
        images: Tensor<B, 4>,
        conditional: Tensor<B, 4>, // Дополнительная информация
    ) -> Tensor<B, 4> {
        // Encoder
        let x = self.conv1.forward(images);
        let x = self.act1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.act2.forward(x);
        let skip1 = x.clone();

        let x = self.down1.forward(x);

        let x = self.conv3.forward(x);
        let x = self.act3.forward(x);
        let x = self.conv4.forward(x);
        let x = self.act4.forward(x);
        let skip2 = x.clone();

        let x = self.down2.forward(x);

        // Bottleneck
        let x = self.conv5.forward(x);
        let x = self.act5.forward(x);
        let x = self.conv6.forward(x);
        let x = self.act6.forward(x);

        // Добавить conditional после bottleneck
        // Project conditional to bottleneck spatial dimensions and add
        let cond_proj = self.cond_project.forward(conditional);
        // Pool twice to match bottleneck size (40x40 -> 20x20 -> 10x10)
        let cond_proj = self.down1.forward(cond_proj);
        let cond_proj = self.down2.forward(cond_proj);
        let x = x + cond_proj;

        // Decoder
        let mut x = self.up1.forward(x);

        let [_i1, _i2, i3, i4] = x.dims();
        let [_s1, _s2, s3, s4] = skip2.dims();

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

        let x = Tensor::cat(
            vec![
                x.clone(),
                skip2, // .clone()
                       // .narrow(2, 1, x.dims()[2])
                       // .clone()
                       // .narrow(3, 1, x.dims()[3]),
            ],
            1,
        ); // [b, channels + scip_channels = channels * 2, ...]

        let x = self.conv7.forward(x);
        let x = self.act7.forward(x);
        let x = self.conv8.forward(x);
        let x = self.act8.forward(x);

        let mut x = self.up2.forward(x);

        let [_i1, _i2, i3, i4] = x.dims();
        let [_s1, _s2, s3, s4] = skip1.dims();

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

        let x = Tensor::cat(
            vec![
                x.clone(),
                skip1, // .narrow(2, 1, x.dims()[2]).narrow(3, 1, x.dims()[3]),
            ],
            1,
        );

        let x = self.conv9.forward(x);
        let x = self.act9.forward(x);
        let x = self.conv10.forward(x);
        let x = self.act10.forward(x);

        // let x = self.out_conv.forward(x);

        x
    }
}
