use burn::{
    nn::{
        conv::{ConvTranspose2d, ConvTranspose2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

use common::*;

use crate::models::embedders::*;

use super::{
    blocks::*,
    decoder::{
        Decoder, DecoderConfig,
        unetplusplus::{UnetPlusPlusDecoder, UnetPlusPlusDecoderConfig},
    },
    encoder::{
        Encoder,
        resnet::{ResNet, ResNetConfig},
    },
};

#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    mouse_embedder: MouseEmbedder<B>,
    keys_embedder: KeyboardEmbedder<B>,
    // inc: ConvFusionBlock<B>,

    // down1: DownBlock<B>,
    // down2: DownBlock<B>,
    // down3: DownBlock<B>,
    // down4: DownBlock<B>,

    // up1: UpBlock<B>,
    // up2: UpBlock<B>,
    // up3: UpBlock<B>,
    // up4: UpBlock<B>,

    // out_conv: ConvTranspose2d<B>,
    // adaptive_pool: AdaptiveAvgPool2d,
    encoder: ResNet<B>,
    decoder: UnetPlusPlusDecoder<B>,
}

#[derive(Config, Debug)]
pub struct UNetConfig {
    #[config(default = "8")]
    base_channels: usize,
    #[config(default = "16")]
    embed_dim: usize,
    #[config(default = "16")]
    bottleneck_classes: usize,
}

impl UNetConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNet<B> {
        UNet {
            mouse_embedder: MouseEmbedderConfig::new(self.embed_dim, 16).init(device),
            keys_embedder: KeyboardEmbedderConfig::new(self.embed_dim, 16).init(device),
            // inc: ConvFusionBlockConfig::new(CHANNELS, self.base_channels, self.embed_dim)
            //     .init(device),
            // down1: DownBlockConfig::new(self.base_channels, self.base_channels * 2, self.embed_dim)
            //     .init(device),
            // down2: DownBlockConfig::new(
            //     self.base_channels * 2,
            //     self.base_channels * 4,
            //     self.embed_dim,
            // )
            // .init(device),
            // down3: DownBlockConfig::new(
            //     self.base_channels * 4,
            //     self.base_channels * 8,
            //     self.embed_dim,
            // )
            // .init(device),
            // down4: DownBlockConfig::new(
            //     self.base_channels * 8,
            //     self.base_channels * 16,
            //     self.embed_dim,
            // )
            // .init(device),

            // up1: UpBlockConfig::new(
            //     self.base_channels * 16,
            //     self.base_channels * 8,
            //     self.embed_dim,
            // )
            // .init(device),
            // up2: UpBlockConfig::new(
            //     self.base_channels * 8,
            //     self.base_channels * 4,
            //     self.embed_dim,
            // )
            // .init(device),
            // up3: UpBlockConfig::new(
            //     self.base_channels * 4,
            //     self.base_channels * 2,
            //     self.embed_dim,
            // )
            // .init(device),
            // up4: UpBlockConfig::new(self.base_channels * 2, self.base_channels, self.embed_dim)
            //     .init(device),

            // out_conv: ConvTranspose2dConfig::new([self.base_channels, CHANNELS], [3, 3])
            //     .init(device),
            // adaptive_pool: AdaptiveAvgPool2dConfig::new([HEIGHT, WIDTH]).init(),
            encoder: ResNetConfig::new([16, 8, 4, 2], self.bottleneck_classes, 4).init(device),
            decoder: UnetPlusPlusDecoderConfig::new(
                vec![],
                vec![
                    self.bottleneck_classes,
                    self.bottleneck_classes * 2,
                    self.bottleneck_classes * 4,
                    self.bottleneck_classes * 16,
                ],
            )
            .init(device),
        }
    }
}

impl<B: Backend> UNet<B> {
    pub fn forward(
        &self,
        images: Tensor<B, 4>,
        keys: Tensor<B, 2>,
        mouse: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        // Получаем эмбеддинги
        let mouse_emb = self.mouse_embedder.forward(mouse); // [embed_dim]
        let keys_emb = self.keys_embedder.forward(keys); // [embed_dim]

        let embed: Tensor<B, 3> = Tensor::cat(
            vec![
                mouse_emb.unsqueeze_dim(1),
                keys_emb.unsqueeze_dim(1),
                // timesteps_emb.unsqueeze_dim(1),
            ],
            2,
        ); // [b, 2, embed_dim]
        let embed: Tensor<B, 2> = embed.flatten(1, 2); // [b, embed_dim * 2]

        let [batch_size, channels, height, width] = images.dims();
        let [_, embedding_dim] = embed.dims();

        let embed_map = embed.unsqueeze_dims::<4>(&[2, 3]); // [embed_dim, 1, 1]
        let embed_map = embed_map.expand([batch_size, embedding_dim, height, width]); // [embed_dim, height, width]

        let x = Tensor::cat(vec![images, embed_map], 1); // [b, channels + embed_dim, ...]

        let x = self.encoder.forward(x, vec![]);
        let x = self.decoder.forward(x);

        // // Прямой ход по U-Net
        // let x1 = self.inc.forward(images, embed.clone()); // [channels, height / 3, width / 3]
        // let x2 = self.down1.forward(x1.clone(), embed.clone()); // [channels, height / (3 * 2 * 3), width / (3 * 2 * 3)]
        // let x3 = self.down2.forward(x2.clone(), embed.clone()); // [channels, height / (3 * (2 * 3) * 2), width / (3 * (2 * 3) * 2)]
        // let x4 = self.down3.forward(x3.clone(), embed.clone()); // [channels, height / (3 * (2 * 3) * 3), width / (3 * (2 * 3) * 3)]
        // let x5 = self.down4.forward(x4.clone(), embed.clone()); // [channels, height / (3 * (2 * 3) * 4), width / (3 * (2 * 3) * 4)]

        // // Обратный ход (skip connections)
        // let d1 = self.up1.forward(x5, x4, embed.clone()); // [channels, height / (3 * (2 * 3) * 3), width / (3 * (2 * 3) * 3)]
        // let d2 = self.up2.forward(d1, x3, embed.clone()); // [channels, height / (3 * (2 * 3) * 2), width / (3 * (2 * 3) * 2)]
        // let d3 = self.up3.forward(d2, x2, embed.clone()); // [channels, height / (3 * 2 * 3), width / (3 * 2 * 3)]
        // let d4 = self.up4.forward(d3, x1, embed); // [channels, height / 3, width / 3]

        // let out = self.out_conv.forward(d4);
        // let out = self.adaptive_pool.forward(out);

        // out

        x
    }
}
