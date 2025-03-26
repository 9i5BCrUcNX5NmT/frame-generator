use burn::{
    nn::{
        conv::{ConvTranspose2d, ConvTranspose2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

use common::*;

use crate::models::embedders::*;

use super::blocks::*;

#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    mouse_embedder: MouseEmbedder<B>,
    keys_embedder: KeyboardEmbedder<B>,

    inc: ConvFusionBlock<B>,

    down1: DownBlock<B>,
    down2: DownBlock<B>,
    down3: DownBlock<B>,
    down4: DownBlock<B>,

    up1: UpBlock<B>,
    up2: UpBlock<B>,
    up3: UpBlock<B>,
    up4: UpBlock<B>,

    out_conv: ConvTranspose2d<B>,
    adaptive_pool: AdaptiveAvgPool2d,
}

#[derive(Config, Debug)]
pub struct UNetConfig {
    in_channels: usize,
    #[config(default = "8")]
    base_channels: usize,
    out_channels: usize,
    #[config(default = "16")]
    embed_dim: usize,
    #[config(default = 10)]
    pub num_timestamps: usize,
}

impl UNetConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNet<B> {
        UNet {
            mouse_embedder: MouseEmbedderConfig::new(self.embed_dim, 16).init(device),
            keys_embedder: KeyboardEmbedderConfig::new(self.embed_dim, 16).init(device),
            inc: ConvFusionBlockConfig::new(self.in_channels, self.base_channels, self.embed_dim)
                .init(device),
            down1: DownBlockConfig::new(self.base_channels, self.base_channels * 2, self.embed_dim)
                .init(device),
            down2: DownBlockConfig::new(
                self.base_channels * 2,
                self.base_channels * 4,
                self.embed_dim,
            )
            .init(device),
            down3: DownBlockConfig::new(
                self.base_channels * 4,
                self.base_channels * 8,
                self.embed_dim,
            )
            .init(device),
            down4: DownBlockConfig::new(
                self.base_channels * 8,
                self.base_channels * 16,
                self.embed_dim,
            )
            .init(device),

            up1: UpBlockConfig::new(
                self.base_channels * 16,
                self.base_channels * 8,
                self.embed_dim,
            )
            .init(device),
            up2: UpBlockConfig::new(
                self.base_channels * 8,
                self.base_channels * 4,
                self.embed_dim,
            )
            .init(device),
            up3: UpBlockConfig::new(
                self.base_channels * 4,
                self.base_channels * 2,
                self.embed_dim,
            )
            .init(device),
            up4: UpBlockConfig::new(self.base_channels * 2, self.base_channels, self.embed_dim)
                .init(device),

            out_conv: ConvTranspose2dConfig::new([self.base_channels, self.out_channels], [3, 3])
                .init(device),
            adaptive_pool: AdaptiveAvgPool2dConfig::new([HEIGHT, WIDTH]).init(),
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

        // здесь для простоты просто суммируем
        let embed = mouse_emb + keys_emb; // [embed_dim]

        // Прямой ход по U-Net
        let x1 = self.inc.forward(images, embed.clone()); // [channels, height / 3, width / 3]
        let x2 = self.down1.forward(x1.clone(), embed.clone()); // [channels, height / (3 * 2 * 3), width / (3 * 2 * 3)]
        let x3 = self.down2.forward(x2.clone(), embed.clone()); // [channels, height / (3 * (2 * 3) * 2), width / (3 * (2 * 3) * 2)]
        let x4 = self.down3.forward(x3.clone(), embed.clone()); // [channels, height / (3 * (2 * 3) * 3), width / (3 * (2 * 3) * 3)]
        let x5 = self.down4.forward(x4.clone(), embed.clone()); // [channels, height / (3 * (2 * 3) * 4), width / (3 * (2 * 3) * 4)]

        // Обратный ход (skip connections)
        let d1 = self.up1.forward(x5, x4, embed.clone()); // [channels, height / (3 * (2 * 3) * 3), width / (3 * (2 * 3) * 3)]
        let d2 = self.up2.forward(d1, x3, embed.clone()); // [channels, height / (3 * (2 * 3) * 2), width / (3 * (2 * 3) * 2)]
        let d3 = self.up3.forward(d2, x2, embed.clone()); // [channels, height / (3 * 2 * 3), width / (3 * 2 * 3)]
        let d4 = self.up4.forward(d3, x1, embed); // [channels, height / 3, width / 3]

        let out = self.out_conv.forward(d4);
        let out = self.adaptive_pool.forward(out);

        out
    }
}
