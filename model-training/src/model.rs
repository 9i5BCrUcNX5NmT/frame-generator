use burn::{
    nn::{
        Linear, LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

use crate::blocks::*;

use common::*;

#[derive(Module, Debug)]
pub struct MyModel<B: Backend> {
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
    down1: Conv2d<B>,
    activation1: Relu,

    up1: ConvTranspose2d<B>,
    activation2: Relu,
    // conv: Conv2d<B>,
    // activation: Relu,
    adaptive_pool: AdaptiveAvgPool2d,
}

#[derive(Config, Debug)]
pub struct MyModelConfig {
    #[config(default = "4")]
    in_channels: usize,
    #[config(default = "4")]
    base_channels: usize,
    #[config(default = "4")]
    out_channels: usize,
    #[config(default = "10")]
    embed_dim: usize,
}

impl MyModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MyModel<B> {
        MyModel {
            mouse_embedder: MouseEmbedderConfig::new(self.embed_dim, 16).init(device),
            keys_embedder: KeyboardEmbedderConfig::new(self.embed_dim, 16).init(device),
            // inc: ConvFusionBlockConfig::new(self.in_channels, self.base_channels, self.embed_dim)
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
            down1: Conv2dConfig::new(
                [self.in_channels + self.embed_dim, self.base_channels * 3],
                [3, 3],
            )
            .init(device),
            activation1: Relu,

            up1: ConvTranspose2dConfig::new([self.base_channels * 3, self.out_channels], [3, 3])
                .init(device),
            activation2: Relu,
            // conv: Conv2dConfig::new(
            //     [self.in_channels + self.embed_dim, self.out_channels],
            //     [1, 1],
            // )
            // .init(device),
            // activation: Relu,
            adaptive_pool: AdaptiveAvgPool2dConfig::new([HEIGHT, WIDTH]).init(),
        }
    }
}

impl<B: Backend> MyModel<B> {
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
        // let x1 = self.inc.forward(images, embed.clone()); // [channels, height / 9, width / 9]
        // let x2 = self.down1.forward(x1.clone(), embed.clone());
        // let x3 = self.down2.forward(x2.clone(), embed.clone());
        // let x4 = self.down3.forward(x3.clone(), embed.clone());
        // let x5 = self.down4.forward(x4.clone(), embed.clone());

        // Обратный ход (skip connections)
        // let d1 = self.up1.forward(x5, x4, embed.clone());
        // let d2 = self.up2.forward(d1, x3, embed.clone());
        // let d3 = self.up3.forward(d2, d1, embed.clone());
        // let d4 = self.up4.forward(d3, x1, embed);

        // let out = self.out_conv.forward(d4);
        // let out = self.adaptive_pool.forward(d4);

        // out

        let [batch_size, channels, height, width] = images.dims();
        let [_, embedding_dim] = embed.dims();

        let embed_map = embed.unsqueeze_dims::<4>(&[2, 3]); // [embed_dim, 1, 1]
        let embed_map = embed_map.expand([batch_size, embedding_dim, height, width]); // [embed_dim, height, width]

        let x = Tensor::cat(vec![images.clone(), embed_map.clone()], 1); // [embed_dim + channels, height, width]

        let x = self.down1.forward(x);
        // let x = self.activation1.forward(x);
        // let x = self.down2.forward(x);
        // let x = self.activation2.forward(x);

        let x = self.up1.forward(x);
        // let x = self.activation2.forward(x);
        // let x = self.up2.forward(x);
        // let x = self.activation4.forward(x);

        // let x = self.conv.forward(x);

        let x = self.adaptive_pool.forward(x);

        x
    }
}
