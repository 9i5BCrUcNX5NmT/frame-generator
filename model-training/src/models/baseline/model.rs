use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

use common::*;

use crate::models::embedders::{
    KeyboardEmbedder, KeyboardEmbedderConfig, MouseEmbedder, MouseEmbedderConfig,
};

#[derive(Module, Debug)]
pub struct Baseline<B: Backend> {
    mouse_embedder: MouseEmbedder<B>,
    keys_embedder: KeyboardEmbedder<B>,

    conv: Conv2d<B>,
    out: AdaptiveAvgPool2d,
}

#[derive(Config, Debug)]
pub struct BaselineConfig {
    #[config(default = "4")]
    in_channels: usize,
    #[config(default = "4")]
    out_channels: usize,
    #[config(default = "16")]
    embed_dim: usize,
}

impl BaselineConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Baseline<B> {
        Baseline {
            mouse_embedder: MouseEmbedderConfig::new(self.embed_dim, 16).init(device),
            keys_embedder: KeyboardEmbedderConfig::new(self.embed_dim, 16).init(device),

            conv: Conv2dConfig::new(
                [self.in_channels + self.embed_dim, self.out_channels],
                [1, 1],
            )
            .init(device),
            out: AdaptiveAvgPool2dConfig::new([HEIGHT, WIDTH]).init(),
        }
    }
}

impl<B: Backend> Baseline<B> {
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

        let [batch_size, _channels, height, width] = images.dims();
        let [_, embedding_dim] = embed.dims();

        let embed_map = embed.unsqueeze_dims::<4>(&[2, 3]); // [embed_dim, 1, 1]
        let embed_map = embed_map.expand([batch_size, embedding_dim, height, width]); // [embed_dim, height, width]

        let x = self.conv.forward(embed_map);
        let x = self.out.forward(x);

        x
    }
}
