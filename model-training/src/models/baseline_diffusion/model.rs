use burn::{
    nn::{
        Relu,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

use common::*;

use crate::models::embedders::{
    KeyboardEmbedder, KeyboardEmbedderConfig, MouseEmbedder, MouseEmbedderConfig,
};

#[derive(Module, Debug)]
pub struct Diffusion<B: Backend> {
    mouse_embedder: MouseEmbedder<B>,
    keys_embedder: KeyboardEmbedder<B>,

    conv1: Conv2d<B>,
    act1: Relu,
    conv2: ConvTranspose2d<B>,
    act2: Relu,

    out: AdaptiveAvgPool2d,
}

#[derive(Config, Debug)]
pub struct DiffusionConfig {
    #[config(default = "3")]
    in_channels: usize,
    #[config(default = "64")]
    out_channels: usize,
    #[config(default = "16")]
    embed_dim: usize,
}

impl DiffusionConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Diffusion<B> {
        Diffusion {
            mouse_embedder: MouseEmbedderConfig::new(self.embed_dim, self.embed_dim).init(device),
            keys_embedder: KeyboardEmbedderConfig::new(self.embed_dim, self.embed_dim).init(device),

            conv1: Conv2dConfig::new(
                [self.in_channels + self.embed_dim, self.out_channels],
                [3, 3],
            )
            .init(device),
            act1: Relu,
            conv2: ConvTranspose2dConfig::new([self.out_channels, self.in_channels], [3, 3])
                .init(device),
            act2: Relu,

            out: AdaptiveAvgPool2dConfig::new([HEIGHT, WIDTH]).init(),
        }
    }
}

impl<B: Backend> Diffusion<B> {
    pub fn forward(
        &self,
        images: Tensor<B, 4>,
        keys: Tensor<B, 2>,
        mouse: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        // Добавление шума
        let images = self.add_noise(images, 0.7);

        // Получаем эмбеддинги
        let mouse_emb = self.mouse_embedder.forward(mouse); // [embed_dim]
        let keys_emb = self.keys_embedder.forward(keys); // [embed_dim]

        // здесь для простоты просто суммируем
        let embed = mouse_emb + keys_emb; // [embed_dim]

        let [batch_size, _channels, height, width] = images.dims();
        let [_, embedding_dim] = embed.dims();

        let embed_map = embed.unsqueeze_dims::<4>(&[2, 3]); // [embed_dim, 1, 1]
        let embed_map = embed_map.expand([batch_size, embedding_dim, height, width]); // [embed_dim, height, width]

        let x = self.conv1.forward(embed_map);
        let x = self.act1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.act2.forward(x);

        let x = self.out.forward(x);

        x
    }

    fn add_noise(&self, input: Tensor<B, 4>, noise_level: f32) -> Tensor<B, 4> {
        // Добавление шума к входным данным
        let noise = input.random_like(burn::tensor::Distribution::Default);
        input * (1.0 - noise_level) + noise * (noise_level)
    }
}
