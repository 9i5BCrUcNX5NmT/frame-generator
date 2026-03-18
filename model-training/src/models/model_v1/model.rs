use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};

use common::*;

use crate::models::{
    embedders::{
        KeyboardEmbedder, KeyboardEmbedderConfig, MouseEmbedder, MouseEmbedderConfig,
        TimestepEmbedder, TimestepEmbedderConfig,
    },
    unets::base_unet::model::{BaseUNet, BaseUNetConfig},
};

#[derive(Module, Debug)]
pub struct ConditionalBlock<B: Backend> {
    linear1: Linear<B>,
    activation: Relu,
    linear2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ConditionalBlockConfig {
    condition_dim: usize,
}

impl ConditionalBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConditionalBlock<B> {
        ConditionalBlock {
            // FIX: LinearConfig::new(output, input)
            // input: CHANNELS * HEIGHT * WIDTH = 4 * 40 * 40 = 6400
            // output: same
            linear1: LinearConfig::new(self.condition_dim, self.condition_dim).init(device),
            activation: Relu,
            linear2: LinearConfig::new(self.condition_dim, self.condition_dim).init(device),
        }
    }
}

impl<B: Backend> ConditionalBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let shape = x.shape();
        let x: Tensor<B, 2> = x.flatten(1, 3);

        // panic!(
        //     "{:?} {:?} {} {}",
        //     shape,
        //     x.dims(),
        //     self.linear1,
        //     CHANNELS * HEIGHT * WIDTH
        // );

        let x = self.linear1.forward(x); // TODO: ошибка здест ERROR
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);

        let x = x.reshape(shape);

        x
    }
}

#[derive(Module, Debug)]
pub struct ModelV1<B: Backend> {
    mouse_embedder: MouseEmbedder<B>,
    keys_embedder: KeyboardEmbedder<B>,
    timestep_embedder: TimestepEmbedder<B>,

    // обработка дополнительной информации
    conditional: ConditionalBlock<B>,

    unet: BaseUNet<B>,
}

#[derive(Config, Debug)]
pub struct ModelV1Config {
    #[config(default = "100")]
    embed_dim: usize,
}

impl ModelV1Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ModelV1<B> {
        ModelV1 {
            mouse_embedder: MouseEmbedderConfig::new(self.embed_dim, self.embed_dim).init(device),
            keys_embedder: KeyboardEmbedderConfig::new(self.embed_dim, self.embed_dim).init(device),
            timestep_embedder: TimestepEmbedderConfig::new(self.embed_dim).init(device),

            conditional: ConditionalBlockConfig::new(CHANNELS * HEIGHT * WIDTH).init(device),

            unet: BaseUNetConfig::new()
                .with_embed_dim(self.embed_dim)
                .with_conditional_dim(304) // CHANNELS(4) + embed_dim*3(300) = 304
                .init(device),
        }
    }
}

impl<B: Backend> ModelV1<B> {
    pub fn forward(
        &self,
        images: Tensor<B, 4>,
        keys: Tensor<B, 2>,
        mouse: Tensor<B, 3>,
        next_noise: Tensor<B, 4>, // conditional layers || Зашумлённый следующий кадр при тренировке или случайный шум при генерации
        timestep: Tensor<B, 1>,   // Timestep for diffusion
    ) -> Tensor<B, 4> {
        let [batch_size, channels, height, width] = images.dims();

        // Получаем эмбеддинги
        let mouse_emb = self.mouse_embedder.forward(mouse); // [b, embed_dim]
        let keys_emb = self.keys_embedder.forward(keys); // [b, embed_dim]
        let time_emb = self.timestep_embedder.forward(timestep); // [b, embed_dim]

        // обрабатываем доп информацию
        let embed: Tensor<B, 3> = Tensor::cat(
            vec![
                mouse_emb.unsqueeze_dim(1),
                keys_emb.unsqueeze_dim(1),
                time_emb.unsqueeze_dim(1),
            ],
            2,
        ); // [b, 3, embed_dim]

        // Build conditional for UNet: combine processed next_noise with embedded keys/mouse/time
        // next_noise after conditional block: [batch, CHANNELS=4, 40, 40]
        // embed_map after expand: [batch, embed_dim*3=300, 40, 40]
        // Combined: [batch, 4+300=304, 40, 40]

        // Process next_noise through conditional block
        let conditional = self.conditional.forward(next_noise); // [batch, 4, 40, 40]

        let embed: Tensor<B, 2> = embed.flatten(1, 2); // [b, embed_dim * 3]

        let [_, embedding_dim] = embed.dims();

        let embed_map = embed.unsqueeze_dims::<4>(&[2, 3]); // [batch, 300, 1, 1]
        let embed_map = embed_map.expand([batch_size, embedding_dim, height, width]); // [batch, 300, 40, 40]

        let conditional = Tensor::cat(vec![conditional, embed_map], 1); // [batch, 304, 40, 40]

        // Message:  Dimensions are incompatible for matrix multiplication.
        // FIX: Pass images to UNet (has 4 channels), conditional is used inside UNet
        let x = self.unet.forward(images.clone(), conditional);

        let x = x.reshape([batch_size, channels, height, width]);

        x
    }
}
