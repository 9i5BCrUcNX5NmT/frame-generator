use burn::{nn::BatchNorm, prelude::*};

use common::*;

use crate::models::embedders::{
    KeyboardEmbedder, KeyboardEmbedderConfig, MouseEmbedder, MouseEmbedderConfig,
};

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

#[derive(Module, Debug)]
pub struct WganDecoder<B: Backend> {
    mouse_embedder: MouseEmbedder<B>,
    keys_embedder: KeyboardEmbedder<B>,
    // timestep_embedder: TimestempEmbedder<B>,
    layer1: LayerBlock<B>,
    layer2: LayerBlock<B>,
    layer3: LayerBlock<B>,
    layer4: LayerBlock<B>,
    fc: nn::Linear<B>,
    tanh: nn::Tanh,
    // unet: UNet<B>,
}

#[derive(Config, Debug)]
pub struct WganDecoderConfig {
    #[config(default = "16")]
    embed_dim: usize,
    // #[config(default = 10)]
    // pub num_timestamps: usize,
}

impl WganDecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WganDecoder<B> {
        let layer1 = LayerBlock::new(
            (CHANNELS + self.embed_dim * 2) * WIDTH * HEIGHT,
            128,
            device,
        );
        let layer2 = LayerBlock::new(128, 256, device);
        let layer3 = LayerBlock::new(256, 512, device);
        let layer4 = LayerBlock::new(512, 1024, device);
        let fc = nn::LinearConfig::new(1024, CHANNELS * WIDTH * HEIGHT)
            .with_bias(true)
            .init(device);

        WganDecoder {
            mouse_embedder: MouseEmbedderConfig::new(self.embed_dim, self.embed_dim).init(device),
            keys_embedder: KeyboardEmbedderConfig::new(self.embed_dim, self.embed_dim).init(device),
            // timestep_embedder: TimestempEmbedderConfig::new(self.embed_dim, self.embed_dim)
            //     .init(device),
            layer1,
            layer2,
            layer3,
            layer4,
            fc,
            tanh: nn::Tanh,
            // unet: UNetConfig::new().init(device),
        }
    }
}

impl<B: Backend> WganDecoder<B> {
    pub fn forward(
        &self,
        images: Tensor<B, 4>,
        keys: Tensor<B, 2>,
        mouse: Tensor<B, 3>,
        // timesteps: Tensor<B, 1>,
    ) -> Tensor<B, 4> {
        // Получаем эмбеддинги
        let mouse_emb = self.mouse_embedder.forward(mouse); // [b, embed_dim]
        let keys_emb = self.keys_embedder.forward(keys); // [b, embed_dim]
        // let timesteps_emb = self.timestep_embedder.forward(timesteps); // [b, embed_dim]

        // здесь для простоты просто суммируем
        // let embed = mouse_emb + keys_emb; // [b, embed_dim]
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

        // let embed_map = embed.unsqueeze_dim::<3>(2); // [embed_dim, 1, 1]

        // let x = self.unet.forward(images, timesteps, embed_map);

        let embed_map = embed.unsqueeze_dims::<4>(&[2, 3]); // [embed_dim, 1, 1]
        let embed_map = embed_map.expand([batch_size, embedding_dim, height, width]); // [embed_dim, height, width]

        let x = Tensor::cat(vec![images, embed_map], 1); // [b, channels + embed_dim, ...]

        let x = x.flatten(1, 3);
        let x = self.layer1.forward(x);
        let x = self.layer2.forward(x);
        let x = self.layer3.forward(x);
        let x = self.layer4.forward(x);
        let x = self.fc.forward(x);
        let x = self.tanh.forward(x);

        let x = x.reshape([batch_size, channels, height, width]);

        x
    }

    // pub fn latent_to_image(&self, latent: Tensor<B, 4>) -> Vec<Vec<u8>> {
    //     let [n_batch, _, _, _] = latent.dims();
    //     let image = self.autoencoder.decode_latent(latent * (1.0 / 0.18215));

    //     let num_elements_per_image = CHANNELS * HEIGHT * WIDTH;

    //     // correct size and scale and reorder to
    //     let image = (image + 1.0) / 2.0;
    //     let image = image
    //         .reshape([n_batch, CHANNELS, HEIGHT, WIDTH])
    //         .swap_dims(1, 2)
    //         .swap_dims(2, 3)
    //         .mul_scalar(255.0);

    //     let flattened: Vec<B::FloatElem> = image.into_data().to_vec().unwrap();

    //     (0..n_batch)
    //         .into_iter()
    //         .map(|b| {
    //             let start = b * num_elements_per_image;
    //             let end = start + num_elements_per_image;

    //             flattened[start..end]
    //                 .into_iter()
    //                 .map(|v| v.to_f64().min(255.0).max(0.0) as u8)
    //                 .collect()
    //         })
    //         .collect()
    // }
}
