use burn::{
    nn::{
        LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
    tensor::{Distribution, cast::ToElement},
};

use common::*;

use crate::models::embedders::{
    KeyboardEmbedder, KeyboardEmbedderConfig, MouseEmbedder, MouseEmbedderConfig,
};

use super::blocks::{
    autoencoder::{Autoencoder, AutoencoderConfig},
    my::LayerBlock,
};

#[derive(Module, Debug)]
pub struct Diffusion<B: Backend> {
    mouse_embedder: MouseEmbedder<B>,
    keys_embedder: KeyboardEmbedder<B>,

    // conv1: Conv2d<B>,
    // act1: Relu,
    // conv2: ConvTranspose2d<B>,
    // act2: Relu,

    // out: AdaptiveAvgPool2d,
    layer1: LayerBlock<B>,
    layer2: LayerBlock<B>,
    layer3: LayerBlock<B>,
    layer4: LayerBlock<B>,
    fc: nn::Linear<B>,
    tanh: nn::Tanh,
    // autoencoder: Autoencoder<B>,
}

#[derive(Config, Debug)]
pub struct DiffusionConfig {
    // #[config(default = "3")]
    // in_channels: usize,
    // #[config(default = "64")]
    // out_channels: usize,
    input: usize,
    output: usize,
    #[config(default = "16")]
    embed_dim: usize,
}

impl DiffusionConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Diffusion<B> {
        let layer1 = LayerBlock::new((CHANNELS + self.embed_dim) * WIDTH * HEIGHT, 128, device);
        let layer2 = LayerBlock::new(128, 256, device);
        let layer3 = LayerBlock::new(256, 512, device);
        let layer4 = LayerBlock::new(512, 1024, device);
        let fc = nn::LinearConfig::new(1024, self.output)
            .with_bias(true)
            .init(device);

        Diffusion {
            mouse_embedder: MouseEmbedderConfig::new(self.embed_dim, self.embed_dim).init(device),
            keys_embedder: KeyboardEmbedderConfig::new(self.embed_dim, self.embed_dim).init(device),
            layer1,
            layer2,
            layer3,
            layer4,
            fc,
            tanh: nn::Tanh,
            // conv1: Conv2dConfig::new(
            //     [self.in_channels + self.embed_dim, self.out_channels],
            //     [3, 3],
            // )
            // .init(device),
            // act1: Relu,
            // conv2: ConvTranspose2dConfig::new([self.out_channels, self.in_channels], [3, 3])
            //     .init(device),
            // act2: Relu,

            // out: AdaptiveAvgPool2dConfig::new([HEIGHT, WIDTH]).init(),
            // autoencoder: AutoencoderConfig::new().init(device),
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
        let images = self.add_noise(images, 0.3);

        // Получаем эмбеддинги
        let mouse_emb = self.mouse_embedder.forward(mouse); // [b, embed_dim]
        let keys_emb = self.keys_embedder.forward(keys); // [b, embed_dim]

        // здесь для простоты просто суммируем
        let embed = mouse_emb + keys_emb; // [b, embed_dim]

        let [batch_size, channels, height, width] = images.dims();
        let [_, embedding_dim] = embed.dims();

        let embed_map = embed.unsqueeze_dims::<4>(&[2, 3]); // [embed_dim, 1, 1]
        let embed_map = embed_map.expand([batch_size, embedding_dim, height, width]); // [embed_dim, height, width]

        // let noise = self.latent_to_image(embed_map);
        // let noise = TensorData::from_bytes(
        //     noise.into_iter().flat_map(|x| x).collect(),
        //     [batch_size, channels, height, width],
        //     burn::tensor::DType::F32,
        // );
        // let noise = Tensor::from_data(noise, &self.devices()[0]);

        // let noise = self.autoencoder.decode_latent(embed_map);
        let x = Tensor::cat(vec![images, embed_map], 1); // [b, channels + embed_dim, ...]

        let x = x.flatten(1, 3);
        let x = self.layer1.forward(x);
        let x = self.layer2.forward(x);
        let x = self.layer3.forward(x);
        let x = self.layer4.forward(x);
        let x = self.fc.forward(x);
        let x = self.tanh.forward(x);

        // let x = self.conv1.forward(embed_map);
        // let x = self.act1.forward(x);
        // let x = self.conv2.forward(x);
        // let x = self.act2.forward(x);

        // let x = self.out.forward(x);

        let x = x.reshape([batch_size, channels, height, width]); // .add(noise)

        x
    }

    fn add_noise(&self, input: Tensor<B, 4>, noise_level: f32) -> Tensor<B, 4> {
        // Добавление шума к входным данным
        let noise = input.random_like(burn::tensor::Distribution::Default);
        input * (1.0 - noise_level) + noise * (noise_level)
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
