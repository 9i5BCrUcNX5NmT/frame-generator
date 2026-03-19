use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        pool::MaxPool2d,
        pool::MaxPool2dConfig,
        Relu,
    },
    prelude::*,
};
use common::{CHANNELS, HEIGHT, WIDTH};

use crate::models::{
    attention::{CrossAttention, CrossAttentionConfig},
    embedders::{
        KeyboardEmbedder, KeyboardEmbedderConfig, MouseEmbedder, MouseEmbedderConfig,
        TimestepEmbedder, TimestepEmbedderConfig,
    },
    noise_schedule::CosineNoiseSchedule,
    vae::{VAEConfig, VAE},
};

/// Latent-space U-Net that operates on VAE latent representations.
///
/// Input: [B, latent_channels, H/4, W/4] (10x10x8 for 40x40 images)
/// Conditioning via cross-attention at bottleneck.
#[derive(Module, Debug)]
pub struct LatentUNet<B: Backend> {
    // Encoder
    conv1: Conv2d<B>,
    act1: Relu,
    conv2: Conv2d<B>,
    act2: Relu,
    down1: MaxPool2d,

    // Bottleneck
    conv3: Conv2d<B>,
    act3: Relu,
    conv4: Conv2d<B>,
    act4: Relu,

    // Cross-attention at bottleneck
    cross_attn: CrossAttention<B>,

    // Decoder
    up1: ConvTranspose2d<B>,
    conv5: Conv2d<B>,
    act5: Relu,
    conv6: Conv2d<B>,
    act6: Relu,
}

#[derive(Config, Debug)]
pub struct LatentUNetConfig {
    #[config(default = "8")]
    latent_channels: usize,
    #[config(default = "32")]
    hidden_dim: usize,
    /// Combined dimension of all condition embeddings (mouse + keys + timestep)
    #[config(default = "300")]
    condition_dim: usize,
}

impl LatentUNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LatentUNet<B> {
        let lc = self.latent_channels;
        let hd = self.hidden_dim;

        LatentUNet {
            // Encoder: 10x10x8 -> 10x10x32
            conv1: Conv2dConfig::new([lc, hd], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act1: Relu,
            conv2: Conv2dConfig::new([hd, hd], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act2: Relu,
            // 10x10x32 -> 5x5x32
            down1: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),

            // Bottleneck: 5x5x32 -> 5x5x64
            conv3: Conv2dConfig::new([hd, hd * 2], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act3: Relu,
            conv4: Conv2dConfig::new([hd * 2, hd * 2], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act4: Relu,

            // Cross-attention: condition on action embeddings
            cross_attn: CrossAttentionConfig::new(hd * 2, self.condition_dim).init(device),

            // Decoder: 5x5x64 -> 10x10x32
            up1: ConvTranspose2dConfig::new([hd * 2, hd], [2, 2])
                .with_stride([2, 2])
                .init(device),
            // Skip connection: 32 + 32 = 64 input channels
            conv5: Conv2dConfig::new([hd * 2, hd], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act5: Relu,
            conv6: Conv2dConfig::new([hd, lc], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            act6: Relu,
        }
    }
}

impl<B: Backend> LatentUNet<B> {
    /// Forward pass through latent U-Net.
    ///
    /// - `x`: noisy latent [B, latent_ch, H', W']
    /// - `condition`: combined action+timestep embedding [B, condition_dim]
    pub fn forward(&self, x: Tensor<B, 4>, condition: Tensor<B, 2>) -> Tensor<B, 4> {
        // Encoder
        let h = self.conv1.forward(x);
        let h = self.act1.forward(h);
        let h = self.conv2.forward(h);
        let h = self.act2.forward(h);
        let skip = h.clone();

        let h = self.down1.forward(h);

        // Bottleneck
        let h = self.conv3.forward(h);
        let h = self.act3.forward(h);
        let h = self.conv4.forward(h);
        let h = self.act4.forward(h);

        // Cross-attention conditioning
        let h = self.cross_attn.forward(h, condition);

        // Decoder
        let h = self.up1.forward(h);

        // Handle potential size mismatch with skip connection
        let [_b, _c, h_up, w_up] = h.dims();
        let [_b2, _c2, h_skip, w_skip] = skip.dims();
        let h = if h_up != h_skip || w_up != w_skip {
            let dy = h_skip - h_up;
            let dx = w_skip - w_up;
            h.pad((dx / 2, dx - dx / 2, dy / 2, dy - dy / 2), 0.0)
        } else {
            h
        };

        // Skip connection
        let h = Tensor::cat(vec![h, skip], 1);

        let h = self.conv5.forward(h);
        let h = self.act5.forward(h);
        let h = self.conv6.forward(h);
        self.act6.forward(h)
    }
}

/// Full conditional diffusion model (Level 2).
///
/// Architecture:
/// 1. VAE encodes images to latent space (40x40x4 → 10x10x8)
/// 2. Action embedders produce condition vectors (mouse + keys + timestep)
/// 3. Latent U-Net predicts noise in latent space with cross-attention conditioning
/// 4. VAE decodes back to pixel space for inference
#[derive(Module, Debug)]
pub struct ModelV2<B: Backend> {
    pub vae: VAE<B>,
    mouse_embedder: MouseEmbedder<B>,
    keys_embedder: KeyboardEmbedder<B>,
    timestep_embedder: TimestepEmbedder<B>,
    latent_unet: LatentUNet<B>,
}

#[derive(Config, Debug)]
pub struct ModelV2Config {
    #[config(default = "100")]
    pub embed_dim: usize,
    #[config(default = "8")]
    pub latent_channels: usize,
    #[config(default = "32")]
    pub unet_hidden_dim: usize,
    #[config(default = "1000")]
    pub num_timesteps: usize,
}

impl ModelV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ModelV2<B> {
        let condition_dim = self.embed_dim * 3; // mouse + keys + timestep

        ModelV2 {
            vae: VAEConfig::new()
                .with_latent_channels(self.latent_channels)
                .init(device),
            mouse_embedder: MouseEmbedderConfig::new(self.embed_dim, self.embed_dim).init(device),
            keys_embedder: KeyboardEmbedderConfig::new(self.embed_dim, self.embed_dim).init(device),
            timestep_embedder: TimestepEmbedderConfig::new(self.embed_dim).init(device),
            latent_unet: LatentUNetConfig::new()
                .with_latent_channels(self.latent_channels)
                .with_hidden_dim(self.unet_hidden_dim)
                .with_condition_dim(condition_dim)
                .init(device),
        }
    }

    /// Create the noise schedule for this model configuration
    pub fn noise_schedule(&self) -> CosineNoiseSchedule {
        CosineNoiseSchedule::new(self.num_timesteps)
    }
}

impl<B: Backend> ModelV2<B> {
    /// Compute combined condition embedding from actions and timestep.
    ///
    /// Returns [B, embed_dim * 3] tensor.
    fn compute_condition(
        &self,
        keys: Tensor<B, 2>,
        mouse: Tensor<B, 3>,
        timestep: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let mouse_emb = self.mouse_embedder.forward(mouse); // [B, embed_dim]
        let keys_emb = self.keys_embedder.forward(keys); // [B, embed_dim]
        let time_emb = self.timestep_embedder.forward(timestep); // [B, embed_dim]

        // Concatenate all embeddings: [B, embed_dim * 3]
        Tensor::cat(vec![mouse_emb, keys_emb, time_emb], 1)
    }

    /// Training forward: predict noise in latent space.
    ///
    /// 1. Encode target to latent space
    /// 2. Add noise at given timestep
    /// 3. Predict noise with conditioned U-Net
    ///
    /// Returns (predicted_noise, mu, logvar) for loss computation.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_train(
        &self,
        targets: Tensor<B, 4>,
        keys: Tensor<B, 2>,
        mouse: Tensor<B, 3>,
        timestep_indices: Tensor<B, 1>,
        noise: Tensor<B, 4>,
        alpha: f32,
        sigma: f32,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        // 1. Encode to latent space
        let (mu, logvar) = self.vae.encode(targets);
        let z0 = self.vae.reparameterize(mu.clone(), logvar.clone());

        // 2. Add noise: z_t = alpha * z_0 + sigma * noise
        let z_t = z0 * alpha + noise.clone() * sigma;

        // 3. Compute condition embedding
        let condition = self.compute_condition(keys, mouse, timestep_indices);

        // 4. Predict noise
        let predicted_noise = self.latent_unet.forward(z_t, condition);

        (predicted_noise, mu, logvar)
    }

    /// Inference: generate next frame using DDIM sampling in latent space.
    ///
    /// Starts from random noise in latent space and iteratively denoises.
    pub fn sample(
        &self,
        keys: Tensor<B, 2>,
        mouse: Tensor<B, 3>,
        schedule: &CosineNoiseSchedule,
        num_steps: usize,
    ) -> Tensor<B, 4> {
        let device = keys.device();
        let batch_size = keys.dims()[0];
        let latent_channels = self
            .vae
            .encode(Tensor::zeros([1, CHANNELS, HEIGHT, WIDTH], &device))
            .0
            .dims()[1];
        let latent_h = HEIGHT / 4;
        let latent_w = WIDTH / 4;

        // Start from pure noise in latent space
        let mut z_t = Tensor::random(
            [batch_size, latent_channels, latent_h, latent_w],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        // DDIM sampling loop
        let step_size = schedule.num_timesteps / num_steps;
        for i in (0..num_steps).rev() {
            let t = i * step_size;
            let timestep: Tensor<B, 1> = Tensor::from_data(
                burn::tensor::TensorData::new(vec![t as f32 / schedule.num_timesteps as f32], [1]),
                &device,
            );

            // Expand timestep for batch
            let timestep: Tensor<B, 1> = timestep.expand([batch_size]);

            let condition = self.compute_condition(keys.clone(), mouse.clone(), timestep);

            let predicted_noise = self.latent_unet.forward(z_t.clone(), condition);

            // DDIM step
            z_t = schedule.ddim_step(z_t, predicted_noise, t);
        }

        // Decode from latent space to pixel space
        self.vae.decode(z_t)
    }
}
