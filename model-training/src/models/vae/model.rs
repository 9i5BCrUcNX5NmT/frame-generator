use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        Relu,
    },
    prelude::*,
    tensor::Distribution,
};
use common::CHANNELS;

/// VAE Encoder: compresses images from pixel space to latent space.
/// 40x40x4 → 20x20x16 → 10x10x8 (4x spatial compression, 2x channel expansion)
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    conv1: Conv2d<B>,
    act1: Relu,
    conv2: Conv2d<B>,
    act2: Relu,
    conv3: Conv2d<B>,
    act3: Relu,
    // Output: mu and logvar (double channels for reparameterization)
    conv_mu: Conv2d<B>,
    conv_logvar: Conv2d<B>,
}

/// VAE Decoder: reconstructs images from latent space to pixel space.
/// 10x10x8 → 20x20x16 → 40x40x4
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    conv1: Conv2d<B>,
    act1: Relu,
    up1: ConvTranspose2d<B>,
    conv2: Conv2d<B>,
    act2: Relu,
    up2: ConvTranspose2d<B>,
    conv3: Conv2d<B>,
}

/// Variational Autoencoder for latent space diffusion.
///
/// Compresses 40x40x4 images to 10x10x8 latent representations.
/// Uses reparameterization trick for differentiable sampling.
#[derive(Module, Debug)]
pub struct VAE<B: Backend> {
    encoder: Encoder<B>,
    decoder: Decoder<B>,
}

#[derive(Config, Debug)]
pub struct VAEConfig {
    /// Latent space channels (default 8)
    #[config(default = "8")]
    pub latent_channels: usize,
    /// Hidden channels in encoder/decoder
    #[config(default = "16")]
    hidden_channels: usize,
}

impl VAEConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VAE<B> {
        let hc = self.hidden_channels;
        let lc = self.latent_channels;

        VAE {
            encoder: Encoder {
                // 40x40x4 -> 40x40x16
                conv1: Conv2dConfig::new([CHANNELS, hc], [3, 3])
                    .with_padding(nn::PaddingConfig2d::Same)
                    .init(device),
                act1: Relu,
                // 40x40x16 -> 20x20x16 (stride 2)
                conv2: Conv2dConfig::new([hc, hc], [3, 3])
                    .with_padding(nn::PaddingConfig2d::Explicit(1, 1))
                    .with_stride([2, 2])
                    .init(device),
                act2: Relu,
                // 20x20x16 -> 10x10x16 (stride 2)
                conv3: Conv2dConfig::new([hc, hc], [3, 3])
                    .with_padding(nn::PaddingConfig2d::Explicit(1, 1))
                    .with_stride([2, 2])
                    .init(device),
                act3: Relu,
                // 10x10x16 -> 10x10x8 (mu)
                conv_mu: Conv2dConfig::new([hc, lc], [1, 1]).init(device),
                // 10x10x16 -> 10x10x8 (logvar)
                conv_logvar: Conv2dConfig::new([hc, lc], [1, 1]).init(device),
            },
            decoder: Decoder {
                // 10x10x8 -> 10x10x16
                conv1: Conv2dConfig::new([lc, hc], [3, 3])
                    .with_padding(nn::PaddingConfig2d::Same)
                    .init(device),
                act1: Relu,
                // 10x10x16 -> 20x20x16
                up1: ConvTranspose2dConfig::new([hc, hc], [2, 2])
                    .with_stride([2, 2])
                    .init(device),
                // 20x20x16 -> 20x20x16
                conv2: Conv2dConfig::new([hc, hc], [3, 3])
                    .with_padding(nn::PaddingConfig2d::Same)
                    .init(device),
                act2: Relu,
                // 20x20x16 -> 40x40x16
                up2: ConvTranspose2dConfig::new([hc, hc], [2, 2])
                    .with_stride([2, 2])
                    .init(device),
                // 40x40x16 -> 40x40x4
                conv3: Conv2dConfig::new([hc, CHANNELS], [3, 3])
                    .with_padding(nn::PaddingConfig2d::Same)
                    .init(device),
            },
        }
    }
}

impl<B: Backend> VAE<B> {
    /// Encode image to latent distribution parameters (mu, logvar)
    pub fn encode(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let h = self.encoder.conv1.forward(x);
        let h = self.encoder.act1.forward(h);
        let h = self.encoder.conv2.forward(h);
        let h = self.encoder.act2.forward(h);
        let h = self.encoder.conv3.forward(h);
        let h = self.encoder.act3.forward(h);

        let mu = self.encoder.conv_mu.forward(h.clone());
        let logvar = self.encoder.conv_logvar.forward(h);

        (mu, logvar)
    }

    /// Reparameterization trick: z = mu + std * epsilon
    pub fn reparameterize(&self, mu: Tensor<B, 4>, logvar: Tensor<B, 4>) -> Tensor<B, 4> {
        let std = (logvar / 2.0).exp();
        let eps = std.random_like(Distribution::Normal(0.0, 1.0));
        mu + std * eps
    }

    /// Encode image to latent vector (sample from distribution)
    pub fn encode_sample(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let (mu, logvar) = self.encode(x);
        self.reparameterize(mu, logvar)
    }

    /// Decode latent vector to image
    pub fn decode(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.decoder.conv1.forward(z);
        let h = self.decoder.act1.forward(h);
        let h = self.decoder.up1.forward(h);
        let h = self.decoder.conv2.forward(h);
        let h = self.decoder.act2.forward(h);
        let h = self.decoder.up2.forward(h);
        self.decoder.conv3.forward(h)
    }

    /// Full forward pass: encode → reparameterize → decode
    /// Returns (reconstruction, mu, logvar) for VAE loss computation
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let (mu, logvar) = self.encode(x);
        let z = self.reparameterize(mu.clone(), logvar.clone());
        let reconstruction = self.decode(z);
        (reconstruction, mu, logvar)
    }
}
