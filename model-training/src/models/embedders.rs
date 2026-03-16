use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::{Tensor, TensorData, activation::gelu},
};
use common::MOUSE_VECTOR_LENGTH;

#[derive(Module, Debug)]
pub struct MouseEmbedder<B: Backend> {
    linear1: Linear<B>,
    activation: Relu,
    linear2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct MouseEmbedderConfig {
    embed_dim: usize,
    hidden_dim: usize,
}

impl MouseEmbedderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MouseEmbedder<B> {
        MouseEmbedder {
            linear1: LinearConfig::new(2 * MOUSE_VECTOR_LENGTH, self.hidden_dim).init(device),
            activation: Relu,
            linear2: LinearConfig::new(self.hidden_dim, self.embed_dim).init(device),
        }
    }
}

impl<B: Backend> MouseEmbedder<B> {
    /// Normal method added to a struct.
    pub fn forward(&self, mouse: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = mouse.flatten(1, 2); // [n, 2, MOUSE_VECTOR_LENGTH] -> [n, MOUSE_VECTOR_LENGTH * 2]
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);

        x
    }
}

#[derive(Module, Debug)]
pub struct KeyboardEmbedder<B: Backend> {
    linear1: Linear<B>,
    activation: Relu,
    linear2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct KeyboardEmbedderConfig {
    embed_dim: usize,
    hidden_dim: usize,
}

impl KeyboardEmbedderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> KeyboardEmbedder<B> {
        KeyboardEmbedder {
            linear1: LinearConfig::new(108, self.hidden_dim).init(device),
            activation: Relu,
            linear2: LinearConfig::new(self.hidden_dim, self.embed_dim).init(device),
        }
    }
}

impl<B: Backend> KeyboardEmbedder<B> {
    /// Normal method added to a struct.
    pub fn forward(&self, keys: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(keys);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);

        x
    }
}

/// Timestep embedder using sinusoidal positional encoding
#[derive(Module, Debug)]
pub struct TimestepEmbedder<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct TimestepEmbedderConfig {
    pub embed_dim: usize,
}

impl TimestepEmbedderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TimestepEmbedder<B> {
        TimestepEmbedder {
            linear1: LinearConfig::new(self.embed_dim, self.embed_dim * 4).init(device),
            linear2: LinearConfig::new(self.embed_dim * 4, self.embed_dim).init(device),
        }
    }
}

impl<B: Backend> TimestepEmbedder<B> {
    /// Forward pass with sinusoidal embeddings
    pub fn forward(&self, timesteps: Tensor<B, 1>) -> Tensor<B, 2> {
        let device = timesteps.device();
        let [batch_size] = timesteps.dims();

        // Create sinusoidal embeddings - simpler approach using expand
        // We'll use a different approach: create [batch_size, embed_dim] tensor
        let hidden_dim = self.linear1.weight.shape()[1];

        // Get timesteps as Vec<f32>
        let t_values: Vec<f32> = timesteps.to_data().to_vec::<f32>().unwrap();

        // Create [batch_size, hidden_dim] tensor manually
        let mut data: Vec<f32> = Vec::with_capacity(batch_size * hidden_dim);

        for t in t_values.iter() {
            for j in 0..hidden_dim {
                let freq = (j as f32 / 2.0_f32).exp() * std::f32::consts::PI;
                let value = (t * freq).sin(); // Sinusoidal encoding
                data.push(value);
            }
        }

        let tensor_data = TensorData::new(data, [batch_size, hidden_dim]);
        let x = Tensor::<B, 2>::from_data(tensor_data, &device);

        // Pass through MLPs
        let x = self.linear1.forward(x);
        let x = gelu(x);
        let x = self.linear2.forward(x);

        x
    }
}
