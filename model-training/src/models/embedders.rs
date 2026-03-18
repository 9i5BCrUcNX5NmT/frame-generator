use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::{Tensor, TensorData},
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
            // LinearConfig::new(input_features, output_features)
            // input: 2 * MOUSE_VECTOR_LENGTH = 400, output: hidden_dim = 100
            linear1: LinearConfig::new(2 * MOUSE_VECTOR_LENGTH, self.hidden_dim).init(device),
            activation: Relu,
            // input: hidden_dim = 100, output: embed_dim = 100
            linear2: LinearConfig::new(self.hidden_dim, self.embed_dim).init(device),
        }
    }
}

impl<B: Backend> MouseEmbedder<B> {
    pub fn forward(&self, mouse: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = mouse.flatten(1, 2); // [n, 2, 200] -> [n, 400]
        let x = self.linear1.forward(x); // [n, 400] -> [n, 100]
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x); // [n, 100] -> [n, 100]
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
            // LinearConfig::new(input_features, output_features)
            // input: 108, output: hidden_dim = 100
            linear1: LinearConfig::new(108, self.hidden_dim).init(device),
            activation: Relu,
            // input: hidden_dim = 100, output: embed_dim = 100
            linear2: LinearConfig::new(self.hidden_dim, self.embed_dim).init(device),
        }
    }
}

impl<B: Backend> KeyboardEmbedder<B> {
    pub fn forward(&self, keys: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(keys); // [n, 108] -> [n, 100]
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x); // [n, 100] -> [n, 100]
        x
    }
}

/// Timestep embedder using sinusoidal positional encoding
#[derive(Module, Debug)]
pub struct TimestepEmbedder<B: Backend> {
    // LinearConfig::new(input=embed_dim*4, output=embed_dim)
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct TimestepEmbedderConfig {
    pub embed_dim: usize,
}

impl TimestepEmbedderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TimestepEmbedder<B> {
        let hidden_dim = self.embed_dim * 4;
        TimestepEmbedder {
            // LinearConfig::new(input_features, output_features)
            // input: hidden_dim = 400 (sinusoidal), output: embed_dim = 100
            linear: LinearConfig::new(hidden_dim, self.embed_dim).init(device),
        }
    }
}

impl<B: Backend> TimestepEmbedder<B> {
    pub fn forward(&self, timesteps: Tensor<B, 1>) -> Tensor<B, 2> {
        let device = timesteps.device();
        let [batch_size] = timesteps.dims();

        // Derive dimensions from linear weight shape
        // weight shape: [input=400, output=100]
        let hidden_dim = self.linear.weight.shape().dims[0]; // input = 400

        let t_values: Vec<f32> = timesteps.to_data().to_vec::<f32>().unwrap();

        // Create [batch_size, hidden_dim] sinusoidal encoding
        let mut data: Vec<f32> = Vec::with_capacity(batch_size * hidden_dim);
        for t in t_values.iter() {
            for j in 0..hidden_dim {
                let freq = (j as f32 / 2.0_f32).exp() * std::f32::consts::PI;
                let value = (t * freq).sin();
                data.push(value);
            }
        }

        let tensor_data = TensorData::new(data, [batch_size, hidden_dim]);
        let x = Tensor::<B, 2>::from_data(tensor_data, &device);

        // Linear forward: [batch, 400] -> [batch, 100]
        self.linear.forward(x)
    }
}
