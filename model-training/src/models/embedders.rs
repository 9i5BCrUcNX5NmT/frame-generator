use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::Tensor,
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

// #[derive(Module, Debug)]
// pub struct TimestempEmbedder<B: Backend> {
//     linear1: Linear<B>,
//     activation: Relu,
//     linear2: Linear<B>,
// }

// #[derive(Config, Debug)]
// pub struct TimestempEmbedderConfig {
//     input_dim: usize,
//     output_dim: usize,
// }

// impl TimestempEmbedderConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> TimestempEmbedder<B> {
//         TimestempEmbedder {
//             linear1: LinearConfig::new(self.input_dim, self.input_dim * 2).init(device),
//             activation: Relu,
//             linear2: LinearConfig::new(self.input_dim * 2, self.output_dim).init(device),
//         }
//     }
// }

// impl<B: Backend> TimestempEmbedder<B> {
//     /// Normal method added to a struct.
//     pub fn forward(&self, timesteps: Tensor<B, 1>) -> Tensor<B, 2> {
//         let x = timesteps.reshape([-1, 1]).expand([-1, 16]); // TODO: как заменить 16?
//         let x = self.linear1.forward(x);
//         let x = self.activation.forward(x);
//         let x = self.linear2.forward(x);

//         x
//     }
// }
