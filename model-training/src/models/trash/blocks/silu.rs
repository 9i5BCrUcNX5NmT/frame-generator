use burn::{
    module::Module,
    prelude::Backend,
    tensor::{Tensor, activation::sigmoid},
};

#[derive(Module, Clone, Debug)]
pub struct SILU {}

impl SILU {
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        x.clone() * sigmoid(x)
    }
}
