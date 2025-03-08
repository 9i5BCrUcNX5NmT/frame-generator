use burn::{
    prelude::Backend,
    tensor::{Shape, Tensor},
};

fn add_dims<B: Backend, const D: usize>(input: Tensor<B, D>, n: usize) -> Tensor<B, D> {}
