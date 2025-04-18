// from https://github.com/phenylshima/segmentation-models-burn/tree/main/src/encoder

pub mod resnet;

use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};

pub trait Encoder: Module<Self::Backend> {
    type Backend: Backend;
    fn forward(
        &self,
        x: Tensor<Self::Backend, 4>,
        feature_idxs: Vec<usize>,
    ) -> Vec<Tensor<Self::Backend, 4>>;
}

pub trait EncoderConfig<B: Backend> {
    type Encoder: Encoder;
    fn init(&self, device: &B::Device) -> Self::Encoder;
    fn out_channels(&self) -> Vec<usize>;
    fn feature_idxs(&self) -> Vec<usize>;
}
