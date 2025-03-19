use burn::{
    module::Module,
    nn::{self, BatchNorm},
    prelude::Backend,
    tensor::Tensor,
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
