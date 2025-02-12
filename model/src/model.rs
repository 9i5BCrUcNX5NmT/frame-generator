use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Dropout, DropoutConfig, Relu,
    },
    prelude::*,
};
use nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    // conv11: Conv2d<B>,
    // activation11: Relu,

    // conv12: Conv2d<B>,
    // activation12: Relu,

    // conv13: Conv2d<B>,
    // activation13: Relu,
    conv21: Conv2d<B>,
    activation21: Relu,

    // conv22: Conv2d<B>,
    // activation22: Relu,

    // conv23: Conv2d<B>,
    // activation23: Relu,
    dropout: Dropout,

    pool: AdaptiveAvgPool2d,

    // add
    conv31: Conv2d<B>,
    activation31: Relu,
    // conv32: Conv2d<B>,
    // activation32: Relu,

    // conv33: Conv2d<B>,
    // activation33: Relu,

    // conv34: Conv2d<B>,
    // activation34: Relu,

    // conv35: Conv2d<B>,
    // reshape 6
    // transpose
    // reshape 4
    // concat
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            // conv11: Conv2dConfig::new([16, 3], [3, 3]).init(device),
            // activation11: Relu,
            // conv12: Conv2dConfig::new([32, 16], [3, 3]).init(device),
            // activation12: Relu,
            // conv13: Conv2dConfig::new([64, 32], [3, 3]).init(device),
            // activation13: Relu,
            conv21: Conv2dConfig::new([1, 32], [3, 3]).init(device),
            activation21: Relu,
            // conv22: Conv2dConfig::new([64, 64], [1, 1]).init(device),
            // activation22: Relu,
            // conv23: Conv2dConfig::new([64, 64], [1, 1]).init(device),
            // activation23: Relu,
            dropout: DropoutConfig::new(self.dropout).init(),
            conv31: Conv2dConfig::new([32, 4], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([200, 200]).init(),
            activation31: Relu,
            // conv32: Conv2dConfig::new([64, 64], [3, 3]).init(device),
            // activation32: Relu,
            // conv33: Conv2dConfig::new([64, 64], [3, 3]).init(device),
            // activation33: Relu,
            // conv34: Conv2dConfig::new([64, 64], [3, 3]).init(device),
            // activation34: Relu,
            // conv35: Conv2dConfig::new([192, 64], [3, 3]).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, color, height, width]
    ///   - Output [batch_size, color, height, width]
    pub fn forward(&self, inputs: Tensor<B, 4>) -> Tensor<B, 4> {
        // let [batch_size, color, height, width] = inputs.dims();

        // // Create a channel at the second dimension.
        // let x = images.reshape([batch_size, 1, height, width]);

        // let x = self.conv11.forward(x);
        // let x = self.activation11.forward(x);

        // let y = inputs.reshape([batch_size, colors, height, width]);

        // let y = inputs.reshape([batch_size, 1, height, width]);
        // println!("{:?}", inputs.dims());

        let x = self.conv21.forward(inputs);
        let x = self.activation21.forward(x);
        let x = self.dropout.forward(x);
        let x = self.conv31.forward(x);
        let x = self.pool.forward(x);
        let x = self.activation31.forward(x);

        // let y = y.reshape([batch_size, height, width]);

        // let z = x + y;

        // let z = self.conv31.forward(z);
        // let z = self.activation31.forward(z);

        // let r = y.reshape([batch_size, colors, height, width]);

        x
    }
}
