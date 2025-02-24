use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
};
use nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv12: Conv2d<B>,
    activation12: Relu,

    conv21: Conv2d<B>,
    activation21: Relu,

    linear1: Linear<B>,
    linear2: Linear<B>,
    activation2: Relu,

    linear3: Linear<B>,
    linear4: Linear<B>,
    activation3: Relu,

    dropout: Dropout,
    pool: AdaptiveAvgPool2d,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "256")]
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
    #[config(default = "6")]
    embedding_dim: usize,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(108, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.embedding_dim).init(device),
            activation2: Relu,

            linear3: LinearConfig::new(400, self.hidden_size).init(device),
            linear4: LinearConfig::new(self.hidden_size, self.embedding_dim).init(device),
            activation3: Relu,

            conv12: Conv2dConfig::new([self.embedding_dim * 2 + 4, 4], [3, 3]).init(device),
            activation12: Relu,

            conv21: Conv2dConfig::new([self.embedding_dim * 2 + 4, 4], [3, 3]).init(device),
            activation21: Relu,

            dropout: DropoutConfig::new(self.dropout).init(),
            pool: AdaptiveAvgPool2dConfig::new([200, 200]).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, color, height, width]
    ///   - Keys [batch_size, type, key_number]
    ///   - Output [batch_size, color, height, width]
    pub fn forward(
        &self,
        images: Tensor<B, 4>,
        keys: Tensor<B, 2>,
        mouse: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let [batch_size, channels, height, width] = images.dims();

        // // Create a channel at the second dimension.
        // let x = images.reshape([batch_size, 1, height, width]);

        // let x = self.conv11.forward(x);
        // let x = self.activation11.forward(x);

        // let y = inputs.reshape([batch_size, colors, height, width]);

        // let y = inputs.reshape([batch_size, 1, height, width]);
        // println!("{:?}", inputs.dims());

        // Обработка ключей
        let k = self.linear1.forward(keys); // [n, 108] -> [n, hidden_size]
        let k = self.activation2.forward(k);
        let k = self.linear2.forward(k); // [n, hidden_size] -> [n, embed_dim]

        // Обработка мыши
        let m: Tensor<B, 2> = mouse.flatten(1, 2); // [n, 2, 200] -> [n, 400]
        let m = self.linear3.forward(m); // [n, 400] -> [n, hidden_size]
        let m = self.activation3.forward(m);
        let m = self.linear4.forward(m); // [n, hidden_size] -> [n, embed_dim]

        // Совмещение эмбеддингов
        let embed_map = Tensor::cat(vec![k, m], 1); // [n, embed_dim * 2]

        let [_, embedding_dim] = embed_map.dims();

        let embed_map = embed_map.unsqueeze_dims::<4>(&[2, 3]); // [n, embed_dim * 2] -> [n, embed_dim * 2, 1, 1]
        let embed_map = embed_map.expand([batch_size, embedding_dim, height, width]); // [n, embed_dim * 2, 1, 1] -> [n, embed_dim * 2, 200, 200]

        // Обработка изображений
        let x = Tensor::cat(vec![images, embed_map.clone()], 1); // [n, embed_dim * 2 + 4, 200, 200]

        // encoder
        let x = self.conv12.forward(x);
        let x = self.activation12.forward(x);

        // decoder
        // let x = Tensor::cat(vec![x, embed_map], 1);
        let x = self.conv21.forward(x);
        let x = self.activation21.forward(x);

        // output
        let x = self.pool.forward(x);
        let x = x.reshape([batch_size, channels, height, width]);

        // let ;

        // let y = self.linear01.forward(keys);
        // let y = self.activation01.forward(y);
        // let y = self.linear11.forward(y);
        // let y = self.activation11.forward(y);

        // let x = self.conv31.forward(x);
        // let x = self.pool.forward(x);
        // let x = self.activation31.forward(x);

        // let y = y.reshape([batch_size, height, width]);

        // let z = x + y;

        // let z = self.conv31.forward(z);
        // let z = self.activation31.forward(z);

        // let r = y.reshape([batch_size, colors, height, width]);

        x
    }
}
