use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::activation::softmax,
};

/// Cross-attention block for conditioning spatial features on action embeddings.
///
/// Q is projected from spatial features, K and V from context (action embeddings).
/// Supports multi-head attention with configurable number of heads.
#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    num_heads: usize,
}

#[derive(Config, Debug)]
pub struct CrossAttentionConfig {
    /// Dimension of spatial features (channels)
    spatial_dim: usize,
    /// Dimension of context (action embeddings)
    context_dim: usize,
    /// Number of attention heads
    #[config(default = "4")]
    num_heads: usize,
}

impl CrossAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CrossAttention<B> {
        CrossAttention {
            q_proj: LinearConfig::new(self.spatial_dim, self.spatial_dim).init(device),
            k_proj: LinearConfig::new(self.context_dim, self.spatial_dim).init(device),
            v_proj: LinearConfig::new(self.context_dim, self.spatial_dim).init(device),
            out_proj: LinearConfig::new(self.spatial_dim, self.spatial_dim).init(device),
            num_heads: self.num_heads,
        }
    }
}

impl<B: Backend> CrossAttention<B> {
    /// Forward pass: cross-attend spatial features to context.
    ///
    /// - `x`: spatial features [B, C, H, W]
    /// - `context`: action/condition embeddings [B, D]
    ///
    /// Returns: conditioned spatial features [B, C, H, W]
    pub fn forward(&self, x: Tensor<B, 4>, context: Tensor<B, 2>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        let seq_len = height * width;
        let head_dim = channels / self.num_heads;

        // Flatten spatial dims: [B, C, H*W] -> [B, H*W, C]
        let x_flat = x.clone().reshape([batch, channels, seq_len]);
        let x_flat = x_flat.swap_dims(1, 2); // [B, H*W, C]

        // Project Q from spatial features
        let q = self.q_proj.forward(x_flat); // [B, H*W, C]

        // Expand context to sequence: [B, D] -> [B, 1, D]
        let ctx: Tensor<B, 3> = context.unsqueeze_dim(1); // [B, 1, D]

        // Project K, V from context
        let k = self.k_proj.forward(ctx.clone()); // [B, 1, C]
        let v = self.v_proj.forward(ctx); // [B, 1, C]

        // Reshape for multi-head: [B, seq, C] -> [B, heads, seq, head_dim]
        let q = q
            .reshape([batch, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2); // [B, heads, H*W, head_dim]
        let k = k
            .reshape([batch, 1, self.num_heads, head_dim])
            .swap_dims(1, 2); // [B, heads, 1, head_dim]
        let v = v
            .reshape([batch, 1, self.num_heads, head_dim])
            .swap_dims(1, 2); // [B, heads, 1, head_dim]

        // Scaled dot-product attention
        let scale = (head_dim as f64).powf(-0.5);
        let attn = q.matmul(k.swap_dims(2, 3)) * scale; // [B, heads, H*W, 1]
        let attn = softmax(attn, 3);
        let out = attn.matmul(v); // [B, heads, H*W, head_dim]

        // Reshape back: [B, heads, H*W, head_dim] -> [B, H*W, C]
        let out = out.swap_dims(1, 2).reshape([batch, seq_len, channels]);

        // Output projection
        let out = self.out_proj.forward(out); // [B, H*W, C]

        // Reshape to spatial: [B, H*W, C] -> [B, C, H, W]
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, channels, height, width]);

        // Residual connection
        x + out
    }
}
