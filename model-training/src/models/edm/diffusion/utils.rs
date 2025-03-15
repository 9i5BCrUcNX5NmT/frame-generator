use std::collections::HashMap;

use burn::{
    prelude::Backend,
    tensor::{Bool, Tensor},
};

pub struct EdmBatch<B: Backend> {
    pub obs: Tensor<B, 5>,
    pub act: Tensor<B, 2>,
    pub rew: Tensor<B, 5>,
    pub end: Tensor<B, 5>,
    pub trunc: Tensor<B, 5>,
    pub mask_padding: Tensor<B, 1, Bool>,
    pub info: Vec<HashMap<String, String>>, // ??
    pub segment_ids: Vec<usize>,
}
