use burn::{
    prelude::Backend,
    tensor::{Tensor, activation::softmax},
};

pub fn qkv_attention<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    mask: Option<Tensor<B, 2>>,
    n_head: usize,
) -> Tensor<B, 3> {
    let [n_batch, n_qctx, n_state] = q.dims();
    let [_, n_ctx, _] = k.dims();

    let scale = (n_state as f64 / n_head as f64).powf(-0.25);
    let n_hstate = n_state / n_head;

    let q = q
        .reshape([n_batch, n_qctx, n_head, n_hstate])
        .swap_dims(1, 2)
        * scale;
    let k = k
        .reshape([n_batch, n_ctx, n_head, n_hstate])
        .swap_dims(1, 2)
        .transpose()
        * scale;
    let v = v
        .reshape([n_batch, n_ctx, n_head, n_hstate])
        .swap_dims(1, 2);

    let qk = q.matmul(k);

    // apply mask
    let qk = if let Some(mask) = mask {
        qk + mask.slice([0..n_qctx, 0..n_ctx]).unsqueeze::<4>()
    } else {
        qk
    };

    // normalize value weightings
    let w = softmax(qk, 3);
    let o = w.matmul(v).swap_dims(1, 2).flatten(2, 3);

    return o;
}
