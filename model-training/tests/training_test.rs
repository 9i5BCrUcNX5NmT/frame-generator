//! Tests for model forward pass debugging and training
//! Run: cargo test -p model-training --test training_test -- --nocapture
//! Run with CUDA: cargo test -p model-training --features cuda --test training_test -- --nocapture

use burn::backend::NdArray;
use burn::nn::LinearConfig;
use burn::tensor::{Tensor, TensorData};
use common::*;

/// Verify LinearConfig::new(input, output) API
#[test]
fn test_linear_config_api() {
    type B = NdArray<f32>;
    let device = Default::default();

    // LinearConfig::new(10, 20) => weight [10, 20], forward([b, 10]) => [b, 20]
    let linear = LinearConfig::new(10, 20).init::<B>(&device);
    assert_eq!(linear.weight.shape().dims, [10, 20]);

    let x = Tensor::<B, 2>::from_data(TensorData::new(vec![1.0f32; 2 * 10], [2, 10]), &device);
    let out = linear.forward(x);
    assert_eq!(out.dims(), [2, 20]);
}

/// Test MouseEmbedder: [batch, 2, 200] -> [batch, embed_dim]
#[test]
fn test_mouse_embedder() {
    use model_training::models::embedders::MouseEmbedderConfig;
    type B = NdArray<f32>;
    let device = Default::default();

    let embedder = MouseEmbedderConfig::new(100, 100).init::<B>(&device);

    let mouse = Tensor::<B, 3>::from_data(
        TensorData::new(
            vec![0.0f32; 4 * 2 * MOUSE_VECTOR_LENGTH],
            [4, 2, MOUSE_VECTOR_LENGTH],
        ),
        &device,
    );

    let output = embedder.forward(mouse);
    assert_eq!(
        output.dims(),
        [4, 100],
        "MouseEmbedder output should be [batch, embed_dim]"
    );
}

/// Test KeyboardEmbedder: [batch, 108] -> [batch, embed_dim]
#[test]
fn test_keyboard_embedder() {
    use model_training::models::embedders::KeyboardEmbedderConfig;
    type B = NdArray<f32>;
    let device = Default::default();

    let embedder = KeyboardEmbedderConfig::new(100, 100).init::<B>(&device);

    let keys = Tensor::<B, 2>::from_data(TensorData::new(vec![0.0f32; 4 * 108], [4, 108]), &device);

    let output = embedder.forward(keys);
    assert_eq!(
        output.dims(),
        [4, 100],
        "KeyboardEmbedder output should be [batch, embed_dim]"
    );
}

/// Test TimestepEmbedder: [batch] -> [batch, embed_dim]
#[test]
fn test_timestep_embedder() {
    use model_training::models::embedders::TimestepEmbedderConfig;
    type B = NdArray<f32>;
    let device = Default::default();

    let embedder = TimestepEmbedderConfig::new(100).init::<B>(&device);

    let timestep = Tensor::<B, 1>::from_data(
        TensorData::new(vec![0.5f32, 0.25f32, 0.75f32, 1.0f32], [4]),
        &device,
    );

    let output = embedder.forward(timestep);
    assert_eq!(
        output.dims(),
        [4, 100],
        "TimestepEmbedder output should be [batch, embed_dim]"
    );
}

/// Test ConditionalBlock: [batch, C, H, W] -> [batch, C, H, W]
#[test]
fn test_conditional_block() {
    use model_training::models::model_v1::model::ConditionalBlockConfig;
    type B = NdArray<f32>;
    let device = Default::default();

    let block = ConditionalBlockConfig::new(CHANNELS).init::<B>(&device);

    let x = Tensor::<B, 4>::from_data(
        TensorData::new(
            vec![0.5f32; 4 * CHANNELS * HEIGHT * WIDTH],
            [4, CHANNELS, HEIGHT, WIDTH],
        ),
        &device,
    );

    let output = block.forward(x);
    assert_eq!(
        output.dims(),
        [4, CHANNELS, HEIGHT, WIDTH],
        "ConditionalBlock should preserve shape"
    );
}

/// Test full ModelV1 forward: all inputs -> [batch, C, H, W]
#[test]
fn test_model_forward() {
    use model_training::models::model_v1::model::ModelV1Config;
    type B = NdArray<f32>;
    let device = Default::default();

    let model = ModelV1Config::new().init::<B>(&device);
    let batch = 4;

    let images = Tensor::<B, 4>::from_data(
        TensorData::new(
            vec![0.5f32; batch * CHANNELS * HEIGHT * WIDTH],
            [batch, CHANNELS, HEIGHT, WIDTH],
        ),
        &device,
    );
    let keys = Tensor::<B, 2>::from_data(
        TensorData::new(vec![0.0f32; batch * 108], [batch, 108]),
        &device,
    );
    let mouse = Tensor::<B, 3>::from_data(
        TensorData::new(
            vec![0.0f32; batch * 2 * MOUSE_VECTOR_LENGTH],
            [batch, 2, MOUSE_VECTOR_LENGTH],
        ),
        &device,
    );
    let next_noise = Tensor::<B, 4>::from_data(
        TensorData::new(
            vec![0.5f32; batch * CHANNELS * HEIGHT * WIDTH],
            [batch, CHANNELS, HEIGHT, WIDTH],
        ),
        &device,
    );
    let timestep =
        Tensor::<B, 1>::from_data(TensorData::new(vec![0.5f32; batch], [batch]), &device);

    let output = model.forward(images, keys, mouse, next_noise, timestep);
    assert_eq!(
        output.dims(),
        [batch, CHANNELS, HEIGHT, WIDTH],
        "ModelV1 output should match input image shape"
    );
}

/// Full training run — reads real data from data/hdf5_files/ and trains.
/// Uses CUDA backend when --features cuda, otherwise NdArray.
///
/// Run: cargo test -p model-training --features cuda --test training_test test_full_training -- --nocapture --ignored
#[test]
#[ignore] // Ignored by default — requires data files and GPU. Run explicitly.
fn test_full_training() {
    // cargo test -p model-training sets CWD to model-training/,
    // but data lives at project root data/hdf5_files/.
    // Change to project root so training::run() finds the data.
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let project_root = manifest_dir.parent().expect("project root");
    std::env::set_current_dir(project_root).expect("failed to set CWD to project root");

    // Verify data exists before starting
    assert!(
        std::path::Path::new("data/hdf5_files").exists(),
        "Training data not found at data/hdf5_files/. Run preprocessing first."
    );

    model_training::training::run();
}
