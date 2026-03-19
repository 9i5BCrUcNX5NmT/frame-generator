use burn::{prelude::*, tensor::Distribution};

/// Cosine noise schedule for diffusion models.
///
/// Implements the improved cosine schedule from "Improved Denoising Diffusion
/// Probabilistic Models" (Nichol & Dhariwal, 2021).
/// alpha_bar(t) = cos²((t + s) / (1 + s) * π/2), where s = 0.008
pub struct CosineNoiseSchedule {
    pub num_timesteps: usize,
}

impl CosineNoiseSchedule {
    pub fn new(num_timesteps: usize) -> Self {
        Self { num_timesteps }
    }

    /// Compute cumulative signal rate alpha_bar at normalized time t ∈ [0, 1]
    fn alpha_bar_at(&self, t: f32) -> f32 {
        let s = 0.008_f32;
        let val = ((t + s) / (1.0 + s) * std::f32::consts::FRAC_PI_2).cos();
        val * val
    }

    /// Get (alpha, sigma) for a given discrete timestep
    /// alpha = sqrt(alpha_bar), sigma = sqrt(1 - alpha_bar)
    pub fn get(&self, t: usize) -> (f32, f32) {
        let t_norm = t as f32 / self.num_timesteps as f32;
        let ab = self.alpha_bar_at(t_norm);
        (ab.sqrt(), (1.0 - ab).sqrt())
    }

    /// Add noise to clean data: x_t = alpha * x_0 + sigma * noise
    pub fn add_noise<B: Backend>(
        &self,
        x0: Tensor<B, 4>,
        noise: Tensor<B, 4>,
        timestep: usize,
    ) -> Tensor<B, 4> {
        let (alpha, sigma) = self.get(timestep);
        x0 * alpha + noise * sigma
    }

    /// Compute step size for DDIM sampling at timestep t
    pub fn step_size(&self, t: usize) -> f32 {
        if t == 0 {
            return 0.0;
        }
        let (_alpha_t, sigma_t) = self.get(t);
        let (_alpha_prev, sigma_prev) = self.get(t - 1);

        // DDIM deterministic step coefficient
        sigma_prev / sigma_t
    }

    /// DDIM step: compute x_{t-1} from x_t and predicted noise
    pub fn ddim_step<B: Backend>(
        &self,
        x_t: Tensor<B, 4>,
        predicted_noise: Tensor<B, 4>,
        t: usize,
    ) -> Tensor<B, 4> {
        let (alpha_t, sigma_t) = self.get(t);
        let (alpha_prev, sigma_prev) = if t > 0 { self.get(t - 1) } else { (1.0, 0.0) };

        // Predict x_0 from x_t and predicted noise
        let x0_pred = (x_t.clone() - predicted_noise.clone() * sigma_t) / alpha_t;

        // Compute x_{t-1} using DDIM formula (deterministic, eta=0)
        x0_pred.clone() * alpha_prev + predicted_noise * sigma_prev
    }

    /// Sample a random timestep tensor for a batch (values in [0, num_timesteps))
    pub fn sample_timesteps<B: Backend>(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        Tensor::random(
            [batch_size],
            Distribution::Uniform(0.0, self.num_timesteps as f64),
            device,
        )
    }
}
