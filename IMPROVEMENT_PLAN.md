# Frame Generator - План улучшений

## Текущее состояние

- **Размер кадра:** 40x40x4 (очень маленький — это плюс для обучения!)
- **Embedders:** Мышь, клавиатура и timestep — все подключены и работают
- **TimestepEmbedder:** Реализован (sinusoidal encoding + linear projection)
- **ModelV1:** U-Net с conditional injection в bottleneck + DDPM sampling
- **ModelV2:** Полноценная latent diffusion — VAE (40x40x4 → 10x10x8) + LatentUNet + CrossAttention + DDIM sampling
- **Noise Schedule:** Cosine schedule с DDIM step
- **Инференс:** DDPM sampling (50 шагов) для V1, DDIM sampling в latent space для V2
- **GPU:** CUDA backend поддерживается (`cargo run --features cuda`)

---

## 🟢 Уровень 1: Быстрые исправления
**Сложность:** ⭐☆☆☆☆ (1/5)  
**Время:** 2-4 часа  
**Цель:** Заставить работать текущую архитектуру с минимальными изменениями

### 1.1 Подключить Conditional в U-Net

**Файл:** `model-training/src/models/unets/base_unet/model.rs`

**Проблема:** Параметр `conditional` передаётся, но не используется (строка 131: `// TODO: Добавить обработку condinional`)

```rust
// Добавить cross-attention или concat после bottleneck
pub fn forward(
    &self,
    images: Tensor<B, 4>,
    conditional: Tensor<B, 4>, // [B, embed_dim*2, H, W]
) -> Tensor<B, 4> {
    // ... encoder ...
    
    // В bottleneck добавить conditioning:
    let x = self.conv5.forward(x);
    let x = self.act5.forward(x);
    
    // КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: добавить conditional после encoder
    let x = x + conditional;  // или cross-attention
    
    // ... decoder ...
}
```

**Ожидаемый результат:** Нажатия кнопок начнут хоть как-то влиять на генерацию

---

### 1.2 Добавить Timestep Embedder

**Файл:** `model-training/src/models/embedders.rs` (раскомментировать)

```rust
#[derive(Module, Debug)]
pub struct TimestepEmbedder<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl TimestepEmbedderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TimestepEmbedder<B> {
        TimestepEmbedder {
            linear1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            linear2: LinearConfig::new(self.hidden_dim, self.output_dim).init(device),
        }
    }
}

impl<B: Backend> TimestepEmbedder<B> {
    pub fn forward(&self, timesteps: Tensor<B, 1>) -> Tensor<B, 2> {
        // Sinusoidal embeddings
        let x = self.linear1.forward(timesteps);
        let x = gelu(x);  // или swish
        let x = self.linear2.forward(x);
        x
    }
}
```

---

### 1.3 Исправить инференс — добавить DDPM sampling

**Файл:** `model-training/src/inference.rs`

**БЫЛО:**
```rust
let noise = batch.images.random_like(...);
let output = model.forward(...);
let output = batch.images * 0.9 + output * 0.1;
```

**СТАТЬ:**
```rust
// DDPM/DDIM sampling с 50-100 шагами
fn generate_ddpm(model: &Model, start_image: Tensor, actions: Tensor, num_steps: usize) -> Tensor {
    let mut x_t = Tensor::random([1, CHANNELS, HEIGHT, WIDTH], Distribution::Normal(0, 1));
    
    for t in (0..num_steps).rev() {
        let timestep = Tensor::from_data([t as f32 / num_steps as f32]);
        let predicted = model.forward(x_t.clone(), actions.clone(), timestep);
        x_t = x_t - predicted * (1.0 / num_steps as f32);  // упрощённый Euler
    }
    x_t
}
```

**Ожидаемый результат:** Модель начнёт реально генерировать, а не просто смешивать

---

### 1.4 Обновить ModelV1 для использования timesteps

**Файл:** `model-training/src/models/model_v1/model.rs`

```rust
impl<B: Backend> ModelV1<B> {
    pub fn forward(
        &self,
        images: Tensor<B, 4>,
        keys: Tensor<B, 2>,
        mouse: Tensor<B, 3>,
        noisy_input: Tensor<B, 4>,  // сейчас это "next_noise"
        timestep: Tensor<B, 1>,     // ДОБАВИТЬ
    ) -> Tensor<B, 4> {
        // Получить embeddings
        let mouse_emb = self.mouse_embedder.forward(mouse);
        let keys_emb = self.keys_embedder.forward(keys);
        let time_emb = self.timestep_embedder.forward(timestep);  // ДОБАВИТЬ
        
        // Конкатенация всех условий
        let embed = Tensor::cat(vec![mouse_emb, keys_emb, time_emb], 1);
        // ... reshape в spatial map ...
        
        // Передать в U-Net
        self.unet.forward(images, conditional)
    }
}
```

---

### Чеклист Уровня 1:

- [x] Подключить conditional в BaseUNet
- [x] Раскомментировать/реализовать TimestepEmbedder
- [x] Добавить DDPM sampling loop в inference
- [x] Обновить ModelV1.forward() для timesteps
- [x] Обновить training forward для использования timesteps

---

## 🟡 Уровень 2: Правильная архитектура диффузии
**Сложность:** ⭐⭐⭐☆☆ (3/5)  
**Время:** 1-2 недели  
**Цель:** Полноценная условная диффузия как в DIAMOND/GameNGen

### 2.1 Реализовать полноценный noise schedule

**Файл:** новый `model-training/src/models/noise_schedule.rs`

```rust
/// Cosine noise schedule (рекомендуется для 2024-2025)
pub struct CosineNoiseSchedule {
    pub num_timesteps: usize,
}

impl CosineNoiseSchedule {
    pub fn alpha_bar(&self, t: f32) -> f32 {
        // cos²((t + 0.008) * π / 2) для t ∈ [0, 1]
        let t = t + 0.008_f32;
        let cos_t = (t * std::f32::consts::PI / 2.0).cos();
        cos_t * cos_t
    }
    
    pub fn get(&self, t: usize) -> (f32, f32) {
        let t = t as f32 / self.num_timesteps as f32;
        let alpha = self.alpha_bar(t).sqrt();
        let sigma = (1.0 - self.alpha_bar(t)).sqrt();
        (alpha, sigma)
    }
}

/// DDPM + improved noise schedule (ICLR 2025)
pub struct EDANoiseSchedule {
    // exponential moving average schedule
}
```

---

### 2.2 Добавить VAE для latent space

**Файл:** новый `model-training/src/models/vae/model.rs`

```rust
/// VAE для сжатия изображений в latent space
/// 40x40x4 → 10x10x8 (8x сжатие)
#[derive(Module, Debug)]
pub struct VAE<B: Backend> {
    encoder: Encoder<B>,
    decoder: Decoder<B>,
}

impl VAE<B: Backend> {
    pub fn encode(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // 40x40x4 → 10x10x8 (latent)
        let h = self.encoder.conv1.forward(x);
        // ... residual blocks ...
        h
    }
    
    pub fn decode(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        // 10x10x8 → 40x40x4
        let h = self.decoder.conv1.forward(z);
        // ... residual blocks ...
        h
    }
    
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 1>, Tensor<B, 1>) {
        let z = self.encode(x);
        let mu = z.clone();
        let logvar = z.clone();  // для VAE reparameterization
        let z = self.reparameterize(mu, logvar);
        let reconstruction = self.decode(z);
        (reconstruction, mu, logvar)
    }
}
```

**Преимущества:**
- 8x меньше данных для обработки
- Модель учит семантику, а не пиксели
- Качество не страдает

---

### 2.3 Cross-attention для conditioning

**Файл:** `model-training/src/models/attention.rs`

```rust
/// Cross-attention block для передачи action information
#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    q_linear: Linear<B>,
    k_linear: Linear<B>,
    v_linear: Linear<B>,
    out_linear: Linear<B>,
}

impl<B: Backend> CrossAttention<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 4>,           // [B, C, H, W] - spatial features
        context: Tensor<B, 2>,       // [B, D] - action embeddings
    ) -> Tensor<B, 4> {
        // Q из spatial features
        // K, V из action embeddings
        // Multi-head attention
        // Output: [B, C, H, W]
    }
}
```

---

### 2.4 Полная архитектура ModelV2

**Файл:** новый `model-training/src/models/model_v2/mod.rs`

```rust
/// Полноценная условная диффузионная модель
pub struct ModelV2<B: Backend> {
    vae: VAE<B>,
    action_embedder: ActionEmbedder<B>,  // keys + mouse
    timestep_embedder: TimestepEmbedder<B>,
    unet: UNet<B>,  // работает в latent space
}

impl ModelV2<B: Backend> {
    pub fn forward_train(
        &self,
        x0: Tensor<B, 4>,      // оригинальное изображение
        actions: Tensor<B, 2>,  // keys + mouse
        timestep: usize,
    ) -> Tensor<B, 4> {
        // 1. Зашумление по schedule
        let (alpha, sigma) = noise_schedule.get(timestep);
        let noise = Tensor::random_like(x0, Distribution::Normal(0, 1));
        let xt = x0 * alpha + noise * sigma;
        
        // 2. VAE encode (опционально, для latent diffusion)
        // let xt = self.vae.encode(xt);
        
        // 3. Получить embeddings
        let action_emb = self.action_embedder.forward(actions);
        let time_emb = self.timestep_embedder.forward(timestep);
        
        // 4. Diffusion forward
        let predicted_noise = self.unet.forward(xt, action_emb, time_emb);
        
        // 5. Loss = MSE(predicted_noise, true_noise)
        predicted_noise
    }
    
    pub fn sample(
        &self,
        actions: Tensor<B, 2>,
        num_steps: usize,
    ) -> Tensor<B, 4> {
        // DDIM sampling
        let mut xt = Tensor::random_like(start, Distribution::Normal(0, 1));
        
        for t in (0..num_steps).rev() {
            let predicted = self.forward_train(xt, actions, t);
            xt = xt - predicted * noise_schedule.get_step_size(t);
        }
        
        // VAE decode если используется latent diffusion
        // self.vae.decode(xt)
        xt
    }
}
```

---

### 2.5 Training loop с правильным diffusion loss

**Файл:** `model-training/src/models/model_v2/training.rs`

```rust
impl<B: AutodiffBackend> TrainStep for ModelV2<B> {
    fn step(&self, batch: FrameBatch<B>) -> TrainOutput<DiffusionOutput<B>> {
        // 1. Sample random timestep
        let t = Tensor::random([batch_size], Distribution::Uniform(0, num_timesteps));
        
        // 2. Add noise по schedule
        let (alpha, sigma) = noise_schedule.get(t);
        let noise = Tensor::random_like(batch.images, Distribution::Normal(0, 1));
        let noisy_images = batch.images * alpha + noise * sigma;
        
        // 3. Forward pass
        let predicted_noise = self.forward(noisy_images, batch.actions, t);
        
        // 4. MSE loss между predicted и true noise
        let loss = mse_loss(predicted_noise, noise);
        
        TrainOutput::new(self, loss.backward(), DiffusionOutput { loss, t })
    }
}
```

---

### Чеклист Уровня 2:

- [x] Реализовать NoiseSchedule (cosine) — `models/noise_schedule.rs`
- [x] Добавить VAE encoder/decoder — `models/vae/model.rs` (40x40x4 → 10x10x8)
- [x] Реализовать CrossAttention block — `models/attention.rs` (multi-head, residual)
- [x] Создать ModelV2 с правильным diffusion forward — `models/model_v2/model.rs`
- [x] Обновить training loop с MSE(noise, predicted_noise) + KL loss — `models/model_v2/training.rs`
- [x] Реализовать DDIM sampling в inference — встроен в `ModelV2::sample()`
- [x] Data loader уже передаёт действия через FrameBatch

---

## 🔴 Уровень 3: Production-ready (2025-2026 best practices)
**Сложность:** ⭐⭐⭐⭐⭐ (5/5)  
**Время:** 2-4 месяца  
**Цель:** Модель уровня DIAMOND/GameNGen

### 3.1 DiT (Diffusion Transformer) вместо U-Net

**Файл:** новый `model-training/src/models/dit/model.rs`

```rust
/// DiT - Diffusion Transformer (Sora, Stable Diffusion 3 tech)
/// Вместо сверток - attention + FFN
pub struct DiT<B: Backend> {
    patch_embed: Linear<B>,     // patchify
    blocks: Vec<DiTBlock<B>>,   // transformer blocks
    final_layer: Linear<B>,     // unpatchify
}

pub struct DiTBlock<B: Backend> {
    norm1: AdaptiveLayerNorm<B>,
    attn: SelfAttention<B>,
    norm2: AdaptiveLayerNorm<B>,
    ff: FeedForward<B>,
}

impl DiTBlock<B: Backend> {
    pub fn forward(&self, x: Tensor<B, 2>, context: Tensor<B, 2>, timestep: Tensor<B, 2>) {
        // Adaptive Layer Norm получает timestep как condition
        let x = x + self.attn(self.norm1(x, timestep));
        let x = x + self.ff(self.norm2(x, timestep));
    }
}
```

**Преимущества:**
- Масштабируемость (больше параметров = лучше качество)
- LLM-подобная архитектура
- Sora, Stable Diffusion 3 используют DiT

---

### 3.2 Context frames с noise augmentation

**По GameNGen:** Обучать на 4+ предыдущих кадрах

```rust
/// Многокадровый вход
pub struct ContextFrames {
    pub frames: Vec<Tensor>,  // [t-3, t-2, t-1, t]
}

impl ContextFrames {
    pub fn with_noise_augmentation(&self, noise_level: f32) -> Self {
        // Добавлять разное количество шума к разным кадрам
        // Это критично для стабильности при авторегрессии!
        Self {
            frames: self.frames.iter()
                .map(|f| f + random_noise() * noise_level)
                .collect()
        }
    }
}
```

---

### 3.3 Classifier-free guidance

**При обучении:** С вероятностью 10-20% dropout действий

```rust
/// Classifier-free guidance training
pub fn forward(&self, x: Tensor, actions: Option<Tensor>, timestep: Tensor) {
    let cond = if actions.is_some() && random() > 0.15 {
        actions  // normal conditioning
    } else {
        None     // dropout - учим unconditional generation
    };
    
    // При инференсе: combine conditional + unconditional
    let out_cond = model(x, actions, timestep);
    let out_uncond = model(x, None, timestep);
    let output = out_uncond + guidance_scale * (out_cond - out_uncond);
}
```

---

### 3.4 Авторегенеративная генерация с KV cache

```rust
/// Генерация длинных последовательностей с кэшированием
pub fn generate_trajectory(
    &self,
    initial_frames: Vec<Tensor>,
    actions: Vec<Tensor>,
    num_frames: usize,
) -> Vec<Tensor> {
    let mut generated = initial_frames.clone();
    let mut cache = None;
    
    for i in 0..num_frames {
        // Используем последние N кадров как контекст
        let context = generated.last_n_frames(4);
        
        // KV cache для ускорения (не пересчитывать attention для старых кадров)
        let (output, new_cache) = self.forward_with_cache(
            context, 
            actions[i], 
            cache
        );
        cache = new_cache;
        
        generated.push(output);
    }
    generated
}
```

---

### 3.5 Action tokenization (как в LLM)

```rust
/// Токенизация действий как в языковых моделях
pub struct ActionTokenizer {
    pub num_key_tokens: usize,
    pub num_mouse_tokens: usize,
}

impl ActionTokenizer {
    pub fn tokenize(&self, keys: &[u8], mouse: &[[i32; 2]]) -> Vec<usize> {
        let mut tokens = Vec::new();
        
        // Key tokens (one-hot style)
        for i in 0..108 {
            tokens.push(if keys.contains(&i) { 1 } else { 0 });
        }
        
        // Mouse tokens (quantized position)
        for pos in mouse.iter().take(10) {  // последние 10 позиций
            let x_token = (pos[0] / 20) as usize;  // 20px bins
            let y_token = (pos[1] / 20) as usize;
            tokens.push(x_token + y_token * 100);
        }
        
        tokens
    }
}
```

---

### 3.6 Multi-step prediction (предсказание нескольких кадров сразу)

**По AVID/MultiGen:** Предсказывать сразу 2-4 кадра

```rust
/// Предсказание нескольких будущих кадров
pub fn forward_multi_step(
    &self,
    context_frames: Vec<Tensor>,
    actions: Vec<Tensor>,
    num_steps: usize,  // 2-4
) -> Vec<Tensor> {
    // DIAMOND-style: предсказываем N кадров вперёд
    // Один forward = N будущих кадров (а не один)
}
```

---

### Чеклист Уровня 3:

- [ ] Заменить U-Net на DiT (Diffusion Transformer)
- [ ] Реализовать context frames (4+ кадра)
- [ ] Добавить noise augmentation для контекста
- [ ] Добавить classifier-free guidance
- [ ] Реализовать KV cache для длинных rollouts
- [ ] Токенизация действий (action as tokens)
- [ ] Multi-step prediction (2-4 кадра вперёд)
- [ ] Optimizer: AdamW с cosine LR schedule
- [ ] Gradient accumulation для больших batch sizes
- [ ] Mixed precision training (FP16/BF16)

---

## 📊 Итоговая таблица

| Уровень | Сложность | Время | Основные изменения | Статус |
|---------|-----------|-------|-------------------|--------|
| **1** | ⭐☆☆☆☆ | 2-4ч | Подключить conditional + timesteps + DDPM loop | ✅ Готово |
| **2** | ⭐⭐⭐☆☆ | 1-2нед | VAE + CrossAttention + правильный diffusion loss | ✅ Готово |
| **3** | ⭐⭐⭐⭐⭐ | 2-4мес | DiT + KV cache + multi-step + 2025-2026 practices | ⬜ Следующий |

---

## 🎯 Рекомендуемая последовательность

```
Уровень 1 → Уровень 2 → Уровень 3
   ↓           ↓           ↓
 2-4ч      1-2нед      2-4мес
   ↓           ↓           ↓
 ✅ Done    ✅ Done    ⬜ Next
```

**Следующий шаг — Уровень 3.** Уровни 1 и 2 реализованы. Можно начинать обучение на ModelV2 и параллельно работать над Уровнем 3.

---

## 🔬 Ключевые ссылки на исследования

1. **DIAMOND** (NeurIPS 2024) - Диффузионный world model для Atari
   - https://arxiv.org/abs/2405.12399
   - https://github.com/eloialonso/diamond

2. **GameNGen** (ICLR 2025) - Neural game engine для DOOM
   - https://arxiv.org/abs/2408.14837
   - https://gamengen.github.io

3. **Vid2World** (2025) - Превращение video diffusion в interactive world models
   - https://arxiv.org/abs/2505.14357

4. **AVID** (RLJ 2025) - Адаптация pretrained video diffusion
   - https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_64.pdf

5. **PAN** (2025) - LLM backbone + diffusion decoder
   - https://arxiv.org/html/2511.09057v1

6. **AdaWorld** (2025) - Latent actions для адаптивных world models
   - https://arxiv.org/abs/2503.18938

7. **Cosmos Transfer** (NVIDIA 2025) - ControlNet-style conditioning
   - https://arxiv.org/pdf/2503.14492

8. **Improved Noise Schedule** (ICLR 2025)
   - https://openreview.net/pdf?id=j3U6CJLhqw
