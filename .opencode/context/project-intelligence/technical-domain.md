<!-- Context: project-intelligence/technical | Priority: critical | Version: 2.0 | Updated: 2026-03-16 -->

# Technical Domain

> Технический стек, архитектура и ключевые решения проекта Frame Generator.

## Quick Reference

- **Purpose**: Понять как устроен проект технически
- **Update When**: Новые фичи, рефакторинг, изменение стека
- **Audience**: Разработчики, DevOps

## Primary Stack

| Layer | Technology | Version | Rationale |
|-------|-----------|---------|-----------|
| Language | Rust | 2024 (nightly) | Современный Rust с latest features |
| ML Framework | Burn | 0.20.1 | Neural network training/inference |
| TUI Framework | Ratatui | 0.29.0 | Terminal UI |
| Image Processing | image | 0.25.x | Загрузка/обработка изображений |
| Parallelism | rayon | 1.10.0 | Data parallelism в preprocessor |
| Error Handling | color-eyre | 0.6.5 | Pretty error reporting |
| Serialization | serde | 1.0.x | JSON/HDF5 serialization |
| Screen Capture | xcap, rdev | latest | Запись экрана и клавиш |

## Architecture Pattern

```
Type: Rust Workspace (monolithic)
Pattern: Multi-crate architecture с разделением ответственности
```

### Crate Structure

```
frame-generator/
├── ui/                  # Main TUI приложение (корневой crate)
├── common/              # Общие константы
├── recorder/            # Запись экрана и клавиш
├── preprocessor/        # Предобработка изображений (rayon)
└── model-training/     # Обучение модели (Burn)
```

## Key Technical Decisions

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Burn ML framework | Эффективный tensor computation | GPU acceleration (cuda/wgpu) |
| Ratatui для TUI |Активная разработка, crossterm совместимость | Кроссплатформенный terminal UI |
| Rayon для parallelism | Простой data-parallelism | Ускорение preprocessing |
| Workspace organization | Разделение domain-логики | Масштабируемость |
| color-eyre | Красивые ошибки в CLI | Удобство отладки |

## Integration Points

| System | Purpose | Protocol | Direction |
|--------|---------|----------|-----------|
| ffmpeg | Извлечение кадров из видео | CLI | Outbound |
| hdf5 | Сохранение данных | hdf5-metno | Internal |
| Terminal | TUI интерфейс | crossterm | Inbound |

## Technical Constraints

| Constraint | Origin | Impact |
|------------|--------|--------|
| ffmpeg required | Внешняя зависимость | Установка для сборки |
| hdf5 required | Хранение данных | Установка для preprocessing |
| Edition 2024 | nightly Rust | Требует nightly toolchain |

## Development Environment

```
Setup: cargo build
Requirements: Rust nightly, ffmpeg, hdf5
Local Dev: cargo run
Testing: cargo test
Single Test: cargo test test_name
Linting: cargo clippy
Format: cargo fmt
```

### Running Tests

```bash
# Все тесты
cargo test

# Конкретный crate
cargo test -p recorder
cargo test -p preprocessor

# Один тест
cargo test test_function_name

# С выводом
cargo test -- --nocapture
```

## Deployment

```
Environment: Desktop application
Platform: Cross-platform (Windows primarily)
CI/CD: None configured
```

## Onboarding Checklist

- [x] Знать основной tech stack (Rust 2024, Burn, Ratatui)
- [x] Понимать архитектуру workspace (5 crates)
- [x] Знать ключевые директории и их назначение
- [x] Понимать решения: Burn ML, Ratatui TUI, Rayon parallelism
- [x] Знать внешние зависимости (ffmpeg, hdf5)
- [x] Уметь запускать локально (`cargo run`)
- [x] Уметь запускать тесты (`cargo test`)

## 📂 Codebase References

- `Cargo.toml` (root) - Workspace definition, dependencies
- `src/main.rs` - TUI приложение, точка входа
- `src/utils.rs` - Утилиты
- `common/src/lib.rs` - Общие константы (WIDTH, HEIGHT, DATA_DIR)
- `recorder/src/` - Запись экрана и клавиш
- `preprocessor/src/` - Параллельная предобработка изображений
- `model-training/src/` - Обучение модели с Burn

## Related Files

- `AGENTS.md` - Полные гайдлайны для агентов (сборка, тесты, стиль)
- `business-domain.md` - Бизнес-контекст
