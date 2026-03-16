# AGENTS.md - Agent Coding Guidelines

This file provides guidance for agentic coding agents operating in this repository.

## Project Overview

Frame Generator is a Rust project using the [Burn](https://github.com/burn-rs/burn) machine learning library for frame generation. It consists of a workspace with multiple crates:

- **ui** (root) - Terminal UI using ratatui
- **common** - Shared constants
- **recorder** - Screen/keyboard recording
- **preprocessor** - Image preprocessing with rayon parallelism
- **model-training** - Burn-based model training

---

## Build, Lint, and Test Commands

### Building

```bash
# Build entire workspace (all crates)
cargo build

# Build specific crate
cargo build -p recorder
cargo build -p preprocessor
cargo build -p model-training

# Build with specific backend features
cargo build --features wgpu   # Use WGPU backend
cargo build --features cuda   # Use CUDA backend

# Release build
cargo build --release
```

### Running

```bash
# Run the main application
cargo run

# Run with specific features
cargo run --features wgpu
cargo run --release
```

### Testing

```bash
# Run all tests in workspace
cargo test

# Run tests for specific crate
cargo test -p recorder
cargo test -p preprocessor
cargo test -p model-training
cargo test -p common

# Run a SINGLE test by name
cargo test test_function_name
cargo test --test integration_test_name
cargo test -p recorder test_specific_function

# Run tests with output visible
cargo test -- --nocapture

# Run tests for specific crate with output
cargo test -p recorder -- --nocapture
```

### Linting and Formatting

```bash
# Run clippy for all linting suggestions
cargo clippy

# Fix automatically fixable issues
cargo clippy --fix --allow-dirty

# Format all code
cargo fmt

# Check formatting without modifying
cargo fmt --check
```

### Documentation

```bash
# Generate and view documentation
cargo doc --open

# Build documentation without opening
cargo doc --no-deps
```

### Workspace Commands

```bash
# Check for dependencies updates
cargo outdated

# Clean build artifacts
cargo clean

# Run with verbose output
cargo build -v
```

---

## Code Style Guidelines

### Formatting

- **Use `cargo fmt`** before committing - enforces Rust standard formatting
- Maximum line length: 100 characters (default)
- Use 4 spaces for indentation (Rust standard)
- Use trailing commas in multi-line collections

### Imports

- **Group imports** in the following order (use `cargo fmt` to enforce):
  1. Standard library (`std::`, `core::`)
  2. External crates (`crate::`, `extern::`)
  3. Local modules (`super::`, `self::`)

```rust
// Good import organization
use std::{io, path::PathBuf, str::FromStr};

use crossterm::event::{self, Event, KeyCode};
use image::DynamicImage;
use ratatui::{DefaultTerminal, Frame, Widget};

use crate::utils::helper_function;
```

- **Avoid wildcard imports** (`use module::*`) except for re-exports
- **Use absolute paths** for crate imports (`crate::`, `module::`)

### Types and Type Annotations

- **Prefer explicit type annotations** in function signatures
- **Use generic types** when appropriate for reusability
- **Leverage Rust's type inference** for local variables

```rust
// Good: Explicit return type
fn process_data(input: &str) -> Result<Vec<u8>, Error> { ... }

// Good: Type inference for local variables
let mut items = Vec::new();
let result = calculate_value(42);
```

### Naming Conventions

- **Snake_case** for functions, variables, and modules
- **PascalCase** for types, structs, enums, and traits
- **SCREAMING_SNAKE_CASE** for constants
- **CamelCase** for enum variants

```rust
// Constants
const MAX_BUFFER_SIZE: usize = 1024;
const DATA_DIR: &str = "data/";

// Structs and Types
pub struct VideoRecorder { ... }
pub enum RecordingState { ... }

// Functions and variables
fn process_my_images() { ... }
let mut writer = Writer::new();
```

### Error Handling

- **Use `color_eyre`** for application-level error handling (already configured)
- Install in `main()` with `color_eyre::install()?`
- **Prefer `Result` types** over exceptions
- **Use `?` operator** for propagating errors

```rust
// Main function with color_eyre
fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    
    let file = open_file("path.txt")?;
    process_file(file)?;
    
    Ok(())
}

// Propagate errors with ?
fn get_first_file_in_directory(path: &str) -> io::Result<Option<PathBuf>> {
    let mut entries = fs::read_dir(path)?;
    Ok(entries.next().map(|e| e.map(|e| e.path())))
}
```

- **Use `expect()` or `unwrap()` sparingly** - only for truly unrecoverable cases
- Document panics with clear messages when they occur

### Module Organization

- **One module per file** - use `mod module_name;` to include
- **Use `lib.rs`** for crate root public API
- **Use `mod.rs`** for submodules (or barrel file pattern)

```rust
// recorder/src/lib.rs
mod keys_recorder;
mod video_recorder;

pub use keys_recorder::KeysRecorder;
pub use video_recorder::VideoRecorder;

pub fn run() { ... }
```

### Async and Concurrency

- This project uses **rayon** for data parallelism in preprocessor
- Use `rayon::par_iter()` for parallel operations
- Use `Arc<Mutex<T>>` for shared state (as seen in recorder)

```rust
use rayon::prelude::*;

images
    .par_iter()
    .map(|img| process_image(img))
    .collect::<Vec<_>>()
```

### Documentation

- **Document public APIs** with doc comments (`///` or `//!`)
- Include examples where helpful
- Use comments (`//`) for implementation details

```rust
/// Runs the application's main loop until the user quits
pub fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> { ... }
```

### Dependencies

- All crate dependencies are in respective `Cargo.toml` files
- Workspace dependencies defined in root `Cargo.toml`
- Use **exact versions** (e.g., `0.29.0`) for stability

---

## External Dependencies

The project requires these system dependencies:

- **ffmpeg** - For video frame extraction
- **hdf5** - For HDF5 data file handling

---

## Key Files

| File/Directory | Purpose |
|---------------|---------|
| `src/main.rs` | Application entry point with TUI |
| `src/utils.rs` | Utility functions |
| `common/src/lib.rs` | Shared constants |
| `recorder/src/` | Screen and keyboard recording |
| `preprocessor/src/` | Image preprocessing pipeline |
| `model-training/src/` | ML model training with Burn |

---

## Notes for Agents

1. **Always run `cargo fmt`** before committing changes
2. **Run `cargo clippy`** to catch common mistakes
3. **Test single functions** using `cargo test function_name`
4. **Use edition = "2024"** for any new crates (requires nightly Rust)
5. **color_eyre** is used for pretty error reporting - use `?` for error propagation
