# rig-nvidia

NVIDIA NIM provider integration for the [Rig](https://github.com/0xplaygrounds/rig) AI framework.

The NVIDIA NIM API is **OpenAI-compatible**, so this crate follows the same patterns as `rig-openai` with a few NVIDIA-specific adaptations (e.g. stringified JSON tool-call arguments, `null`-or-vec deserialization).

## Features

- **Chat completions** — non-streaming and streaming (SSE)
- **Embeddings** — via `/v1/embeddings`
- **Tool calling** — with incremental argument accumulation in streaming mode
- **Reasoning content** — chain-of-thought deltas in streaming
- **Configurable base URL** — for self-hosted NIM or proxy endpoints

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
rig-nvidia = "0.1"
rig-core = "0.38"
tokio = { version = "1", features = ["full"] }
```

### Chat Completion

```rust,no_run
use rig_nvidia::NvidiaClient;
use rig_core::completion::Prompt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = NvidiaClient::from_env()?;

    let agent = client
        .agent("nvidia/nemotron-3-super-120b-a12b")
        .preamble("You are a Rust expert.")
        .build();

    let response = agent.prompt("Explain ownership in Rust").await?;
    println!("{}", response);

    Ok(())
}
```

### Embeddings

```rust,no_run
use rig_nvidia::NvidiaClient;
use rig_core::embeddings::EmbeddingModel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = NvidiaClient::from_env()?;
    let model = client.embedding_model("nvidia/nv-embedqa-e5-v5", 1024);

    let embeddings = model.embed_texts(vec!["Hello world".to_string()]).await?;
    println!("Dimension: {}", embeddings[0].vec.len());

    Ok(())
}
```

### Custom Base URL (self-hosted NIM)

```rust
let client = NvidiaClient::new_without_key()
    .with_base_url("http://localhost:8000/v1");
```

## Authentication

Set the `NVIDIA_API_KEY` environment variable:

```bash
export NVIDIA_API_KEY=nvapi-xxxx
```

Or pass the key explicitly:

```rust
let client = NvidiaClient::new("nvapi-xxxx");
```

For self-hosted NIM deployments that don't require authentication:

```rust
let client = NvidiaClient::new_without_key();
```

## Architecture

```
src/
├── lib.rs          # Crate root — docs, re-exports, module declarations
├── client.rs       # NvidiaClient — entry point for agents & embeddings
├── completion.rs   # NvidiaCompletionModel, request/response types, CompletionModel impl
├── embedding.rs    # NvidiaEmbeddingModel, request/response types, EmbeddingModel impl
├── error.rs        # NvidiaError enum
├── json_utils.rs   # Stringified JSON, null-or-vec, merge utilities
├── message.rs      # NVIDIA message types, conversion from Rig, NvidiaUsage
├── streaming.rs    # SSE stream parser, streaming types, finalize_tool_call
└── tests/          # 52 unit tests across 6 modules
```

### Public API

| Type | Module | Description |
|---|---|---|
| `NvidiaClient` | `client` | Main entry point |
| `NvidiaCompletionModel` | `completion` | Chat completion model (implements `CompletionModel`) |
| `NvidiaCompletionResponse` | `completion` | Non-streaming response |
| `NvidiaEmbeddingModel` | `embedding` | Embedding model (implements `EmbeddingModel`) |
| `NvidiaStreamingResponse` | `streaming` | Streaming final response with usage |
| `NvidiaUsage` | `message` | Token usage statistics |
| `NvidiaError` | `error` | Error type |
| `json_utils` | `json_utils` | Serialization utilities (public for reuse) |

## Running Tests

```bash
cargo test
```

All 52 unit tests run without network access.

## License

MIT
