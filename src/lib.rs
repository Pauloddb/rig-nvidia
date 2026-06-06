//! # rig-nvidia
//!
//! [![crates.io](https://img.shields.io/crates/v/rig-nvidia.svg)](https://crates.io/crates/rig-nvidia)
//! [![docs.rs](https://docs.rs/rig-nvidia/badge.svg)](https://docs.rs/rig-nvidia)
//!
//! **NVIDIA NIM** provider for the [Rig](https://github.com/0xplaygrounds/rig) framework.
//!
//! The NVIDIA API is **OpenAI-compatible**, which simplifies the implementation:
//! the `/v1/chat/completions` and `/v1/embeddings` endpoints follow the same format
//! as OpenAI's, with minor adaptations (e.g., tool call arguments serialized
//! as stringified JSON).
//!
//! ## Features
//!
//! - **Chat completions** (non-streaming and streaming via SSE)
//! - **Embeddings** via `/v1/embeddings`
//! - **Tool calling** with incremental argument accumulation in streaming
//! - **Reasoning content** (chain-of-thought) in streaming
//! - Authentication via `NVIDIA_API_KEY` or explicit API key
//! - Customizable base URL support (self-hosted NIM / proxy)
//!
//! ## Quick Start
//!
//! ```no_run
//! use rig_nvidia::NvidiaClient;
//! use rig_core::completion::Prompt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = NvidiaClient::from_env()?;
//!
//!     let agent = client
//!         .agent("nvidia/nemotron-3-super-120b-a12b")
//!         .preamble("You are a Rust expert.")
//!         .build();
//!
//!     let response = agent.prompt("Explain ownership in Rust").await?;
//!     println!("{}", response);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Embeddings
//!
//! ```no_run
//! use rig_nvidia::NvidiaClient;
//! use rig_core::embeddings::EmbeddingModel;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = NvidiaClient::from_env()?;
//!     let model = client.embedding_model("nvidia/nv-embedqa-e5-v5", 1024);
//!
//!     let embeddings = model.embed_texts(vec!["Hello world".to_string()]).await?;
//!     println!("Dimension: {}", embeddings[0].vec.len());
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Modules
//!
//! - [`client`] — [`NvidiaClient`] (main entry point)
//! - [`completion`] — Completion model and request/response types
//! - [`embedding`] — Embedding model and request/response types
//! - [`error`] — [`NvidiaError`]
//! - [`json_utils`] — Serialization utilities (stringified JSON, null-or-vec, merge)
//! - [`message`] — NVIDIA message types and conversion from Rig
//! - [`streaming`] — Streaming types and SSE parser

pub mod client;
pub mod completion;
pub mod embedding;
pub mod error;
pub mod json_utils;
pub mod message;
pub mod streaming;

pub use client::NvidiaClient;
pub use completion::{NvidiaCompletionModel, NvidiaCompletionResponse};
pub use embedding::NvidiaEmbeddingModel;
pub use error::NvidiaError;
pub use message::NvidiaUsage;
pub use streaming::NvidiaStreamingResponse;

#[cfg(test)]
mod tests;
