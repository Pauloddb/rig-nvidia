//! Embedding model, request/response types, and the [`EmbeddingModel`] impl.
//!
//! This module provides [`NvidiaEmbeddingModel`], which implements Rig's
//! [`EmbeddingModel`] trait for the NVIDIA NIM `/v1/embeddings` endpoint.

use std::sync::Arc;

use rig_core::embeddings::{Embedding, EmbeddingError, EmbeddingModel};
use serde::{Deserialize, Serialize};

use crate::message::NvidiaUsage;

const AUTH_HEADER: &str = "Authorization";

/// NVIDIA NIM embedding model.
///
/// Implements [`EmbeddingModel`] so it can be used with Rig's embedding
/// infrastructure.
///
/// Typically created via [`NvidiaClient::embedding_model`].
///
/// [`NvidiaClient::embedding_model`]: crate::client::NvidiaClient::embedding_model
#[derive(Clone, Debug)]
pub struct NvidiaEmbeddingModel {
    pub(crate) client: Arc<reqwest::Client>,
    pub(crate) api_key: Option<String>,
    /// Model identifier (e.g. `"nvidia/nv-embedqa-e5-v5"`).
    pub model: String,
    pub(crate) base_url: String,
    /// Expected output vector dimensionality.
    pub ndims: usize,
}

/// Request body sent to `/v1/embeddings`.
#[derive(Debug, Serialize)]
pub(crate) struct NvidiaEmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<String>,
}

/// Response body from `/v1/embeddings`.
#[derive(Debug, Deserialize)]
pub(crate) struct NvidiaEmbeddingResponse {
    /// Embedding vectors returned by the API.
    pub data: Vec<NvidiaEmbeddingData>,
    #[allow(dead_code)]
    usage: Option<NvidiaUsage>,
}

/// A single embedding vector inside a [`NvidiaEmbeddingResponse`].
#[derive(Debug, Deserialize)]
pub(crate) struct NvidiaEmbeddingData {
    /// The embedding vector (`f64` to match Rig's [`Embedding`] type).
    pub embedding: Vec<f64>,
    /// Index of this embedding relative to the input texts.
    pub index: usize,
}

/// Map a [`reqwest::Error`] into an [`EmbeddingError::HttpError`].
pub(crate) fn map_reqwest_embedding_error(e: reqwest::Error) -> EmbeddingError {
    EmbeddingError::HttpError(rig_core::http_client::Error::Instance(Box::new(e)))
}

impl EmbeddingModel for NvidiaEmbeddingModel {
    const MAX_DOCUMENTS: usize = 256;
    type Client = crate::client::NvidiaClient;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        Self {
            client: Arc::clone(&client.http_client),
            api_key: client.api_key.clone(),
            model: model.into(),
            base_url: client.base_url.clone(),
            ndims: dims.unwrap_or(1024),
        }
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + rig_core::wasm_compat::WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>>
    + rig_core::wasm_compat::WasmCompatSend {
        let texts: Vec<String> = texts.into_iter().collect();
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();
        let base_url = self.base_url.clone();

        async move {
            if texts.is_empty() {
                return Ok(Vec::new());
            }

            let payload = NvidiaEmbeddingRequest {
                model,
                input: texts.clone(),
                encoding_format: Some("float".to_string()),
                input_type: None,
                truncate: None,
            };

            let mut request = client
                .post(format!("{}/embeddings", base_url))
                .json(&payload);

            if let Some(ref key) = api_key {
                request = request.header(AUTH_HEADER, format!("Bearer {}", key));
            }

            let response = request.send().await.map_err(map_reqwest_embedding_error)?;

            if !response.status().is_success() {
                let status = response.status().as_u16();
                let body = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                return Err(EmbeddingError::ProviderError(format!(
                    "NVIDIA embedding error {}: {}",
                    status, body
                )));
            }

            let embed_response: NvidiaEmbeddingResponse = response.json().await.map_err(|e| {
                EmbeddingError::ResponseError(format!("Failed to parse embedding response: {}", e))
            })?;

            let mut embeddings = Vec::with_capacity(texts.len());
            for (i, text) in texts.into_iter().enumerate() {
                let vec = embed_response
                    .data
                    .iter()
                    .find(|d| d.index == i)
                    .map(|d| d.embedding.clone())
                    .unwrap_or_default();

                embeddings.push(Embedding {
                    document: text,
                    vec,
                });
            }

            Ok(embeddings)
        }
    }
}
