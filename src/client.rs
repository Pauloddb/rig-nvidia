//! NVIDIA NIM client — entry point for constructing agents and embedding models.
//!
//! [`NvidiaClient`] holds the API key, base URL, and a shared HTTP client.
//! It provides convenience methods to create completion agents and embedding
//! models.

use std::sync::Arc;

use crate::completion::NvidiaCompletionModel;
use crate::embedding::NvidiaEmbeddingModel;
use crate::error::NvidiaError;

/// Default base URL for the NVIDIA NIM API.
pub(crate) const DEFAULT_BASE_URL: &str = "https://integrate.api.nvidia.com/v1";
const AUTH_HEADER: &str = "Authorization";

/// Client for the NVIDIA NIM API.
///
/// This is the main entry point. It stores the API key, base URL, and a
/// reusable [`reqwest::Client`] shared across all model instances.
///
/// # Examples
///
/// ```ignore
/// use rig_nvidia::NvidiaClient;
/// use rig_core::completion::Prompt;
///
/// let client = NvidiaClient::from_env()?;
/// let agent = client
///     .agent("nvidia/nemotron-3-super-120b-a12b")
///     .preamble("You are a helpful assistant.")
///     .build();
///
/// let response = agent.prompt("Hello!").await?;
/// ```
#[derive(Clone, Debug)]
pub struct NvidiaClient {
    pub(crate) api_key: Option<String>,
    pub(crate) base_url: String,
    pub(crate) http_client: Arc<reqwest::Client>,
}

impl NvidiaClient {
    /// Create a new client with the given API key.
    ///
    /// Uses the default NVIDIA NIM base URL (`https://integrate.api.nvidia.com/v1`).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: Some(api_key.into()),
            base_url: DEFAULT_BASE_URL.to_string(),
            http_client: Arc::new(reqwest::Client::new()),
        }
    }

    /// Create a client **without** an API key.
    ///
    /// Useful for local NIM deployments that don't require authentication.
    pub fn new_without_key() -> Self {
        Self {
            api_key: None,
            base_url: DEFAULT_BASE_URL.to_string(),
            http_client: Arc::new(reqwest::Client::new()),
        }
    }

    /// Create a client from the `NVIDIA_API_KEY` environment variable.
    ///
    /// # Errors
    ///
    /// Returns [`NvidiaError::EnvVarError`] if `NVIDIA_API_KEY` is not set.
    pub fn from_env() -> Result<Self, NvidiaError> {
        let api_key = std::env::var("NVIDIA_API_KEY")
            .map_err(|_| NvidiaError::EnvVarError("NVIDIA_API_KEY".to_string()))?;
        Ok(Self::new(api_key))
    }

    /// Override the base URL.
    ///
    /// Useful for pointing at a self-hosted NIM endpoint or a proxy.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Create an [`rig_core::agent::AgentBuilder`] for the given completion model.
    ///
    /// The returned builder follows Rig's standard agent-building pattern:
    ///
    /// ```ignore
    /// use rig_nvidia::NvidiaClient;
    /// use rig_core::completion::Prompt;
    ///
    /// let client = NvidiaClient::new("nvapi-xxx");
    /// let agent = client
    ///     .agent("nvidia/nemotron-3-super-120b-a12b")
    ///     .preamble("You are a Rust expert.")
    ///     .build();
    ///
    /// let answer = agent.prompt("Explain ownership.").await?;
    /// ```
    pub fn agent(&self, model: &str) -> rig_core::agent::AgentBuilder<NvidiaCompletionModel> {
        let model = NvidiaCompletionModel {
            client: Arc::clone(&self.http_client),
            api_key: self.api_key.clone(),
            model: model.to_string(),
            base_url: self.base_url.clone(),
        };
        rig_core::agent::AgentBuilder::new(model)
    }

    /// Create an [`NvidiaEmbeddingModel`] for the given model and dimensionality.
    ///
    /// `ndims` specifies the expected output vector dimension.
    pub fn embedding_model(&self, model: &str, ndims: usize) -> NvidiaEmbeddingModel {
        NvidiaEmbeddingModel {
            client: Arc::clone(&self.http_client),
            api_key: self.api_key.clone(),
            model: model.to_string(),
            base_url: self.base_url.clone(),
            ndims,
        }
    }

    /// List available models from the NVIDIA NIM `/v1/models` endpoint.
    ///
    /// Returns a list of model IDs (strings).
    pub async fn list_models(&self) -> Result<Vec<String>, NvidiaError> {
        let mut request = self.http_client.get(format!("{}/models", self.base_url));

        if let Some(ref key) = self.api_key {
            request = request.header(AUTH_HEADER, format!("Bearer {}", key));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(NvidiaError::ApiError {
                status,
                message: text,
            });
        }

        let json: serde_json::Value = response.json().await?;
        let models = json["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }
}
