use crate::client::{DEFAULT_BASE_URL, NvidiaClient};
use crate::embedding::NvidiaEmbeddingModel;
use rig_core::embeddings::EmbeddingModel;

#[test]
fn test_client_new() {
    let client = NvidiaClient::new("nvapi-test123");
    assert_eq!(client.base_url, DEFAULT_BASE_URL);
    assert!(client.api_key.is_some());
    assert_eq!(client.api_key.as_deref(), Some("nvapi-test123"));
}

#[test]
fn test_client_without_key() {
    let client = NvidiaClient::new_without_key();
    assert_eq!(client.base_url, DEFAULT_BASE_URL);
    assert!(client.api_key.is_none());
}

#[test]
fn test_client_from_env_missing() {
    let result = std::env::var("NVIDIA_API_KEY___NONEXISTENT");
    assert!(result.is_err());
}

#[test]
fn test_client_custom_url() {
    let client = NvidiaClient::new_without_key().with_base_url("http://localhost:8000/v1");
    assert_eq!(client.base_url, "http://localhost:8000/v1");
}

#[test]
fn test_agent_builder() {
    let client = NvidiaClient::new("nvapi-test");
    let builder = client.agent("nvidia/nemotron-3-super-120b-a12b");
    let _ = builder;
}

#[test]
fn test_embedding_model_creation() {
    let client = NvidiaClient::new("nvapi-test");
    let model = client.embedding_model("nvidia/nv-embedqa-e5-v5", 1024);
    assert_eq!(model.model, "nvidia/nv-embedqa-e5-v5");
    assert_eq!(model.ndims, 1024);
}

#[test]
fn test_embedding_model_make_default_ndims() {
    let client = NvidiaClient::new("nvapi-test");
    let model: NvidiaEmbeddingModel =
        EmbeddingModel::make(&client, "nvidia/nv-embedqa-e5-v5", None);
    assert_eq!(model.ndims, 1024);
}

#[test]
fn test_embedding_model_make_custom_dims() {
    let client = NvidiaClient::new("nvapi-test");
    let model: NvidiaEmbeddingModel =
        EmbeddingModel::make(&client, "nvidia/nv-embedqa-e5-v5", Some(512));
    assert_eq!(model.ndims, 512);
}
