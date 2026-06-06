use crate::client::{DEFAULT_BASE_URL, NvidiaClient};
use crate::embedding::{NvidiaEmbeddingModel, NvidiaEmbeddingRequest, NvidiaEmbeddingResponse};
use rig_core::embeddings::EmbeddingModel;

#[test]
fn test_embedding_request_serialization() {
    let req = NvidiaEmbeddingRequest {
        model: "nvidia/nv-embedqa-e5-v5".to_string(),
        input: vec!["hello".to_string(), "world".to_string()],
        encoding_format: Some("float".to_string()),
        input_type: None,
        truncate: None,
    };
    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["model"], "nvidia/nv-embedqa-e5-v5");
    assert_eq!(json["input"], serde_json::json!(["hello", "world"]));
    assert_eq!(json["encoding_format"], "float");
    assert!(json.get("input_type").is_none());
    assert!(json.get("truncate").is_none());
}

#[test]
fn test_embedding_response_parsing() {
    let json = r#"{
        "data": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0},
            {"embedding": [0.4, 0.5, 0.6], "index": 1}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5}
    }"#;
    let resp: NvidiaEmbeddingResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.data.len(), 2);
    assert_eq!(resp.data[0].index, 0);
    assert_eq!(resp.data[0].embedding, vec![0.1, 0.2, 0.3]);
    assert_eq!(resp.data[1].index, 1);
}

#[test]
fn test_embedding_model_make() {
    let client = NvidiaClient::new("nvapi-test");
    let model: NvidiaEmbeddingModel =
        EmbeddingModel::make(&client, "nvidia/nv-embedqa-e5-v5", Some(768));
    assert_eq!(model.model, "nvidia/nv-embedqa-e5-v5");
    assert_eq!(model.ndims, 768);
    assert_eq!(model.base_url, DEFAULT_BASE_URL);
}

#[test]
fn test_embedding_model_ndims() {
    let client = NvidiaClient::new("nvapi-test");
    let model = client.embedding_model("nvidia/nv-embedqa-e5-v5", 2048);
    assert_eq!(model.ndims(), 2048);
}

#[test]
fn test_embedding_max_documents() {
    assert_eq!(NvidiaEmbeddingModel::MAX_DOCUMENTS, 256);
}

#[test]
fn test_embedding_response_out_of_order() {
    let json = r#"{
        "data": [
            {"embedding": [0.4, 0.5, 0.6], "index": 1},
            {"embedding": [0.1, 0.2, 0.3], "index": 0}
        ],
        "usage": null
    }"#;
    let resp: NvidiaEmbeddingResponse = serde_json::from_str(json).unwrap();
    let emb0 = resp.data.iter().find(|d| d.index == 0).unwrap();
    let emb1 = resp.data.iter().find(|d| d.index == 1).unwrap();
    assert_eq!(emb0.embedding, vec![0.1, 0.2, 0.3]);
    assert_eq!(emb1.embedding, vec![0.4, 0.5, 0.6]);
}

#[test]
fn test_embedding_data_f64_not_f32() {
    let json = r#"{
        "data": [
            {"embedding": [0.123456789012345], "index": 0}
        ],
        "usage": null
    }"#;
    let resp: NvidiaEmbeddingResponse = serde_json::from_str(json).unwrap();
    let val = resp.data[0].embedding[0];
    assert!(val - 0.123456789012345_f64 < f64::EPSILON);
}
