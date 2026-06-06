use crate::message::NvidiaUsage;
use crate::streaming::{NvidiaStreamChunk, NvidiaStreamingResponse, finalize_tool_call};
use rig_core::{completion::GetTokenUsage, streaming};

#[test]
fn test_finalize_tool_call_empty() {
    let tc = streaming::RawStreamingToolCall::empty();
    let finalized = finalize_tool_call(tc);
    assert_eq!(finalized.name, "unknown");
    assert!(finalized.arguments.is_object());
    assert!(finalized.arguments.as_object().unwrap().is_empty());
}

#[test]
fn test_finalize_tool_call_partial() {
    let mut tc = streaming::RawStreamingToolCall::empty();
    tc.id = "call_1".to_string();
    tc.name = "search".to_string();
    tc.arguments = serde_json::Value::Null;
    let finalized = finalize_tool_call(tc);
    assert_eq!(finalized.id, "call_1");
    assert_eq!(finalized.name, "search");
    assert!(finalized.arguments.is_object());
}

#[test]
fn test_finalize_tool_call_with_args() {
    let mut tc = streaming::RawStreamingToolCall::empty();
    tc.id = "call_2".to_string();
    tc.name = "calc".to_string();
    tc.arguments = serde_json::json!({"x": 1});
    let finalized = finalize_tool_call(tc);
    assert_eq!(finalized.name, "calc");
    assert_eq!(finalized.arguments, serde_json::json!({"x": 1}));
}

#[test]
fn test_stream_chunk_deserialization() {
    let json = r#"{
        "choices": [
            {
                "delta": {"content": "Hello"},
                "finish_reason": null
            }
        ],
        "usage": null
    }"#;
    let chunk: NvidiaStreamChunk = serde_json::from_str(json).unwrap();
    assert_eq!(chunk.choices.len(), 1);
    assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    assert!(chunk.usage.is_none());
}

#[test]
fn test_stream_chunk_with_usage() {
    let json = r#"{
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }"#;
    let chunk: NvidiaStreamChunk = serde_json::from_str(json).unwrap();
    assert!(chunk.choices.is_empty());
    let usage = chunk.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 5);
}

#[test]
fn test_stream_delta_with_tool_calls() {
    let json = r#"{
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "index": 0,
                            "function": {"name": "search", "arguments": "{\"q\":"}
                        }
                    ]
                },
                "finish_reason": null
            }
        ],
        "usage": null
    }"#;
    let chunk: NvidiaStreamChunk = serde_json::from_str(json).unwrap();
    let delta = &chunk.choices[0].delta;
    assert_eq!(delta.tool_calls.len(), 1);
    assert_eq!(delta.tool_calls[0].id, Some("call_1".to_string()));
    assert_eq!(
        delta.tool_calls[0].function.name,
        Some("search".to_string())
    );
}

#[test]
fn test_stream_delta_with_reasoning() {
    let json = r#"{
        "choices": [
            {
                "delta": {"reasoning_content": "Let me think..."},
                "finish_reason": null
            }
        ],
        "usage": null
    }"#;
    let chunk: NvidiaStreamChunk = serde_json::from_str(json).unwrap();
    assert_eq!(
        chunk.choices[0].delta.reasoning_content,
        Some("Let me think...".to_string())
    );
}

#[test]
fn test_stream_delta_empty_content() {
    let json = r#"{
        "choices": [
            {
                "delta": {"content": ""},
                "finish_reason": null
            }
        ],
        "usage": null
    }"#;
    let chunk: NvidiaStreamChunk = serde_json::from_str(json).unwrap();
    assert_eq!(chunk.choices[0].delta.content, Some(String::new()));
}

#[test]
fn test_nvidia_streaming_response_token_usage() {
    let response = NvidiaStreamingResponse {
        usage: NvidiaUsage {
            prompt_tokens: 50,
            completion_tokens: 25,
            total_tokens: 75,
        },
    };
    let usage = response.token_usage().unwrap();
    assert_eq!(usage.input_tokens, 50);
    assert_eq!(usage.output_tokens, 25);
    assert_eq!(usage.total_tokens, 75);
}
