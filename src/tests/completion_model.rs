use crate::completion::{NvidiaChatRequest, NvidiaChoice, NvidiaCompletionResponse};
use crate::message::{
    NvidiaFunction, NvidiaMessage, NvidiaToolCall, NvidiaToolDefinition, NvidiaToolType,
    NvidiaUsage,
};
use rig_core::{
    completion::{self, CompletionError, CompletionRequest, GetTokenUsage},
    message, OneOrMany,
};

#[test]
fn test_response_with_tool_calls() {
    let response = NvidiaCompletionResponse {
        id: Some("chatcmpl-123".to_string()),
        model: Some("nvidia/test".to_string()),
        choices: vec![NvidiaChoice {
            index: 0,
            message: NvidiaMessage::Assistant {
                content: Some("".to_string()),
                tool_calls: vec![NvidiaToolCall {
                    id: "call_1".to_string(),
                    index: Some(0),
                    r#type: NvidiaToolType::Function,
                    function: NvidiaFunction {
                        name: "get_weather".to_string(),
                        arguments: serde_json::json!({"city": "Lisbon"}),
                    },
                }],
            },
            finish_reason: Some("tool_calls".to_string()),
        }],
        usage: Some(NvidiaUsage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        }),
    };

    let result: completion::CompletionResponse<NvidiaCompletionResponse> =
        response.try_into().unwrap();
    let items: Vec<_> = result.choice.into_iter().collect();
    assert_eq!(items.len(), 1);
    assert!(
        matches!(&items[0], completion::AssistantContent::ToolCall(tc) if tc.function.name == "get_weather")
    );
}

#[test]
fn test_response_text_only() {
    let response = NvidiaCompletionResponse {
        id: Some("chatcmpl-456".to_string()),
        model: Some("nvidia/test".to_string()),
        choices: vec![NvidiaChoice {
            index: 0,
            message: NvidiaMessage::Assistant {
                content: Some("Hello world".to_string()),
                tool_calls: vec![],
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Some(NvidiaUsage {
            prompt_tokens: 5,
            completion_tokens: 2,
            total_tokens: 7,
        }),
    };

    let result: completion::CompletionResponse<NvidiaCompletionResponse> =
        response.try_into().unwrap();
    let items: Vec<_> = result.choice.into_iter().collect();
    assert_eq!(items.len(), 1);
    assert!(
        matches!(&items[0], completion::AssistantContent::Text(t) if t.text() == "Hello world")
    );
}

#[test]
fn test_response_empty_choices() {
    let response = NvidiaCompletionResponse {
        id: None,
        model: None,
        choices: vec![],
        usage: None,
    };

    let result: Result<completion::CompletionResponse<NvidiaCompletionResponse>, CompletionError> =
        response.try_into();
    assert!(result.is_err());
    match result.unwrap_err() {
        CompletionError::ResponseError(msg) => {
            assert!(msg.contains("no choices"));
        }
        other => panic!("Expected ResponseError, got {:?}", other),
    }
}

#[test]
fn test_response_non_assistant_message() {
    let response = NvidiaCompletionResponse {
        id: None,
        model: None,
        choices: vec![NvidiaChoice {
            index: 0,
            message: NvidiaMessage::User {
                content: "unexpected".to_string(),
            },
            finish_reason: None,
        }],
        usage: None,
    };

    let result: Result<completion::CompletionResponse<NvidiaCompletionResponse>, CompletionError> =
        response.try_into();
    assert!(result.is_err());
    match result.unwrap_err() {
        CompletionError::ResponseError(msg) => {
            assert!(msg.contains("assistant"));
        }
        other => panic!("Expected ResponseError, got {:?}", other),
    }
}

#[test]
fn test_response_usage_conversion() {
    let response = NvidiaCompletionResponse {
        id: None,
        model: None,
        choices: vec![NvidiaChoice {
            index: 0,
            message: NvidiaMessage::Assistant {
                content: Some("hi".to_string()),
                tool_calls: vec![],
            },
            finish_reason: None,
        }],
        usage: Some(NvidiaUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        }),
    };

    let result: completion::CompletionResponse<NvidiaCompletionResponse> =
        response.try_into().unwrap();
    assert_eq!(result.usage.input_tokens, 100);
    assert_eq!(result.usage.output_tokens, 50);
    assert_eq!(result.usage.total_tokens, 150);
}

#[test]
fn test_nvidia_usage_get_token_usage() {
    let usage = NvidiaUsage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };
    let rig_usage = usage.token_usage().unwrap();
    assert_eq!(rig_usage.input_tokens, 10);
    assert_eq!(rig_usage.output_tokens, 20);
    assert_eq!(rig_usage.total_tokens, 30);
}

#[test]
fn test_tool_definition_conversion() {
    let tool_def = completion::ToolDefinition {
        name: "calculator".to_string(),
        description: "Does math".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "expr": {"type": "string"}
            }
        }),
    };
    let nvidia_def: NvidiaToolDefinition = tool_def.into();
    assert_eq!(nvidia_def.r#type, "function");
    assert_eq!(nvidia_def.function.name, "calculator");
}

#[test]
fn test_request_with_preamble_and_history() {
    let request = CompletionRequest {
        model: Some("nvidia/test".to_string()),
        preamble: Some("You are helpful.".to_string()),
        chat_history: OneOrMany::one(message::Message::user("Hi")),
        documents: vec![],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: Some(100),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
    };

    let nvidia_req = NvidiaChatRequest::from_completion_request(request, "nvidia/test").unwrap();
    assert_eq!(nvidia_req.model, "nvidia/test");
    assert_eq!(nvidia_req.temperature, Some(0.7));
    assert_eq!(nvidia_req.max_tokens, Some(100));
    assert!(nvidia_req
        .messages
        .iter()
        .any(|m| matches!(m, NvidiaMessage::System { .. })));
    assert!(nvidia_req
        .messages
        .iter()
        .any(|m| matches!(m, NvidiaMessage::User { .. })));
}

#[test]
fn test_request_tool_choice_auto() {
    let request = CompletionRequest {
        model: Some("nvidia/test".to_string()),
        preamble: None,
        chat_history: OneOrMany::one(message::Message::user("Hi")),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: Some(message::ToolChoice::Auto),
        additional_params: None,
        output_schema: None,
    };

    let nvidia_req = NvidiaChatRequest::from_completion_request(request, "nvidia/test").unwrap();
    assert_eq!(nvidia_req.tool_choice, Some(serde_json::json!("auto")));
}

#[test]
fn test_request_tool_choice_required() {
    let request = CompletionRequest {
        model: Some("nvidia/test".to_string()),
        preamble: None,
        chat_history: OneOrMany::one(message::Message::user("Hi")),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: Some(message::ToolChoice::Required),
        additional_params: None,
        output_schema: None,
    };

    let nvidia_req = NvidiaChatRequest::from_completion_request(request, "nvidia/test").unwrap();
    assert_eq!(nvidia_req.tool_choice, Some(serde_json::json!("required")));
}

#[test]
fn test_request_tool_choice_specific() {
    let request = CompletionRequest {
        model: Some("nvidia/test".to_string()),
        preamble: None,
        chat_history: OneOrMany::one(message::Message::user("Hi")),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: Some(message::ToolChoice::Specific {
            function_names: vec!["search".to_string()],
        }),
        additional_params: None,
        output_schema: None,
    };

    let nvidia_req = NvidiaChatRequest::from_completion_request(request, "nvidia/test").unwrap();
    let tc = nvidia_req.tool_choice.unwrap();
    assert_eq!(tc["type"], "function");
    assert_eq!(tc["function"]["name"], "search");
}

#[test]
fn test_request_additional_params_tools_extraction() {
    let request = CompletionRequest {
        model: Some("nvidia/test".to_string()),
        preamble: None,
        chat_history: OneOrMany::one(message::Message::user("Hi")),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: Some(serde_json::json!({
            "tools": [{"type": "function", "function": {"name": "extra_tool"}}],
            "custom_flag": true
        })),
        output_schema: None,
    };

    let nvidia_req = NvidiaChatRequest::from_completion_request(request, "nvidia/test").unwrap();
    assert_eq!(nvidia_req.tools.len(), 1);
    assert!(nvidia_req.additional_params.is_some());
    let params = nvidia_req.additional_params.unwrap();
    assert!(params.get("custom_flag").is_some());
    assert!(params.get("tools").is_none());
}
