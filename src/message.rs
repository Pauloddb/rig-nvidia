//! NVIDIA message types and conversion from Rig's message model.
//!
//! This module defines the wire-format message types used by the NVIDIA NIM
//! API (which follows the OpenAI-compatible format) and provides conversions
//! from Rig's [`message::Message`] into the internal `NvidiaMessage` type.
//!
//! It also contains [`NvidiaUsage`] — the token-usage struct shared by both
//! completion and streaming responses.

use rig_core::completion::{self, GetTokenUsage};
use rig_core::message::{self, Document as MsgDocument, DocumentSourceKind};
use serde::{Deserialize, Serialize};

use crate::json_utils;

/// A chat message in NVIDIA's OpenAI-compatible wire format.
///
/// Each variant corresponds to a `role` tag in the JSON body sent to the
/// `/v1/chat/completions` endpoint.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub(crate) enum NvidiaMessage {
    /// System prompt (`"role": "system"`).
    System { content: String },
    /// User message (`"role": "user"`).
    User { content: String },
    /// Assistant response (`"role": "assistant"`).
    Assistant {
        content: Option<String>,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_vec",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<NvidiaToolCall>,
    },
    /// Tool result (`"role": "tool"`).
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

/// A tool call inside an assistant message.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct NvidiaToolCall {
    /// Unique identifier for this tool call (e.g. `"call_abc123"`).
    pub id: String,
    /// Optional index used for ordering in streaming deltas.
    #[serde(default)]
    pub index: Option<usize>,
    /// Always `Function` for NVIDIA NIM.
    #[serde(default = "default_function")]
    pub r#type: NvidiaToolType,
    /// The function name and stringified-JSON arguments.
    pub function: NvidiaFunction,
}

fn default_function() -> NvidiaToolType {
    NvidiaToolType::Function
}

/// Tool type discriminator — currently only `Function` is supported.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone, Default)]
#[serde(rename_all = "lowercase")]
pub(crate) enum NvidiaToolType {
    #[default]
    Function,
}

/// The function invocation within a tool call.
///
/// `arguments` is stored as a [`serde_json::Value`] but serialized/deserialized
/// as a **stringified JSON** string because NVIDIA NIM encodes arguments as
/// `"arguments": "{\"key\": \"value\"}"`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct NvidiaFunction {
    /// Name of the function to invoke.
    pub name: String,
    /// Arguments as a JSON value (stringified on the wire).
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
}

/// NVIDIA's tool definition wrapper.
///
/// Wraps a Rig [`completion::ToolDefinition`] and adds the `"type": "function"`
/// field required by the API.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct NvidiaToolDefinition {
    /// Always `"function"`.
    pub r#type: String,
    /// The underlying tool definition.
    pub function: completion::ToolDefinition,
}

impl From<completion::ToolDefinition> for NvidiaToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: tool,
        }
    }
}

impl From<message::ToolResult> for NvidiaMessage {
    fn from(tool_result: message::ToolResult) -> Self {
        let content = match tool_result.content.first() {
            message::ToolResultContent::Text(text) => text.text,
            message::ToolResultContent::Image(_) => String::from("[Image]"),
        };
        NvidiaMessage::ToolResult {
            tool_call_id: tool_result.id,
            content,
        }
    }
}

impl From<message::ToolCall> for NvidiaToolCall {
    fn from(tool_call: message::ToolCall) -> Self {
        Self {
            id: tool_call.id,
            index: None,
            r#type: NvidiaToolType::Function,
            function: NvidiaFunction {
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            },
        }
    }
}

/// Convert a Rig [`message::Message`] into one or more [`NvidiaMessage`]s.
///
/// A single Rig message can produce multiple NVIDIA messages — for example,
/// a `User` message that contains both tool results and text is split into
/// separate `ToolResult` and `User` messages.
pub(crate) fn convert_message(
    message: message::Message,
) -> Result<Vec<NvidiaMessage>, message::MessageError> {
    match message {
        message::Message::System { content } => Ok(vec![NvidiaMessage::System { content }]),
        message::Message::User { content } => {
            let mut messages = vec![];

            let tool_results: Vec<NvidiaMessage> = content
                .clone()
                .into_iter()
                .filter_map(|c| match c {
                    message::UserContent::ToolResult(tr) => Some(NvidiaMessage::from(tr)),
                    _ => None,
                })
                .collect();
            messages.extend(tool_results);

            let text_content: String =
                content
                    .into_iter()
                    .filter_map(|c| match c {
                        message::UserContent::Text(text) => Some(text.text),
                        message::UserContent::Document(MsgDocument {
                            data:
                                DocumentSourceKind::Base64(content)
                                | DocumentSourceKind::String(content),
                            ..
                        }) => Some(content),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

            if !text_content.is_empty() {
                messages.push(NvidiaMessage::User {
                    content: text_content,
                });
            }

            Ok(messages)
        }
        message::Message::Assistant { content, .. } => {
            let mut text_content = String::new();
            let mut tool_calls = Vec::new();

            for item in content.iter() {
                match item {
                    message::AssistantContent::Text(text) => {
                        text_content.push_str(text.text());
                    }
                    message::AssistantContent::ToolCall(tc) => {
                        tool_calls.push(NvidiaToolCall::from(tc.clone()));
                    }
                    message::AssistantContent::Reasoning(_) => {}
                    message::AssistantContent::Image(_) => {}
                }
            }

            Ok(vec![NvidiaMessage::Assistant {
                content: if text_content.is_empty() {
                    None
                } else {
                    Some(text_content)
                },
                tool_calls,
            }])
        }
    }
}

/// Token usage statistics returned by the NVIDIA API.
///
/// This type is public because it appears in both [`NvidiaCompletionResponse`]
/// and [`NvidiaStreamingResponse`].
///
/// [`NvidiaCompletionResponse`]: crate::completion::NvidiaCompletionResponse
/// [`NvidiaStreamingResponse`]: crate::streaming::NvidiaStreamingResponse
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct NvidiaUsage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
    /// Total tokens (prompt + completion).
    pub total_tokens: u32,
}

impl GetTokenUsage for NvidiaUsage {
    fn token_usage(&self) -> Option<completion::Usage> {
        Some(completion::Usage {
            input_tokens: self.prompt_tokens as u64,
            output_tokens: self.completion_tokens as u64,
            total_tokens: self.total_tokens as u64,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        })
    }
}
