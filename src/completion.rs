//! Chat completion model, request/response types, and the [`CompletionModel`] impl.
//!
//! This module contains everything needed to send chat-completion requests to
//! the NVIDIA NIM API (`/v1/chat/completions`) and parse the responses, both
//! non-streaming and streaming.

use std::sync::Arc;

use rig_core::completion::{self, CompletionError, CompletionModel, CompletionRequest};
use rig_core::message;
use rig_core::{OneOrMany, streaming::StreamingCompletionResponse};
use serde::{Deserialize, Serialize};

use crate::json_utils;
use crate::message::{NvidiaMessage, NvidiaToolDefinition, NvidiaUsage, convert_message};

const AUTH_HEADER: &str = "Authorization";

/// NVIDIA NIM chat-completion model.
///
/// Implements Rig's [`CompletionModel`] trait, so it can be used directly
/// with [`rig_core::agent::AgentBuilder`].
///
/// Typically you create this indirectly via [`NvidiaClient::agent`].
///
/// [`NvidiaClient::agent`]: crate::client::NvidiaClient::agent
#[derive(Clone, Debug)]
pub struct NvidiaCompletionModel {
    pub(crate) client: Arc<reqwest::Client>,
    pub(crate) api_key: Option<String>,
    /// Model identifier (e.g. `"nvidia/nemotron-3-super-120b-a12b"`).
    pub model: String,
    pub(crate) base_url: String,
}

/// The request body sent to `/v1/chat/completions`.
///
/// This is an internal type; it is constructed from a [`CompletionRequest`]
/// via [`TryFrom`].
#[derive(Debug, Serialize)]
pub(crate) struct NvidiaChatRequest {
    pub model: String,
    pub messages: Vec<NvidiaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<serde_json::Value>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

/// Response body from `/v1/chat/completions` (non-streaming).
///
/// Contains the model's choices and optional usage statistics.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct NvidiaCompletionResponse {
    /// Response ID assigned by the API.
    pub id: Option<String>,
    /// Model that generated the response.
    pub model: Option<String>,
    /// List of completion choices (typically one).
    pub choices: Vec<NvidiaChoice>,
    /// Token usage statistics (present when `stream_options.include_usage` is used).
    pub usage: Option<NvidiaUsage>,
}

/// A single completion choice inside a [`NvidiaCompletionResponse`].
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct NvidiaChoice {
    /// Index of this choice within the `choices` array.
    pub index: usize,
    /// The assistant message (text and/or tool calls).
    pub(crate) message: NvidiaMessage,
    /// Reason the model stopped generating (e.g. `"stop"`, `"tool_calls"`).
    pub finish_reason: Option<String>,
}

impl TryFrom<NvidiaCompletionResponse>
    for completion::CompletionResponse<NvidiaCompletionResponse>
{
    type Error = CompletionError;

    fn try_from(response: NvidiaCompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let content: Vec<completion::AssistantContent> = match &choice.message {
            NvidiaMessage::Assistant {
                content,
                tool_calls,
            } => {
                let mut items: Vec<completion::AssistantContent> = Vec::new();

                if let Some(text) = content
                    && !text.trim().is_empty()
                {
                    items.push(completion::AssistantContent::text(text));
                }

                items.extend(
                    tool_calls
                        .iter()
                        .map(|call| {
                            completion::AssistantContent::tool_call(
                                &call.id,
                                &call.function.name,
                                call.function.arguments.clone(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );

                Ok(items)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain an assistant message".into(),
            )),
        }?;

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = response
            .usage
            .as_ref()
            .map(|u| completion::Usage {
                input_tokens: u.prompt_tokens as u64,
                output_tokens: u.completion_tokens as u64,
                total_tokens: u.total_tokens as u64,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
            message_id: None,
        })
    }
}

impl TryFrom<CompletionRequest> for NvidiaChatRequest {
    type Error = CompletionError;

    fn try_from(req: CompletionRequest) -> Result<Self, Self::Error> {
        let model = req.model.clone().unwrap_or_default();

        let mut full_history: Vec<NvidiaMessage> = match &req.preamble {
            Some(preamble) => vec![NvidiaMessage::System {
                content: preamble.clone(),
            }],
            None => vec![],
        };

        if let Some(docs) = req.normalized_documents() {
            let docs: Vec<NvidiaMessage> = convert_message(docs)?;
            full_history.extend(docs);
        }

        let chat_history: Vec<NvidiaMessage> = req
            .chat_history
            .into_iter()
            .map(convert_message)
            .collect::<Result<Vec<Vec<NvidiaMessage>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        full_history.extend(chat_history);

        let mut additional_params_payload =
            req.additional_params.unwrap_or(serde_json::Value::Null);
        let mut additional_tools =
            extract_tools_from_additional_params(&mut additional_params_payload);

        let mut tools: Vec<serde_json::Value> = req
            .tools
            .into_iter()
            .map(NvidiaToolDefinition::from)
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()?;
        tools.append(&mut additional_tools);

        let additional_params = if additional_params_payload.is_null() {
            None
        } else {
            Some(additional_params_payload)
        };

        let tool_choice = req.tool_choice.map(|tc| match tc {
            message::ToolChoice::Auto => serde_json::json!("auto"),
            message::ToolChoice::None => serde_json::json!("none"),
            message::ToolChoice::Required => serde_json::json!("required"),
            message::ToolChoice::Specific { function_names } => {
                if function_names.len() == 1 {
                    serde_json::json!({
                        "type": "function",
                        "function": { "name": function_names[0] }
                    })
                } else {
                    serde_json::json!("auto")
                }
            }
        });

        Ok(Self {
            model,
            messages: full_history,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            tools,
            tool_choice,
            stop: None,
            additional_params,
        })
    }
}

/// Extract a `"tools"` key from `additional_params` (if present) and remove it.
///
/// This allows callers to inject extra tools via `additional_params` without
/// conflicting with the tools already declared in the [`CompletionRequest`].
fn extract_tools_from_additional_params(
    additional_params: &mut serde_json::Value,
) -> Vec<serde_json::Value> {
    if let Some(map) = additional_params.as_object_mut()
        && let Some(raw_tools) = map.remove("tools")
    {
        return serde_json::from_value::<Vec<serde_json::Value>>(raw_tools).unwrap_or_default();
    }
    Vec::new()
}

/// Map a [`reqwest::Error`] into a [`CompletionError::HttpError`].
pub(crate) fn map_reqwest_error(e: reqwest::Error) -> CompletionError {
    CompletionError::HttpError(rig_core::http_client::Error::Instance(Box::new(e)))
}

/// Inject the `Authorization: Bearer <key>` header into an HTTP request.
fn with_auth_header(
    mut req: http::Request<Vec<u8>>,
    api_key: Option<&str>,
) -> Result<http::Request<Vec<u8>>, CompletionError> {
    if let Some(key) = api_key {
        let auth_value = http::HeaderValue::from_str(&format!("Bearer {}", key))
            .map_err(|e| CompletionError::HttpError(e.into()))?;
        req.headers_mut()
            .insert(http::header::AUTHORIZATION, auth_value);
    }
    Ok(req)
}

impl CompletionModel for NvidiaCompletionModel {
    type Response = NvidiaCompletionResponse;
    type StreamingResponse = crate::streaming::NvidiaStreamingResponse;
    type Client = crate::client::NvidiaClient;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self {
            client: Arc::clone(&client.http_client),
            api_key: client.api_key.clone(),
            model: model.into(),
            base_url: client.base_url.clone(),
        }
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<NvidiaCompletionResponse>, CompletionError> {
        let nvidia_request = NvidiaChatRequest::try_from(request)?;

        let mut http_request = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .json(&nvidia_request);

        if let Some(ref key) = self.api_key {
            http_request = http_request.header(AUTH_HEADER, format!("Bearer {}", key));
        }

        let response = http_request.send().await.map_err(map_reqwest_error)?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(CompletionError::ProviderError(format!(
                "NVIDIA NIM error {}: {}",
                status, body
            )));
        }

        let chat_response: NvidiaCompletionResponse = response.json().await.map_err(|e| {
            CompletionError::ResponseError(format!("Failed to parse NVIDIA response: {}", e))
        })?;

        chat_response.try_into()
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let mut nvidia_request = NvidiaChatRequest::try_from(request)?;

        let params = json_utils::merge(
            nvidia_request
                .additional_params
                .unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream": true, "stream_options": {"include_usage": true}}),
        );
        nvidia_request.additional_params = Some(params);

        let body = serde_json::to_vec(&nvidia_request)?;
        let http_req = http::Request::builder()
            .method("POST")
            .uri(format!("{}/chat/completions", self.base_url))
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        let event_source = rig_core::http_client::sse::GenericEventSource::new(
            (*self.client).clone(),
            with_auth_header(http_req, self.api_key.as_deref())?,
        );

        let stream = crate::streaming::nvidia_stream(event_source);

        Ok(StreamingCompletionResponse::stream(Box::pin(stream)))
    }
}
