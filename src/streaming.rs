//! Streaming types and the SSE stream parser for NVIDIA NIM chat completions.
//!
//! The internal `nvidia_stream` function consumes a
//! [`GenericEventSource`](rig_core::http_client::sse::GenericEventSource)
//! and yields [`RawStreamingChoice`]s that Rig uses to build a streaming
//! response for the caller.
//!
//! # Tool-call accumulation
//!
//! NVIDIA NIM sends tool-call arguments as incremental string fragments
//! across multiple SSE chunks. This module accumulates those fragments in
//! a `HashMap<usize, RawStreamingToolCall>` keyed by the tool-call index,
//! finalizing them when a `finish_reason: "tool_calls"` chunk arrives (or
//! when the stream ends).

use futures::StreamExt;
use rig_core::completion::{self, CompletionError, GetTokenUsage};
use rig_core::streaming::{self, RawStreamingChoice};
use serde::Deserialize;

use crate::json_utils;
use crate::message::NvidiaUsage;

/// A single SSE chunk from `/v1/chat/completions` with `stream: true`.
#[derive(Debug, Deserialize)]
pub(crate) struct NvidiaStreamChunk {
    pub choices: Vec<NvidiaStreamChoice>,
    pub usage: Option<NvidiaUsage>,
}

/// One choice inside a [`NvidiaStreamChunk`].
#[derive(Debug, Deserialize)]
pub(crate) struct NvidiaStreamChoice {
    pub delta: NvidiaDelta,
    pub finish_reason: Option<String>,
}

/// The incremental content delta for a streaming chunk.
#[derive(Debug, Deserialize, Default)]
pub(crate) struct NvidiaDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    pub tool_calls: Vec<NvidiaStreamToolCall>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

/// A tool-call fragment inside a streaming delta.
///
/// Fields are `Option` because NVIDIA sends them incrementally —
/// the first chunk may carry `id` and `name`, subsequent chunks carry
/// `arguments` fragments.
#[derive(Debug, Deserialize)]
pub(crate) struct NvidiaStreamToolCall {
    pub id: Option<String>,
    pub index: Option<usize>,
    pub function: NvidiaStreamFunction,
}

/// The function portion of a streaming tool-call delta.
#[derive(Debug, Deserialize)]
pub(crate) struct NvidiaStreamFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// Final streaming response carrying only the aggregated usage.
///
/// Emitted as the last [`RawStreamingChoice::FinalResponse`] in the stream.
#[derive(Clone, Debug, serde::Serialize, Deserialize)]
pub struct NvidiaStreamingResponse {
    /// Aggregated token usage for the entire streaming completion.
    pub usage: NvidiaUsage,
}

impl GetTokenUsage for NvidiaStreamingResponse {
    fn token_usage(&self) -> Option<completion::Usage> {
        self.usage.token_usage()
    }
}

/// Consume an SSE event source and yield [`RawStreamingChoice`] items.
///
/// Handles:
///
/// - Incremental text deltas → [`RawStreamingChoice::Message`]
/// - Reasoning deltas → [`RawStreamingChoice::ReasoningDelta`]
/// - Tool-call name/argument deltas → [`RawStreamingChoice::ToolCallDelta`]
/// - Finalized tool calls (on `finish_reason: "tool_calls"`) →
///   [`RawStreamingChoice::ToolCall`]
/// - Aggregated usage at stream end →
///   [`RawStreamingChoice::FinalResponse`]
pub(crate) fn nvidia_stream(
    mut event_source: rig_core::http_client::sse::GenericEventSource<reqwest::Client, Vec<u8>>,
) -> impl futures::Stream<Item = Result<RawStreamingChoice<NvidiaStreamingResponse>, CompletionError>>
+ Send {
    async_stream::stream! {
        let mut tool_calls: std::collections::HashMap<usize, streaming::RawStreamingToolCall> = std::collections::HashMap::new();
        let mut final_usage: Option<NvidiaUsage> = None;

        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(rig_core::http_client::sse::Event::Open) => {
                    continue;
                }
                Ok(rig_core::http_client::sse::Event::Message(message)) => {
                    if message.data.trim().is_empty() || message.data == "[DONE]" {
                        continue;
                    }

                    let chunk: NvidiaStreamChunk = match serde_json::from_str(&message.data) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };

                    if let Some(usage) = chunk.usage {
                        final_usage = Some(usage);
                    }

                    let Some(choice) = chunk.choices.into_iter().next() else {
                        continue;
                    };

                    for tc in choice.delta.tool_calls {
                        let index = tc.index.unwrap_or(0);
                        let existing = tool_calls
                            .entry(index)
                            .or_insert_with(streaming::RawStreamingToolCall::empty);

                        if let Some(id) = &tc.id
                            && !id.is_empty()
                        {
                            existing.id = id.clone();
                        }

                        if let Some(name) = &tc.function.name
                            && !name.is_empty()
                        {
                            existing.name = name.clone();
                            yield Ok(RawStreamingChoice::ToolCallDelta {
                                id: existing.id.clone(),
                                internal_call_id: existing.internal_call_id.clone(),
                                content: streaming::ToolCallDeltaContent::Name(name.clone()),
                            });
                        }

                        if let Some(arguments) = &tc.function.arguments
                            && !arguments.is_empty()
                        {
                            let current = match &existing.arguments {
                                serde_json::Value::Null => String::new(),
                                serde_json::Value::String(s) => {
                                    if s.trim() == "null" { String::new() } else { s.clone() }
                                }
                                v => v.to_string(),
                            };
                            let combined = format!("{current}{arguments}");
                            if combined.trim_start().starts_with('{') && combined.trim_end().ends_with('}') {
                                if let Ok(parsed) = serde_json::from_str(&combined) {
                                    existing.arguments = parsed;
                                } else {
                                    existing.arguments = serde_json::Value::String(combined);
                                }
                            } else {
                                existing.arguments = serde_json::Value::String(combined);
                            }
                            yield Ok(RawStreamingChoice::ToolCallDelta {
                                id: existing.id.clone(),
                                internal_call_id: existing.internal_call_id.clone(),
                                content: streaming::ToolCallDeltaContent::Delta(arguments.clone()),
                            });
                        }
                    }

                    if let Some(reasoning) = choice.delta.reasoning_content
                        && !reasoning.is_empty()
                    {
                        yield Ok(RawStreamingChoice::ReasoningDelta {
                            id: None,
                            reasoning,
                        });
                    }

                    if let Some(content) = choice.delta.content
                        && !content.is_empty()
                    {
                        yield Ok(RawStreamingChoice::Message(content));
                    }

                    if choice.finish_reason.as_deref() == Some("tool_calls") {
                        let mut pending: Vec<_> = tool_calls.drain().collect();
                        pending.sort_by_key(|(k, _)| *k);
                        for (_, tc) in pending {
                            let finalized = finalize_tool_call(tc);
                            yield Ok(RawStreamingChoice::ToolCall(finalized));
                        }
                    }
                }
                Err(rig_core::http_client::Error::StreamEnded) => {
                    break;
                }
                Err(error) => {
                    yield Err(CompletionError::ProviderError(error.to_string()));
                    break;
                }
            }
        }

        event_source.close();

        let mut remaining: Vec<_> = tool_calls.drain().collect();
        remaining.sort_by_key(|(k, _)| *k);
        for (_, tc) in remaining {
            let finalized = finalize_tool_call(tc);
            yield Ok(RawStreamingChoice::ToolCall(finalized));
        }

        let usage = final_usage.unwrap_or(NvidiaUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        });
        yield Ok(RawStreamingChoice::FinalResponse(
            NvidiaStreamingResponse { usage },
        ));
    }
}

/// Ensure a partially accumulated tool call is in a valid state.
///
/// - If `arguments` is still `Null`, replace it with an empty object `{}`.
/// - If `name` is empty, replace it with `"unknown"`.
pub(crate) fn finalize_tool_call(
    mut tc: streaming::RawStreamingToolCall,
) -> streaming::RawStreamingToolCall {
    if tc.arguments.is_null() {
        tc.arguments = serde_json::Value::Object(serde_json::Map::new());
    }
    if tc.name.is_empty() {
        tc.name = "unknown".to_string();
    }
    tc
}
