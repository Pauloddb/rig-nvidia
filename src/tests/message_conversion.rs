use crate::message::{NvidiaMessage, NvidiaToolCall, NvidiaToolType, convert_message};
use rig_core::{OneOrMany, message};

#[test]
fn test_system_message() {
    let msg = message::Message::system("You are helpful.");
    let nvidia_msgs: Vec<NvidiaMessage> = convert_message(msg).unwrap();
    assert_eq!(nvidia_msgs.len(), 1);
    assert!(
        matches!(&nvidia_msgs[0], NvidiaMessage::System { content } if content == "You are helpful.")
    );
}

#[test]
fn test_user_message() {
    let msg = message::Message::user("Hello");
    let nvidia_msgs: Vec<NvidiaMessage> = convert_message(msg).unwrap();
    assert!(
        nvidia_msgs
            .iter()
            .any(|m| matches!(m, NvidiaMessage::User { content } if content == "Hello"))
    );
}

#[test]
fn test_assistant_with_tool_call() {
    let msg = message::Message::Assistant {
        id: None,
        content: OneOrMany::many(vec![
            message::AssistantContent::text("I'll call a tool"),
            message::AssistantContent::tool_call("call_1", "my_func", serde_json::json!({"x": 1})),
        ])
        .unwrap(),
    };
    let nvidia_msgs: Vec<NvidiaMessage> = convert_message(msg).unwrap();
    assert_eq!(nvidia_msgs.len(), 1);
    match &nvidia_msgs[0] {
        NvidiaMessage::Assistant {
            content,
            tool_calls,
        } => {
            assert_eq!(content.as_deref(), Some("I'll call a tool"));
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].function.name, "my_func");
        }
        _ => panic!("Expected assistant message"),
    }
}

#[test]
fn test_assistant_text_only() {
    let msg = message::Message::Assistant {
        id: None,
        content: OneOrMany::one(message::AssistantContent::text("Just text")),
    };
    let nvidia_msgs: Vec<NvidiaMessage> = convert_message(msg).unwrap();
    assert_eq!(nvidia_msgs.len(), 1);
    match &nvidia_msgs[0] {
        NvidiaMessage::Assistant {
            content,
            tool_calls,
        } => {
            assert_eq!(content.as_deref(), Some("Just text"));
            assert!(tool_calls.is_empty());
        }
        _ => panic!("Expected assistant message"),
    }
}

#[test]
fn test_tool_result_message() {
    let tool_result = message::ToolResult {
        id: "call_42".to_string(),
        call_id: None,
        content: OneOrMany::one(message::ToolResultContent::Text(message::Text {
            text: "{\"temp\": 22}".to_string(),
            additional_params: None,
        })),
    };
    let msg = message::Message::User {
        content: OneOrMany::one(message::UserContent::ToolResult(tool_result)),
    };
    let nvidia_msgs: Vec<NvidiaMessage> = convert_message(msg).unwrap();
    assert_eq!(nvidia_msgs.len(), 1);
    match &nvidia_msgs[0] {
        NvidiaMessage::ToolResult {
            tool_call_id,
            content,
        } => {
            assert_eq!(tool_call_id, "call_42");
            assert_eq!(content, "{\"temp\": 22}");
        }
        _ => panic!("Expected ToolResult message"),
    }
}

#[test]
fn test_tool_call_conversion() {
    let tool_call = message::ToolCall {
        id: "tc_1".to_string(),
        call_id: None,
        function: message::ToolFunction {
            name: "search".to_string(),
            arguments: serde_json::json!({"q": "rust"}),
        },
        signature: None,
        additional_params: None,
    };
    let nvidia_tc: NvidiaToolCall = tool_call.into();
    assert_eq!(nvidia_tc.id, "tc_1");
    assert_eq!(nvidia_tc.function.name, "search");
    assert_eq!(nvidia_tc.r#type, NvidiaToolType::Function);
}

#[test]
fn test_mixed_user_content() {
    let tool_result = message::ToolResult {
        id: "call_1".to_string(),
        call_id: None,
        content: OneOrMany::one(message::ToolResultContent::Text(message::Text {
            text: "result".to_string(),
            additional_params: None,
        })),
    };
    let msg = message::Message::User {
        content: OneOrMany::many(vec![
            message::UserContent::ToolResult(tool_result),
            message::UserContent::Text(message::Text {
                text: "Follow-up question".to_string(),
                additional_params: None,
            }),
        ])
        .unwrap(),
    };
    let nvidia_msgs: Vec<NvidiaMessage> = convert_message(msg).unwrap();
    assert!(nvidia_msgs.len() >= 2);
    let has_tool_result = nvidia_msgs
        .iter()
        .any(|m| matches!(m, NvidiaMessage::ToolResult { .. }));
    let has_user_text = nvidia_msgs
        .iter()
        .any(|m| matches!(m, NvidiaMessage::User { content } if content.contains("Follow-up")));
    assert!(has_tool_result);
    assert!(has_user_text);
}
