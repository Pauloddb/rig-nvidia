use crate::json_utils;
use serde::{Deserialize, Serialize};

#[test]
fn test_null_or_vec_null() {
    #[derive(Deserialize)]
    struct Wrapper {
        #[serde(default, deserialize_with = "json_utils::null_or_vec")]
        items: Vec<String>,
    }

    let w: Wrapper = serde_json::from_str(r#"{"items": null}"#).unwrap();
    assert!(w.items.is_empty());
}

#[test]
fn test_null_or_vec_array() {
    #[derive(Deserialize)]
    struct Wrapper {
        #[serde(default, deserialize_with = "json_utils::null_or_vec")]
        items: Vec<String>,
    }

    let w: Wrapper = serde_json::from_str(r#"{"items": ["a", "b"]}"#).unwrap();
    assert_eq!(w.items, vec!["a", "b"]);
}

#[test]
fn test_null_or_vec_default() {
    #[derive(Deserialize, Default)]
    struct Wrapper {
        #[serde(default, deserialize_with = "json_utils::null_or_vec")]
        items: Vec<String>,
    }

    let w: Wrapper = serde_json::from_str(r#"{}"#).unwrap();
    assert!(w.items.is_empty());
}

#[test]
fn test_stringified_json_roundtrip() {
    #[derive(Serialize, Deserialize)]
    struct Wrapper {
        #[serde(with = "json_utils::stringified_json")]
        data: serde_json::Value,
    }

    let original = Wrapper {
        data: serde_json::json!({"key": "value"}),
    };
    let serialized = serde_json::to_string(&original).unwrap();
    let deserialized: Wrapper = serde_json::from_str(&serialized).unwrap();
    assert_eq!(original.data, deserialized.data);
}

#[test]
fn test_stringified_json_empty() {
    #[derive(Serialize, Deserialize)]
    struct Wrapper {
        #[serde(with = "json_utils::stringified_json")]
        data: serde_json::Value,
    }

    let w: Wrapper = serde_json::from_str(r#"{"data": ""}"#).unwrap();
    assert!(w.data.is_object());
    assert!(w.data.as_object().unwrap().is_empty());
}

#[test]
fn test_merge_objects() {
    let a = serde_json::json!({"x": 1, "y": 2});
    let b = serde_json::json!({"y": 3, "z": 4});
    let merged = json_utils::merge(a, b);
    assert_eq!(merged["x"], 1);
    assert_eq!(merged["y"], 3);
    assert_eq!(merged["z"], 4);
}

#[test]
fn test_merge_non_object() {
    let a = serde_json::json!(42);
    let b = serde_json::json!({"y": 3});
    let merged = json_utils::merge(a, b);
    assert_eq!(merged, 42);
}

#[test]
fn test_merge_with_null() {
    let a = serde_json::json!({"x": 1});
    let b = serde_json::Value::Null;
    let merged = json_utils::merge(a, b);
    assert_eq!(merged, serde_json::json!({"x": 1}));
}

#[test]
fn test_stringified_json_complex() {
    #[derive(Serialize, Deserialize)]
    struct Wrapper {
        #[serde(with = "json_utils::stringified_json")]
        data: serde_json::Value,
    }

    let original = Wrapper {
        data: serde_json::json!({
            "nested": {"arr": [1, 2, 3]},
            "bool": true,
            "null": null
        }),
    };
    let serialized = serde_json::to_string(&original).unwrap();
    let deserialized: Wrapper = serde_json::from_str(&serialized).unwrap();
    assert_eq!(original.data, deserialized.data);
}
