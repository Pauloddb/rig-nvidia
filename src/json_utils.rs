//! JSON serialization/deserialization utilities.
//!
//! Reimplementations of `rig_core::json_utils` (which is `pub(crate)` there)
//! needed for NVIDIA's API quirks:
//!
//! - **`stringified_json`**: NVIDIA sends tool-call arguments as a JSON
//!   string inside a JSON field (e.g. `"arguments": "{\"key\": \"value\"}"`)
//!   instead of an inline object.
//! - **`null_or_vec`**: Some NVIDIA endpoints return `null` where an empty
//!   array would be expected.
//! - **`merge`**: Shallow merge of two JSON objects, used to inject stream
//!   parameters into `additional_params`.

use serde::Serializer;
use serde::de::{self, Deserialize, Deserializer, SeqAccess, Visitor};
use std::fmt;
use std::marker::PhantomData;

/// Shallow-merge two JSON values.
///
/// If both `a` and `b` are objects, keys from `b` overwrite keys in `a`.
/// If `a` is not an object (or `b` is not an object), `a` is returned unchanged.
///
/// # Examples
///
/// ```
/// use rig_nvidia::json_utils;
///
/// let a = serde_json::json!({"x": 1, "y": 2});
/// let b = serde_json::json!({"y": 3, "z": 4});
/// let merged = json_utils::merge(a, b);
/// assert_eq!(merged["y"], 3);
/// assert_eq!(merged["z"], 4);
/// ```
pub fn merge(a: serde_json::Value, b: serde_json::Value) -> serde_json::Value {
    match (a, b) {
        (serde_json::Value::Object(mut a_map), serde_json::Value::Object(b_map)) => {
            b_map.into_iter().for_each(|(key, value)| {
                a_map.insert(key, value);
            });
            serde_json::Value::Object(a_map)
        }
        (a, _) => a,
    }
}

/// Serde adapter for fields that contain a **stringified JSON** value.
///
/// NVIDIA NIM sends tool-call `arguments` as a JSON-encoded string rather
/// than an inline object. This module serializes a `serde_json::Value` to
/// its string representation and deserializes it back.
///
/// # Usage
///
/// ```ignore
/// #[derive(Serialize, Deserialize)]
/// struct NvidiaFunction {
///     #[serde(with = "json_utils::stringified_json")]
///     arguments: serde_json::Value,
/// }
/// ```
pub mod stringified_json {
    use super::*;

    /// Serialize a `serde_json::Value` as a JSON string.
    pub fn serialize<S>(value: &serde_json::Value, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = value.to_string();
        serializer.serialize_str(&s)
    }

    /// Deserialize a JSON string back into a `serde_json::Value`.
    ///
    /// Empty strings are deserialized as an empty object `{}`.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<serde_json::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.trim().is_empty() {
            return Ok(serde_json::Value::Object(serde_json::Map::new()));
        }
        serde_json::from_str(&s).map_err(serde::de::Error::custom)
    }
}

/// Serde deserializer that accepts both a JSON array **and** `null`.
///
/// NVIDIA endpoints sometimes return `null` where an empty array is expected
/// (e.g. `tool_calls: null` instead of `tool_calls: []`). This function
/// deserializes `null` / `None` / unit as `Vec::default()`.
///
/// # Usage
///
/// ```ignore
/// #[derive(Deserialize)]
/// struct NvidiaAssistant {
///     #[serde(default, deserialize_with = "json_utils::null_or_vec")]
///     tool_calls: Vec<NvidiaToolCall>,
/// }
/// ```
pub fn null_or_vec<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    struct NullOrVec<T>(PhantomData<fn() -> T>);

    impl<'de, T> Visitor<'de> for NullOrVec<T>
    where
        T: Deserialize<'de>,
    {
        type Value = Vec<T>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a sequence or null")
        }

        fn visit_seq<A>(self, seq: A) -> Result<Vec<T>, A::Error>
        where
            A: SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }

        fn visit_none<E>(self) -> Result<Vec<T>, E>
        where
            E: de::Error,
        {
            Ok(vec![])
        }

        fn visit_unit<E>(self) -> Result<Vec<T>, E>
        where
            E: de::Error,
        {
            Ok(vec![])
        }
    }

    deserializer.deserialize_any(NullOrVec(PhantomData))
}
