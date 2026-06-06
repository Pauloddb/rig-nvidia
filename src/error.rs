//! Error types for the `rig-nvidia` crate.

/// Errors that can occur when interacting with the NVIDIA NIM API.
#[derive(thiserror::Error, Debug)]
pub enum NvidiaError {
    /// An HTTP request failed (network error, timeout, etc.).
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    /// The NVIDIA API returned a non-success status code.
    #[error("NVIDIA API error (status {status}): {message}")]
    ApiError {
        /// HTTP status code returned by the API.
        status: u16,
        /// Error message returned by the API.
        message: String,
    },

    /// Failed to parse a response body into the expected type.
    #[error("Failed to parse response: {0}")]
    ParseError(String),

    /// A required environment variable was not set.
    #[error("Environment variable {0} not set")]
    EnvVarError(String),

    /// An I/O error occurred.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
