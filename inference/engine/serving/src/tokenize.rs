//! vLLM's `/tokenize`: the token ids a prompt or a chat conversation encodes
//! to, with the tokenizer and chat template generation uses. Host-only: the
//! model is never leased or loaded.
use std::collections::BTreeMap;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use utoipa::ToSchema;

use crate::chat::{ApplyTemplateRequest, ChatMessageRequest, validate_apply_template_request};
use crate::error::{ApiError, ErrorResponse};
use crate::source::{HostChat, TokenizeInput};
use crate::Serving;

/// Either `prompt` (plain) or `messages` (chat), never both.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct TokenizeRequest {
    pub model: String,
    /// Plain form: the text to encode. Special-token text is encoded as
    /// special tokens.
    #[schema(nullable = false)]
    pub prompt: Option<String>,
    /// Chat form: the conversation, rendered with the model's chat template.
    #[schema(nullable = false)]
    pub messages: Option<Vec<ChatMessageRequest>>,
    /// Chat form only; defaults to true.
    #[schema(nullable = false)]
    pub add_generation_prompt: Option<bool>,
    /// Whether the tokenizer adds its own special tokens. Defaults to true for
    /// the plain form and false for the chat form.
    #[schema(nullable = false)]
    pub add_special_tokens: Option<bool>,
    /// Chat form only.
    #[schema(nullable = false)]
    pub chat_template_kwargs: Option<BTreeMap<String, JsonValue>>,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct TokenizeResponse {
    pub count: u64,
    pub max_model_len: u64,
    pub tokens: Vec<u32>,
}

/// The request as the host takes it, refusing every field the chosen form
/// does not use.
pub(crate) fn validate_tokenize_request(
    request: TokenizeRequest,
) -> Result<TokenizeInput, ApiError> {
    let TokenizeRequest {
        model,
        prompt,
        messages,
        add_generation_prompt,
        add_special_tokens,
        chat_template_kwargs,
    } = request;
    match (prompt, messages) {
        (Some(_), Some(_)) => Err(ApiError::invalid(
            "prompt and messages are exclusive: send the plain form or the chat form",
        )
        .with_param("messages")),
        (None, None) => Err(ApiError::invalid("one of prompt and messages is required")
            .with_param("prompt")),
        (Some(text), None) => {
            if add_generation_prompt.is_some() {
                return Err(ApiError::invalid(
                    "add_generation_prompt applies only to the chat form (messages)",
                )
                .with_param("add_generation_prompt"));
            }
            if chat_template_kwargs.is_some() {
                return Err(ApiError::invalid(
                    "chat_template_kwargs applies only to the chat form (messages)",
                )
                .with_param("chat_template_kwargs"));
            }
            Ok(TokenizeInput::Text {
                text,
                add_special_tokens: add_special_tokens.unwrap_or(true),
            })
        }
        (None, Some(messages)) => {
            // The one message parser: the same validation `/api/v1/chat/templates/apply` runs.
            let input = validate_apply_template_request(ApplyTemplateRequest {
                model: Some(model),
                messages,
                tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                response_format: None,
                chat_template_kwargs,
            })?;
            Ok(TokenizeInput::Chat {
                input,
                add_generation_prompt: add_generation_prompt.unwrap_or(true),
                add_special_tokens: add_special_tokens.unwrap_or(false),
            })
        }
    }
}

/// The host's tokens for `input`, off the async threads.
pub(crate) async fn encode(
    host: &std::sync::Arc<dyn HostChat>,
    input: TokenizeInput,
) -> Result<Vec<u32>, ApiError> {
    let host = host.clone();
    tokio::task::spawn_blocking(move || host.tokenize(&input))
        .await
        .map_err(|error| ApiError::server(format!("tokenize task failed: {error}")))?
        .map_err(ApiError::from)
}

#[utoipa::path(post, path = "/tokenize", operation_id = "tokenize", tag = "chat",
    request_body = TokenizeRequest,
    responses(
        (status = 200, description = "The token ids the prompt or conversation encodes to", body = TokenizeResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Requested model is unavailable", body = ErrorResponse),
        (status = 409, description = "Installed model cannot be resolved for serving", body = ErrorResponse),
        (status = 422, description = "Installed model failed integrity validation", body = ErrorResponse),
        (status = 500, description = "Tokenization failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(
    name = "serving.tokenize",
    skip_all,
    fields(model.id = tracing::field::Empty),
    err(Debug)
)]
pub async fn tokenize(
    State(state): State<Serving>,
    payload: Result<Json<TokenizeRequest>, JsonRejection>,
) -> Result<Json<TokenizeResponse>, ApiError> {
    let Json(request) = payload.map_err(|error| ApiError::invalid(error.body_text()))?;
    if request.model.is_empty() {
        return Err(ApiError::invalid("model must not be empty").with_param("model"));
    }
    let model = request.model.clone();
    tracing::Span::current().record("model.id", model.as_str());
    let input = validate_tokenize_request(request)?;
    let host = state
        .models
        .host(&model)
        .await
        .map_err(|error| ApiError::from(error).with_param("model"))?;
    let tokens = encode(&host, input).await?;
    Ok(Json(TokenizeResponse {
        count: tokens.len() as u64,
        max_model_len: host.context_limit(),
        tokens,
    }))
}
