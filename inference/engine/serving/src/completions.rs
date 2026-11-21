//! vLLM's `/v1/completions`, for teacher-forced scoring only: the log-probability
//! of each prompt token under the model, and optionally the greedy next token.
//! The endpoint never generates freely and never ignores an option it cannot
//! honor.
use std::collections::BTreeMap;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::response::{IntoResponse, Response};
use magnitude_engine::error::RequestError;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::error::{ApiError, ErrorResponse, ServingError};
use crate::source::TokenizeInput;
use crate::tokenize::encode;
use crate::{Serving, unix_timestamp, with_request_id};

/// Scoring only. Every field outside these is refused.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    pub model: String,
    /// One prompt string; it is tokenized as `/tokenize`'s plain form with
    /// `add_special_tokens: true`.
    pub prompt: String,
    /// Required. 0 scores the prompt's own tokens; 1 also reports the top-1
    /// token at each position where it differs.
    #[schema(nullable = false)]
    pub prompt_logprobs: Option<u32>,
    /// 0 (default) or 1; 1 also returns the greedy next token.
    #[schema(nullable = false)]
    pub max_tokens: Option<u32>,
    /// Must be 0 when given, and is required when `max_tokens` is 1.
    #[schema(nullable = false)]
    pub temperature: Option<f64>,
    /// Must be false when given.
    #[schema(nullable = false)]
    pub echo: Option<bool>,
}

/// What a validated request asks for.
#[derive(Debug, PartialEq, Eq)]
struct ScoreRequest {
    model: String,
    prompt: String,
    top: bool,
    next: bool,
}

fn validate(request: CompletionRequest) -> Result<ScoreRequest, ApiError> {
    let CompletionRequest {
        model,
        prompt,
        prompt_logprobs,
        max_tokens,
        temperature,
        echo,
    } = request;
    if model.is_empty() {
        return Err(ApiError::invalid("model must not be empty").with_param("model"));
    }
    let top = match prompt_logprobs {
        Some(0) => false,
        Some(1) => true,
        Some(other) => {
            return Err(ApiError::invalid(format!(
                "prompt_logprobs must be 0 or 1, not {other}: this endpoint reports at most the top-1 token"
            ))
            .with_param("prompt_logprobs"));
        }
        None => {
            return Err(ApiError::invalid(
                "prompt_logprobs is required (0 or 1): this endpoint only scores prompts",
            )
            .with_param("prompt_logprobs"));
        }
    };
    let next = match max_tokens {
        None | Some(0) => false,
        Some(1) => true,
        Some(other) => {
            return Err(ApiError::invalid(format!(
                "max_tokens must be 0 or 1, not {other}: this endpoint never generates freely"
            ))
            .with_param("max_tokens"));
        }
    };
    match temperature {
        Some(0.0) => {}
        Some(other) => {
            return Err(ApiError::invalid(format!(
                "temperature must be 0, not {other}: the next token is greedy"
            ))
            .with_param("temperature"));
        }
        None if next => {
            return Err(ApiError::invalid("temperature: 0 is required with max_tokens: 1")
                .with_param("temperature"));
        }
        None => {}
    }
    if echo == Some(true) {
        return Err(ApiError::invalid("echo must be false: the prompt is not returned")
            .with_param("echo"));
    }
    Ok(ScoreRequest {
        model,
        prompt,
        top,
        next,
    })
}

/// A scored token: the real one, or the top-1 token at a position.
#[derive(Debug, Clone, PartialEq, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct PromptLogprob {
    pub logprob: f64,
    pub rank: u32,
    pub decoded_token: String,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CompletionChoice {
    pub index: u32,
    /// The greedy next token when `max_tokens` is 1, else empty.
    pub text: String,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: &'static str,
    /// Position 0 is null; each later position maps token ids, as decimal
    /// strings, to their scores.
    pub prompt_logprobs: Vec<Option<BTreeMap<String, PromptLogprob>>>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CompletionUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: CompletionUsage,
}

#[utoipa::path(post, path = "/v1/completions", operation_id = "createCompletion", tag = "chat",
    request_body = CompletionRequest,
    responses(
        (status = 200, description = "Teacher-forced prompt log-probabilities in vLLM's shape", body = CompletionResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Requested model is unavailable", body = ErrorResponse),
        (status = 409, description = "Runtime model cannot be admitted, or the source cannot score it", body = ErrorResponse),
        (status = 422, description = "Runtime target failed validation", body = ErrorResponse),
        (status = 500, description = "Runtime load or scoring failed", body = ErrorResponse),
        (status = 503, description = "Memory or request capacity is temporarily unavailable", body = ErrorResponse)
    )
)]
#[tracing::instrument(
    name = "serving.completions",
    skip_all,
    fields(completion.id = tracing::field::Empty, model.id = tracing::field::Empty),
    err(Debug)
)]
pub async fn completions(
    State(state): State<Serving>,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = payload.map_err(|error| ApiError::invalid(error.body_text()))?;
    let request = validate(request)?;
    let id = state.next_id("cmpl-icn-");
    let created = unix_timestamp();
    let span = tracing::Span::current();
    span.record("completion.id", id.as_str());
    span.record("model.id", request.model.as_str());

    let host = state
        .models
        .host(&request.model)
        .await
        .map_err(|error| ApiError::from(error).with_param("model"))?;
    let tokens = encode(
        &host,
        TokenizeInput::Text {
            text: request.prompt,
            add_special_tokens: true,
        },
    )
    .await?;
    let (required, limit) = (tokens.len() as u64, host.context_limit());
    if required > limit {
        // The refusal the worker gives an input over its context, so it reads as chat's does.
        return Err(ApiError::from(ServingError::Request(
            RequestError::ContextLengthExceeded { required, limit },
        ))
        .with_param("prompt"));
    }
    if tokens.is_empty() {
        return Err(ApiError::invalid("the prompt encodes to no tokens").with_param("prompt"));
    }
    let outcome = state
        .models
        .score(&request.model, tokens.clone(), request.top, request.next)
        .await
        .map_err(ApiError::from)?;
    if outcome.positions.len() + 1 != tokens.len() {
        return Err(ApiError::from(ServingError::Internal(format!(
            "scoring {} tokens returned {} positions",
            tokens.len(),
            outcome.positions.len()
        ))));
    }
    if request.next && outcome.next.is_none() {
        return Err(ApiError::from(ServingError::Internal(
            "scoring was asked for the next token and returned none".to_owned(),
        )));
    }

    let prompt_tokens = tokens.len() as u64;
    let decoded = tokio::task::spawn_blocking(move || {
        let mut prompt_logprobs = Vec::with_capacity(tokens.len());
        prompt_logprobs.push(None);
        for (position, real) in outcome.positions.iter().zip(&tokens[1..]) {
            let mut entry = BTreeMap::new();
            entry.insert(
                real.to_string(),
                PromptLogprob {
                    logprob: position.logprob,
                    rank: position.rank,
                    decoded_token: host.decode_token(*real)?,
                },
            );
            if let Some(top) = position.top {
                entry.insert(
                    top.token.to_string(),
                    PromptLogprob {
                        logprob: top.logprob,
                        rank: 1,
                        decoded_token: host.decode_token(top.token)?,
                    },
                );
            }
            prompt_logprobs.push(Some(entry));
        }
        let text = match outcome.next {
            Some(next) => host.decode_token(next)?,
            None => String::new(),
        };
        Ok::<_, ServingError>((prompt_logprobs, text))
    })
    .await
    .map_err(|error| ApiError::server(format!("decode task failed: {error}")))?
    .map_err(ApiError::from)?;
    let (prompt_logprobs, text) = decoded;

    let completion_tokens = u64::from(request.next);
    let response = Json(CompletionResponse {
        id: id.clone(),
        object: "text_completion",
        created,
        model: request.model,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            logprobs: None,
            finish_reason: "length",
            prompt_logprobs,
        }],
        usage: CompletionUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response();
    Ok(with_request_id(response, &id))
}
