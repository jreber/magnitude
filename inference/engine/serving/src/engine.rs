//! Serving over the engine library: a bound engine invocation generates
//! through the host chat session, and a model's host artifacts answer the
//! host-only operations. Model sources (the standalone engine, the service)
//! compose these into their [`ServedModels`](crate::ServedModels).
use crate::error::{ModelUnavailable, ServingError};
use crate::source::{GenerationEvent, GenerationStream, HostChat, ModelInvocation, TokenizeInput};
use magnitude_chat::{ChatInput, GenerationRequest};
use magnitude_engine::chat::{
    self, AppliedTemplate, ModelProperties, SessionError, SessionEvent, SessionLimits,
};
use magnitude_engine::host::HostArtifacts;
use magnitude_engine::inputs::{SpecialTokens, TokenId};
use magnitude_engine::invocation::{Invocation, InvocationError};
use magnitude_chat::ByteBpeTokenizer;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Event buffering between a session and its protocol framing.
const EVENT_CAPACITY: usize = 32;

impl From<SessionError> for ServingError {
    fn from(error: SessionError) -> Self {
        match error {
            SessionError::Chat(error) => Self::Chat(error),
            SessionError::Request(error) => Self::Request(error),
        }
    }
}

impl From<InvocationError> for ServingError {
    fn from(error: InvocationError) -> Self {
        match error {
            InvocationError::UnknownModel { .. } | InvocationError::NotInstalled { .. } => {
                Self::Model(ModelUnavailable::NotFound(error.to_string()))
            }
            InvocationError::Load(error) => Self::Load(error),
            InvocationError::Unavailable(error) => Self::Request(error),
        }
    }
}

/// One engine invocation: its host artifacts, worker client and release
/// guard. The worker client holds the guard through request retirement.
pub struct EngineInvocation {
    invocation: Invocation,
    limits: SessionLimits,
}

impl EngineInvocation {
    pub fn new(invocation: Invocation, limits: SessionLimits) -> Self {
        Self { invocation, limits }
    }
}

impl ModelInvocation for EngineInvocation {
    fn generate(self: Box<Self>, request: GenerationRequest) -> GenerationStream {
        let (session_sender, mut session_events) = mpsc::channel(EVENT_CAPACITY);
        let (sender, receiver) = mpsc::channel(EVENT_CAPACITY);
        tokio::spawn(chat::generate(self.invocation, request, self.limits, session_sender));
        tokio::spawn(async move {
            loop {
                let event = tokio::select! {
                    event = session_events.recv() => event,
                    () = sender.closed() => return,
                };
                let Some(event) = event else { return };
                let event = match event {
                    SessionEvent::Progress(progress) => GenerationEvent::Progress(progress),
                    SessionEvent::Admitted { prompt_tokens } => {
                        GenerationEvent::Admitted { prompt_tokens }
                    }
                    SessionEvent::Output { event, snapshot } => {
                        GenerationEvent::Output { event, snapshot }
                    }
                    SessionEvent::Completed(completion) => GenerationEvent::Completed(completion),
                    SessionEvent::Failed(error) => GenerationEvent::Failed(error.into()),
                };
                if sender.send(event).await.is_err() {
                    return;
                }
            }
        });
        GenerationStream::new(receiver)
    }
}

/// A model's host artifacts serving the host-only operations.
pub struct EngineHost {
    host: Arc<HostArtifacts>,
}

impl EngineHost {
    pub fn new(host: Arc<HostArtifacts>) -> Self {
        Self { host }
    }
}

impl HostChat for EngineHost {
    fn count(&self, input: &ChatInput) -> Result<u64, ServingError> {
        Ok(chat::count_tokens(&self.host, input)?)
    }

    fn apply_template(&self, input: &ChatInput) -> Result<AppliedTemplate, ServingError> {
        Ok(chat::apply_template(&self.host, input)?)
    }

    fn properties(&self) -> Result<ModelProperties, ServingError> {
        Ok(chat::model_properties(&self.host)?)
    }

    fn tokenize(&self, input: &TokenizeInput) -> Result<Vec<u32>, ServingError> {
        let (text, add_special_tokens) = match input {
            TokenizeInput::Text {
                text,
                add_special_tokens,
            } => (text.clone(), *add_special_tokens),
            TokenizeInput::Chat {
                input,
                add_generation_prompt,
                add_special_tokens,
            } => {
                // The rendering is exactly what template application returns.
                let applied = chat::apply_template(&self.host, input)?;
                (
                    chat_text(applied, *add_generation_prompt)?,
                    *add_special_tokens,
                )
            }
        };
        encode_text(self.host.tokenizer(), &text, add_special_tokens)
    }

    fn decode_token(&self, token: u32) -> Result<String, ServingError> {
        let piece = self
            .host
            .tokenizer()
            .piece(TokenId(token), false)
            .map_err(ServingError::InvalidRequest)?;
        Ok(String::from_utf8_lossy(piece).into_owned())
    }

    fn context_limit(&self) -> u64 {
        self.host.definition().decoder.context_limit
    }
}

/// The text a chat tokenizes: the template's prompt, without its generation
/// prompt when the caller did not ask for one.
fn chat_text(applied: AppliedTemplate, add_generation_prompt: bool) -> Result<String, ServingError> {
    if add_generation_prompt {
        return Ok(applied.prompt);
    }
    applied
        .prompt
        .strip_suffix(applied.generation_prompt.as_str())
        .map(str::to_owned)
        .ok_or_else(|| {
            ServingError::Internal(
                "the rendered prompt does not end with the template's generation prompt".to_owned(),
            )
        })
}

/// `text` as token ids. Special-token text is encoded as special tokens.
/// `add_special_tokens` adds what the tokenizer inserts into every model input
/// (its start-of-sequence token, once), exactly as generation does; without it
/// the ids are only what the text encodes to.
fn encode_text(
    tokenizer: &ByteBpeTokenizer,
    text: &str,
    add_special_tokens: bool,
) -> Result<Vec<u32>, ServingError> {
    let tokens = if add_special_tokens {
        tokenizer.encode_sequence(text).map(|sequence| sequence.tokens)
    } else {
        tokenizer.encode(text, SpecialTokens::Recognize)
    }
    .map_err(ServingError::Internal)?;
    Ok(tokens.into_iter().map(|token| token.0).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{ApplyTemplateRequest, validate_apply_template_request};
    use magnitude_engine::options::{PackageOptions, ProjectorSelection};
    use serde_json::json;

    fn applied(prompt: &str, generation_prompt: &str) -> AppliedTemplate {
        AppliedTemplate {
            prompt: prompt.to_owned(),
            generation_prompt: generation_prompt.to_owned(),
            grammar: String::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            supports_thinking: false,
            thinking_start_tag: None,
            thinking_end_tag: None,
            template_fingerprint: String::new(),
        }
    }

    #[test]
    fn a_chat_keeps_its_generation_prompt_only_when_asked() {
        let template = applied("<u>hi<a>", "<a>");
        assert_eq!(chat_text(template.clone(), true).unwrap(), "<u>hi<a>");
        assert_eq!(chat_text(template, false).unwrap(), "<u>hi");
    }

    #[test]
    fn a_template_without_a_generation_prompt_has_nothing_to_drop() {
        assert_eq!(chat_text(applied("<u>hi", ""), false).unwrap(), "<u>hi");
    }

    #[test]
    fn a_prompt_not_ending_in_its_generation_prompt_is_an_explicit_error() {
        let error = chat_text(applied("<u>hi", "<a>"), false).unwrap_err();
        assert!(matches!(error, ServingError::Internal(_)), "{error:?}");
    }

    // The tests below need a real tokenizer and chat template: a ChatML GGUF
    // (its template renders `<|im_start|>`), on any machine, no device needed.
    // `MAGNITUDE_TEST_GGUF=M.gguf cargo test -p magnitude-serving --
    // --ignored engine::tests`.
    fn host() -> EngineHost {
        let target = std::env::var_os("MAGNITUDE_TEST_GGUF").expect("set MAGNITUDE_TEST_GGUF");
        let host = HostArtifacts::open(&PackageOptions {
            target_shards: vec![target.into()],
            projector: ProjectorSelection::Disabled,
            draft: None,
        })
        .unwrap();
        EngineHost::new(Arc::new(host))
    }

    fn conversation(kwargs: serde_json::Value) -> ChatInput {
        let request: ApplyTemplateRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "ping"}],
            "chat_template_kwargs": kwargs,
        }))
        .unwrap();
        validate_apply_template_request(request).unwrap()
    }

    #[test]
    #[ignore = "needs a ChatML GGUF: set MAGNITUDE_TEST_GGUF"]
    fn special_token_text_is_one_special_token() {
        let host = host();
        let special = host
            .host
            .tokenizer()
            .encode("<|im_start|>", SpecialTokens::Recognize)
            .unwrap();
        assert_eq!(special.len(), 1, "<|im_start|> is a single token");
        let tokens = host
            .tokenize(&TokenizeInput::Text {
                text: "<|im_start|>user".to_owned(),
                add_special_tokens: false,
            })
            .unwrap();
        assert_eq!(tokens[0], special[0].0);
        assert_eq!(host.decode_token(tokens[0]).unwrap(), "<|im_start|>");
    }

    #[test]
    #[ignore = "needs a ChatML GGUF: set MAGNITUDE_TEST_GGUF"]
    fn the_chat_form_is_the_rendered_template_tokenized() {
        let host = host();
        let input = conversation(json!({"enable_thinking": false}));
        let rendered = host.apply_template(&input).unwrap().prompt;
        let chat = host
            .tokenize(&TokenizeInput::Chat {
                input,
                add_generation_prompt: true,
                add_special_tokens: false,
            })
            .unwrap();
        let text = host
            .tokenize(&TokenizeInput::Text {
                text: rendered,
                add_special_tokens: false,
            })
            .unwrap();
        assert_eq!(chat, text);
    }

    #[test]
    #[ignore = "needs a ChatML GGUF: set MAGNITUDE_TEST_GGUF"]
    fn add_special_tokens_adds_exactly_the_tokenizers_own_token() {
        let host = host();
        let text = |add_special_tokens| {
            host.tokenize(&TokenizeInput::Text {
                text: "hello world".to_owned(),
                add_special_tokens,
            })
            .unwrap()
        };
        let (without, with) = (text(false), text(true));
        match host.host.tokenizer().implicit_bos() {
            Some(bos) => assert_eq!([&[bos.0][..], &without].concat(), with),
            None => assert_eq!(without, with),
        }
        // A text already starting with the token is not given a second one.
        if let Some(bos) = host.host.tokenizer().implicit_bos() {
            let again = host
                .tokenize(&TokenizeInput::Text {
                    text: host.decode_token(bos.0).unwrap(),
                    add_special_tokens: true,
                })
                .unwrap();
            assert_eq!(again, vec![bos.0]);
        }
    }
}
