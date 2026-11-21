//! What protocol handlers need from the models they serve (integration spec
//! §8.2). Handlers are router-independent: they name a model and receive
//! either a generation binding (which may load the model and holds its
//! release guard) or its host chat semantics (which never lease or load).
//! The CLI serves one in-process engine; the service resolves canonical models,
//! ensures instances and leases them. Tests serve scripted models.
use crate::error::ServingError;
use futures_util::future::BoxFuture;
use magnitude_chat::{
    output::{Completion, OutputEvent, Progress, TimingSnapshot},
    ChatInput, GenerationRequest,
};
use magnitude_engine::chat::{AppliedTemplate, ModelProperties};
use serde::Serialize;
use std::sync::Arc;
use utoipa::ToSchema;
use tokio::sync::mpsc;

/// Where a model load is. The service's model instance status names the same stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadStage {
    /// Waiting for memory admission.
    Queued,
    /// Resolving, planning, opening the device and preparing programs.
    Preparing,
    /// Tuning kernels for the device; only a load that tunes enters it.
    Optimizing,
    LoadingWeights,
    /// State allocation, warm-up and readiness verification.
    Finalizing,
}

/// Selected optional tuning work, not a fixed sequence of global phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum KernelTuningOperation {
    Discovery,
    Form,
    Measure,
    Promote,
    Confirm,
}

#[derive(Debug, Clone, PartialEq, Serialize, ToSchema)]
pub struct KernelTuningWork {
    pub entry: String,
    pub operation: KernelTuningOperation,
}

/// Exact stage observations; budget time is not configuration completion.
#[derive(Debug, Clone, PartialEq, Serialize, ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelLoadDetails {
    Tuning {
        elapsed_ms: u64,
        budget_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        #[schema(nullable = false)]
        work: Option<KernelTuningWork>,
        measured_configurations: u64,
        formed_programs: u64,
    },
    Weights {
        completed_bytes: u64,
        total_bytes: u64,
    },
}

/// A model load's stage and completed fraction in `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelLoadProgress {
    pub stage: ModelLoadStage,
    pub fraction: f32,
    pub details: Option<ModelLoadDetails>,
}

/// Observes model loading.
pub type LoadProgress = Arc<dyn Fn(ModelLoadProgress) + Send + Sync>;

/// The models a router serves.
pub trait ServedModels: Send + Sync + 'static {
    /// Bind a generation to the model `model` names, loading it if the source
    /// manages residency. The binding is to one exact model generation and
    /// releases its guard when the returned invocation (and every generation
    /// it started) ends.
    fn invoke(
        &self,
        model: &str,
        progress: Option<LoadProgress>,
    ) -> BoxFuture<'_, Result<Box<dyn ModelInvocation>, ServingError>>;

    /// The model's host chat semantics for operations that never lease or
    /// load: counting, template application.
    fn host(&self, model: &str) -> BoxFuture<'_, Result<Arc<dyn HostChat>, ServingError>>;

    /// Teacher-forced scoring of `tokens` under the model `model` names, which
    /// the source may load. One position per token after the first. `top` asks
    /// for the top-1 token where it differs from the real one; `next` asks for
    /// the greedy token after the prompt. A source that cannot score the model
    /// answers with the error it would give any unsupported operation.
    fn score(
        &self,
        model: &str,
        tokens: Vec<u32>,
        top: bool,
        next: bool,
    ) -> BoxFuture<'_, Result<PromptScore, ServingError>>;
}

/// A prompt's teacher-forced score, in the token ids [`HostChat`] uses: one
/// position per token after the first, and the greedy token after the prompt
/// when it was asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptScore {
    pub positions: Vec<PositionScore>,
    pub next: Option<u32>,
}

/// One prompt position: the real token's log-probability and 1-based rank,
/// and the top-1 token when it was asked for and differs from the real one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionScore {
    pub logprob: f64,
    pub rank: u32,
    pub top: Option<TopScore>,
}

/// A position's most likely token and its log-probability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopScore {
    pub token: u32,
    pub logprob: f64,
}

/// One bound model generation.
pub trait ModelInvocation: Send {
    /// Start generating. Dropping the returned stream cancels the request.
    fn generate(self: Box<Self>, request: GenerationRequest) -> GenerationStream;
}

/// What `/tokenize` encodes: raw text, or chat messages rendered through the
/// model's chat template.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenizeInput {
    /// Special-token text in `text` is encoded as special tokens;
    /// `add_special_tokens` controls only what the tokenizer adds itself, such
    /// as a start-of-text token.
    Text {
        text: String,
        add_special_tokens: bool,
    },
    /// `input` carries the caller's `chat_template_kwargs` as its template
    /// arguments, as template application takes them.
    Chat {
        input: ChatInput,
        add_generation_prompt: bool,
        add_special_tokens: bool,
    },
}

/// Chat semantics that run on the host alone, with the same rendering and
/// tokenization as generation.
pub trait HostChat: Send + Sync {
    /// Input tokens the rendered request occupies, including expanded media.
    fn count(&self, input: &ChatInput) -> Result<u64, ServingError>;
    fn apply_template(&self, input: &ChatInput) -> Result<AppliedTemplate, ServingError>;
    fn properties(&self) -> Result<ModelProperties, ServingError>;
    /// The token ids `input` encodes to, with the tokenizer and template
    /// generation uses.
    fn tokenize(&self, input: &TokenizeInput) -> Result<Vec<u32>, ServingError>;
    /// The text one token id stands for.
    fn decode_token(&self, token: u32) -> Result<String, ServingError>;
    /// The longest prompt, in tokens, the model's serving context takes.
    fn context_limit(&self) -> u64;
}

/// A request's lifecycle as the engine reports it. `Admitted` precedes all
/// output; exactly one of `Completed` and `Failed` ends the stream.
#[derive(Clone, Debug)]
pub enum GenerationEvent {
    Progress(Progress),
    Admitted {
        prompt_tokens: u64,
    },
    Output {
        event: OutputEvent,
        snapshot: Option<TimingSnapshot>,
    },
    Completed(Completion),
    Failed(ServingError),
}

/// The receiving end of one generation. Dropping it cancels the request.
pub struct GenerationStream {
    events: mpsc::Receiver<GenerationEvent>,
}

impl GenerationStream {
    pub fn new(events: mpsc::Receiver<GenerationEvent>) -> Self {
        Self { events }
    }

    /// The next event; `None` only after a terminal event.
    pub async fn next(&mut self) -> Option<GenerationEvent> {
        self.events.recv().await
    }
}
