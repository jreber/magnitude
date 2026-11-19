//! Closed, device-free command vocabulary of the thread-confined execution
//! owner inside a numerical worker.
//!
//! Concrete executor types and device leases cannot be represented here; the
//! one closure is `ScoreRequest::done`, a device-free completion callback.
//! The engine's worker session is the only dispatcher; the host/worker IPC
//! protocol is defined above this crate.

use crate::prefix_cache::PrefixRetention;
use crate::publication::PublicationReceiver;
use crate::owner::{AdmissionError, Status};
use crate::publication::RequestError;
use crate::throughput::ThroughputSnapshot;
use magnitude_executor::platform::DomainReading;
use crate::scoring::ScoreOutcome;
use magnitude_executor::{MemoryChargeReconciliation, RequestId, TokenId};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::GenerationSeed;

pub struct AdmitRequest {
    pub seed: GenerationSeed,
    pub input: PreparedModelInput,
    /// Whether the request resumes from and contributes to the prefix
    /// cache, and where it retains states for later requests.
    pub retention: PrefixRetention,
    pub output_capacity: usize,
}

/// Teacher-forced scoring of `tokens`: the model's log-probability of each
/// token given those before it, with no sampling.
pub struct ScoreRequest {
    pub tokens: Vec<TokenId>,
    /// Report each position's most likely token as well.
    pub top: bool,
    /// Report the most likely token after the last one.
    pub next: bool,
    /// Reports the outcome, from the execution thread. Exactly one of two
    /// things happens: the owner acknowledges the command and calls `done`
    /// exactly once, or the command fails (the owner returns `Err`, execution
    /// has stopped, or the command is never dispatched) and `done` is dropped
    /// uncalled, so the dispatcher reports the failure.
    pub done: Box<dyn FnOnce(Result<ScoreOutcome, RequestError>) + Send>,
}

impl std::fmt::Debug for ScoreRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ScoreRequest")
            .field("tokens", &self.tokens.len())
            .field("top", &self.top)
            .field("next", &self.next)
            .finish_non_exhaustive()
    }
}

pub enum WorkerCommand {
    Admit(AdmitRequest),
    /// Score a token sequence. Either the owner answers
    /// `Ok(WorkerReply::Acknowledged)` and calls `ScoreRequest::done` exactly
    /// once, or the command fails (`Err`, execution stopped, or never
    /// dispatched) and `done` is dropped uncalled.
    Score(ScoreRequest),
    Stop { request: RequestId },
    Status { request: RequestId },
    /// The memory heap's classified holdings and standing.
    Observe,
    /// A live accepted-output token snapshot over the owner's ten-second window.
    Throughput,
}

/// A live request's scheduling status and progress.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestSnapshot {
    pub status: Status,
    pub prompt_tokens: usize,
    /// Leading prompt tokens restored from a retained prefix, not computed.
    pub cached_tokens: usize,
    pub resident_position: usize,
    pub output_tokens: usize,
}

pub enum WorkerReply {
    Admitted {
        request: RequestId,
        receiver: PublicationReceiver,
    },
    AdmissionRefused(AdmissionError),
    /// `None` when the request is not live.
    Status(Option<RequestSnapshot>),
    Observed {
        /// Seismic's charge classified by every holder the owner keeps.
        reconciliation: MemoryChargeReconciliation,
        /// A fresh reading of every memory domain the device uses.
        readings: Vec<DomainReading>,
    },
    Throughput(ThroughputSnapshot),
    Acknowledged,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn score_request_debug_elides_the_callback() {
        let request = ScoreRequest {
            tokens: vec![TokenId(1), TokenId(2)],
            top: true,
            next: false,
            done: Box::new(|_| {}),
        };
        assert_eq!(
            format!("{request:?}"),
            "ScoreRequest { tokens: 2, top: true, next: false, .. }"
        );
    }

    #[test]
    fn protocol_is_device_free_and_sendable() {
        assert_send::<WorkerCommand>();
        assert_send::<WorkerReply>();
    }
}
