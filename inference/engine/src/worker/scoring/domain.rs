//! The executor operations the scorer drives, and the executor domain's
//! answer to them.

use magnitude_executor::{
    DomainError, ExecutorDomain, InvariantError, LogitsRef, Operation, Outcome,
    PendingOperationOutcome, PhysicalDecision, RequestId, ReservedResources, StateBindings,
    SubmitFailure, TargetFlight, TokenId,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_scheduler::protocol::WorkerReply;
use magnitude_state::ShrinkPolicy;

/// The executor operations a scorer drives: a narrow face of
/// [`ExecutorDomain`], so the scorer's bookkeeping runs over a fake.
pub(crate) trait ScoringDomain {
    /// The one right to change the domain's state bindings.
    type Bindings;
    type Reservation;
    type Flight;
    type Pending;
    type Logits;

    /// Make `sequence` resident at position zero with no prompt. A refused
    /// open leaves nothing of the sequence.
    fn open_sequence(
        &mut self,
        bindings: &mut Self::Bindings,
        sequence: RequestId,
    ) -> Result<(), DomainError>;
    /// Reserve one step of `operations`.
    fn reserve_step(
        &mut self,
        bindings: &mut Self::Bindings,
        operations: &[Operation],
    ) -> Result<Self::Reservation, DomainError>;
    /// Submit the reserved step. A refusal returns the bindings.
    fn submit_step(
        &mut self,
        bindings: Self::Bindings,
        operations: &[Operation],
        reservation: Self::Reservation,
    ) -> Result<Self::Flight, Unsubmitted<Self::Bindings>>;
    /// Wait for the step; its outcomes in operation order, and the bindings.
    fn finish_step(
        &mut self,
        flight: Self::Flight,
    ) -> Result<(Vec<Self::Pending>, Self::Bindings), DomainError>;
    /// Commit every row of one outcome: its sequence, and its rows or why
    /// that sequence failed.
    fn commit(
        &mut self,
        pending: Self::Pending,
    ) -> (RequestId, Result<Vec<Forwarded<Self::Logits>>, DomainError>);
    fn close_sequence(&mut self, sequence: RequestId) -> Result<(), String>;
    /// One row's logits on the host.
    fn read_logits(&self, logits: &Self::Logits) -> Result<Vec<f32>, String>;
    /// One probe of every memory domain the model uses.
    fn probe_memory(&mut self) -> Result<(), DomainError>;
    /// Release state backing beyond what the sequences need; whether any
    /// was released.
    fn release_surplus(&mut self, bindings: &mut Self::Bindings) -> Result<bool, DomainError>;
    /// The owner has no work: time until its next launch is idle.
    fn idle(&mut self);
    /// A fresh reading of every memory domain, and the charge reconciled
    /// against the states the domain holds.
    fn observe_memory(&self) -> Result<WorkerReply, String>;
}

/// Why a step was not submitted.
pub(crate) enum Unsubmitted<B> {
    /// Refused before reaching the device: the bindings return.
    Refused(DomainError, B),
    /// The device or an invariant failed: the bindings are gone.
    Failed(DomainError),
}

/// One forwarded row as the domain returns it.
pub(crate) struct Forwarded<L> {
    pub(super) logits: Option<L>,
    pub(super) selected: Option<TokenId>,
}

pub(super) fn invariant(detail: impl Into<String>) -> DomainError {
    DomainError::Invariant(InvariantError {
        context: "scoring",
        detail: detail.into(),
    })
}

impl ScoringDomain for ExecutorDomain {
    type Bindings = StateBindings;
    type Reservation = ReservedResources;
    type Flight = TargetFlight;
    type Pending = PendingOperationOutcome;
    type Logits = LogitsRef;

    fn open_sequence(
        &mut self,
        bindings: &mut StateBindings,
        sequence: RequestId,
    ) -> Result<(), DomainError> {
        self.install_input(sequence, PreparedModelInput::continuation_only())
            .map_err(invariant)?;
        let opened = self.open_state(bindings, sequence, None);
        if matches!(&opened, Ok(encodes) if encodes.is_empty()) {
            return Ok(());
        }
        self.close(sequence).map_err(invariant)?;
        match opened {
            Ok(_) => Err(invariant("a scoring sequence opened with images to encode")),
            Err(error) => Err(error),
        }
    }

    fn reserve_step(
        &mut self,
        bindings: &mut StateBindings,
        operations: &[Operation],
    ) -> Result<ReservedResources, DomainError> {
        let resources = self.reserve(bindings, operations)?.into_resources();
        match resources {
            ReservedResources::Target(_) => Ok(resources),
            _ => Err(invariant(
                "a scoring step reserved a lane other than the target's",
            )),
        }
    }

    fn submit_step(
        &mut self,
        bindings: StateBindings,
        operations: &[Operation],
        reservation: ReservedResources,
    ) -> Result<TargetFlight, Unsubmitted<StateBindings>> {
        let ReservedResources::Target(reservation) = reservation else {
            return Err(Unsubmitted::Refused(
                invariant("a scoring step reserved a lane other than the target's"),
                bindings,
            ));
        };
        self.submit_target(bindings, operations, reservation)
            .map_err(|failure| match failure {
                SubmitFailure::Refused(error, bindings) => Unsubmitted::Refused(error, bindings),
                SubmitFailure::Failed(error) => Unsubmitted::Failed(error),
            })
    }

    fn finish_step(
        &mut self,
        flight: TargetFlight,
    ) -> Result<(Vec<PendingOperationOutcome>, StateBindings), DomainError> {
        self.finish_target(flight)
    }

    fn commit(
        &mut self,
        pending: PendingOperationOutcome,
    ) -> (RequestId, Result<Vec<Forwarded<LogitsRef>>, DomainError>) {
        let sequence = pending.request();
        let outcome = pending.outcome().clone();
        let accepted_rows = pending.rows();
        let rows = self
            .reconcile(pending, PhysicalDecision { accepted_rows })
            .and_then(|()| match outcome {
                Outcome::Forward { rows } => Ok(rows
                    .into_iter()
                    .map(|row| Forwarded {
                        logits: row.logits,
                        selected: row.selected.map(|selected| selected.token),
                    })
                    .collect()),
                _ => Err(invariant(
                    "a scoring step returned an outcome other than a forward",
                )),
            });
        (sequence, rows)
    }

    fn close_sequence(&mut self, sequence: RequestId) -> Result<(), String> {
        self.close(sequence)
    }

    fn read_logits(&self, logits: &LogitsRef) -> Result<Vec<f32>, String> {
        logits.read_to_host().map_err(|error| error.to_string())
    }

    fn probe_memory(&mut self) -> Result<(), DomainError> {
        ExecutorDomain::probe_memory(self)
    }

    fn release_surplus(&mut self, bindings: &mut StateBindings) -> Result<bool, DomainError> {
        let shrunk = self.shrink_state(bindings, ShrinkPolicy::Reclaim)?;
        let idle = self.reclaim_idle(bindings).map_err(invariant)?;
        Ok(shrunk != 0 || idle != 0)
    }

    fn idle(&mut self) {
        self.host_idle();
    }

    /// A scoring load retains no prefix: the domain's census of its live
    /// states is the whole charge.
    fn observe_memory(&self) -> Result<WorkerReply, String> {
        Ok(WorkerReply::Observed {
            readings: self.refresh_memory().map_err(|error| error.to_string())?,
            reconciliation: self.reconcile_memory_charge(&[])?,
        })
    }
}
