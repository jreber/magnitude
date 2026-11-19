//! The executor operations the scorer drives, and the executor domain's
//! answer to them.

use std::collections::BTreeSet;
use std::time::Duration;

use magnitude_executor::memory::ledger::Relief as LedgerRelief;
use magnitude_executor::{
    CapacityError, CompletionWake, DomainError, ExecutorDomain, InvariantError, LogitsRef,
    MemoryReading, Operation, Outcome, PendingOperationOutcome, PhysicalDecision, RequestId,
    ReservedResources, StateBindings, SubmitFailure, TargetFlight, TokenId, Yielded,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_scheduler::{protocol::WorkerReply, serving::ServingCensus};

/// The executor operations a scorer drives: a narrow face of
/// [`ExecutorDomain`], so the scorer's bookkeeping runs over a fake.
pub(crate) trait ScoringDomain {
    /// The one right to change the domain's state bindings.
    type Bindings;
    type Reservation;
    type Flight;
    type Pending;
    type Logits;

    /// Make `sequence` resident at position zero with no prompt, reserving
    /// the `rows` its walk writes, as a request reserves its prompt as it
    /// opens. A refused open leaves nothing of the sequence.
    fn open_sequence(
        &mut self,
        bindings: &mut Self::Bindings,
        sequence: RequestId,
        rows: u64,
    ) -> Result<(), DomainError>;
    /// Reserve one step of `operations`: provisioned, so a memory shortage
    /// is a capacity refusal here.
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
    /// Free what the ledger took of the load's cache, and let the expert
    /// bank follow its target, as at every flight boundary; a release left
    /// pending off this thread runs `wake`'s wake once it ends.
    fn follow_memory(
        &mut self,
        bindings: &mut Self::Bindings,
        wake: &dyn Fn() -> CompletionWake,
    ) -> Result<Freed, DomainError>;
    /// Relief for a claim `capacity` refused: the cache the ledger offers to
    /// cover what it lacks, taken and freed. The scorer's walks are never
    /// preempted for it; it defers them itself.
    fn relieve(
        &mut self,
        bindings: &mut Self::Bindings,
        capacity: &CapacityError,
        wake: &dyn Fn() -> CompletionWake,
    ) -> Result<Freed, DomainError>;
    /// Read the load's memory, `walks` the sequences a relief may name. None
    /// where a change of ours stood beside the sample.
    fn read_memory(&mut self, walks: &BTreeSet<RequestId>) -> Result<Option<Read>, DomainError>;
    /// Whether an OS pressure event came since this was last asked.
    fn pressured(&self) -> bool;
    /// The owner has no work: time until its next launch is idle.
    fn idle(&mut self);
    /// The load's classified holdings, a fresh reading of its memory
    /// domains, and how it serves.
    fn observe_memory(&self) -> Result<WorkerReply, String>;
    /// Claim `bytes` beside the model for the service's measurement job; the
    /// bytes it is short by where refused.
    fn claim_measurement_room(&mut self, bytes: u64) -> Result<(), u64>;
    fn attach_measurement_process(&mut self, process: magnitude_executor::ProcessId);
    fn release_measurement_room(&mut self);
}

/// What freeing memory came to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Freed {
    /// Memory was freed now: what was refused may be tried again.
    Now,
    /// Memory frees off this thread; its end wakes the owner.
    Later,
    /// Nothing this load can free now.
    Nothing,
}

impl From<Yielded> for Freed {
    fn from(yielded: Yielded) -> Self {
        match yielded {
            Yielded::Released { .. } => Self::Now,
            Yielded::Pending => Self::Later,
            Yielded::Nothing => Self::Nothing,
        }
    }
}

/// One reading of the load's memory and its response, the cache-yielding
/// step already taken (`ExecutorDomain::read_memory`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Read {
    /// A shortage persists with nothing to give: the model unloads.
    pub unload: bool,
    /// The walks the relief carried out names, to defer.
    pub preempt: BTreeSet<RequestId>,
    /// New work waits.
    pub admission_waits: bool,
    /// The response moved what the load holds past the reading's own.
    pub moved: bool,
    /// What the reading took.
    pub cost: Duration,
    /// Something stands that only a later reading can answer.
    pub stands: bool,
    /// The shortest interval across which others' memory was seen to move.
    pub ceiling: Option<Duration>,
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
        rows: u64,
    ) -> Result<(), DomainError> {
        // The scorer launches no chain, so none holds the bindings between
        // its steps.
        let Some(unchained) = self.unchained(bindings) else {
            return Err(invariant("a scoring sequence opened beside a chain"));
        };
        self.install_input(sequence, PreparedModelInput::continuation_only())
            .map_err(invariant)?;
        let opened = self.open_state(unchained, sequence, None, rows);
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

    fn follow_memory(
        &mut self,
        bindings: &mut StateBindings,
        wake: &dyn Fn() -> CompletionWake,
    ) -> Result<Freed, DomainError> {
        // A resize of the expert bank begun at an earlier boundary finishes
        // here; the scorer keeps no record of it beyond the bank's own.
        let advanced = self.advance_expert_bank(wake())?.is_some();
        let yielded = Freed::from(self.release_yielded(bindings, wake())?);
        let shrinking = matches!(
            self.follow_cache_targets(bindings, wake())?,
            magnitude_executor::Followed::Shrank(_)
        );
        Ok(match (advanced, yielded, shrinking) {
            (true, Freed::Nothing | Freed::Later, _) => Freed::Now,
            (_, Freed::Nothing, true) => Freed::Later,
            (_, yielded, _) => yielded,
        })
    }

    fn relieve(
        &mut self,
        bindings: &mut StateBindings,
        capacity: &CapacityError,
        wake: &dyn Fn() -> CompletionWake,
    ) -> Result<Freed, DomainError> {
        match self.follow_memory(bindings, wake)? {
            Freed::Nothing => {}
            freed => return Ok(freed),
        }
        let lacks = capacity.required.saturating_sub(capacity.available);
        let Some(plan) = self.plan_relief(capacity.resource, lacks, LedgerRelief::CacheOnly) else {
            return Ok(Freed::Nothing);
        };
        if !self.execute_relief(plan).preempt.is_empty() {
            return Err(invariant("a cache-only relief named requests to preempt"));
        }
        self.follow_memory(bindings, wake)
    }

    fn read_memory(&mut self, walks: &BTreeSet<RequestId>) -> Result<Option<Read>, DomainError> {
        // Each walk is valued as the generation owner values its requests,
        // so a relief preempts the one worth least first.
        self.value_state()?;
        let read = match ExecutorDomain::read_memory(self, false, walks)? {
            MemoryReading::Beside => return Ok(None),
            MemoryReading::Read(read) => read,
        };
        let response = read.response.unwrap_or_default();
        let mut preempt = BTreeSet::new();
        if !response.unload {
            for plan in response.relief {
                preempt.extend(self.execute_relief(plan).preempt);
            }
        }
        Ok(Some(Read {
            unload: response.unload,
            preempt,
            admission_waits: response.admission_waits,
            moved: self.memory_generation() != read.generation,
            cost: read.cost,
            stands: self.memory_stands()?,
            ceiling: self.others_interval(),
        }))
    }

    fn pressured(&self) -> bool {
        self.take_pressure()
    }

    fn idle(&mut self) {
        self.host_idle();
    }

    fn observe_memory(&self) -> Result<WorkerReply, String> {
        Ok(WorkerReply::Observed {
            readings: ExecutorDomain::observe_memory(self),
            stores: self.store_allocations(),
            serving: ServingCensus::of(self)?,
        })
    }

    fn claim_measurement_room(&mut self, bytes: u64) -> Result<(), u64> {
        ExecutorDomain::claim_measurement_room(self, bytes).map_err(|short| short.over)
    }

    fn attach_measurement_process(&mut self, process: magnitude_executor::ProcessId) {
        ExecutorDomain::attach_measurement_process(self, process);
    }

    fn release_measurement_room(&mut self) {
        ExecutorDomain::release_measurement_room(self);
    }
}
