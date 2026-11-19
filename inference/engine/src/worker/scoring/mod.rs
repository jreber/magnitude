//! The worker's thread-confined scoring owner: teacher-forced walks of token
//! sequences through the executor domain, below the scheduler. It is the
//! walk of `magnitude-harness logits`, served; a scoring load runs it in
//! place of the generation owner ([`super::execution::ExecutionOwner`]).
//!
//! Up to [`SCORING_SEQUENCES`] scores hold a sequence each; the rest wait in
//! arrival order. Each step forwards up to `SCORING_WIDTH` next rows of
//! every walk in one launch, exporting their logits and selecting greedily
//! at each. Row `p`'s logits predict token `p + 1`, so a score of `n` tokens
//! has `n - 1` positions, and its last row's selection is the greedy next
//! token. Every row is a decode or verification row: a prompt chunk
//! exports logits for its last row only.
//!
//! The domain pushes back, and the owner yields rather than stopping:
//! - capacity a step cannot have defers its newest walks (their sequences
//!   closed, their progress dropped) to the front of the queue, and
//!   admission waits for a walk to end. A score fails with the shortage only
//!   when it cannot be walked alone;
//! - memory is observed on the generation owner's cadence and condition
//!   ([`magnitude_scheduler::memory_condition`]). In Reclaim the owner
//!   pauses and releases surplus state, then its newest walks; a growth
//!   refused for memory waits for Normal; failed observations are tolerated
//!   for an escalation interval; Reclaim that persists unloads the model.
//!
//! Execution stops only on the errors that stop the generation owner (device,
//! state and invariant failures) and on a logits row that cannot be read.
//! Other errors fail the scores they concern. A step runs to its end within
//! one `run`, so nothing is in flight between calls: commands and events that
//! arrive during a step apply before the next one.

mod domain;
mod walk;

use crate::error::UnloadCause;
use crate::options::SCORING_SEQUENCES;
use domain::{invariant, Forwarded, ScoringDomain, Unsubmitted};
use magnitude_executor::{DeviceError, DomainError, ExecutorDomain, RequestId};
use magnitude_family_contracts::ModelDefinition;
use magnitude_scheduler::{
    memory_condition::{MemoryCondition, MemoryObservation, ObservationCadence},
    owner::{is_fatal, AdmissionError},
    protocol::{ScoreRequest, WorkerCommand, WorkerReply},
    publication::{ModelUnloadCause, RequestError},
    scoring::ScoreOutcome,
    throughput::{ThroughputSnapshot, WINDOW_MS},
    worker::{Driven, Event},
};
use std::collections::VecDeque;
use walk::{Absorbed, Bounds, Row, Walk};

/// The scoring lifecycle. A step ends within `run`, so there is no flight to
/// drain: a stopping owner stops at once.
enum Execution<D: ScoringDomain> {
    Serving(Scorer<D>),
    Stopped(UnloadCause),
}

pub(crate) struct ScoringOwner<D: ScoringDomain = ExecutorDomain> {
    execution: Execution<D>,
}

impl<D: ScoringDomain> ScoringOwner<D> {
    /// Score on `domain`, whose bindings no flight holds, within the context
    /// and vocabulary of the load's `definition`.
    pub fn new(
        domain: D,
        bindings: D::Bindings,
        definition: &ModelDefinition,
    ) -> Result<Self, String> {
        Ok(Self::serving(domain, bindings, Bounds::of(definition)?))
    }

    fn serving(domain: D, bindings: D::Bindings, bounds: Bounds) -> Self {
        Self {
            execution: Execution::Serving(Scorer {
                domain,
                bindings: Some(bindings),
                bounds,
                waiting: VecDeque::new(),
                walks: Vec::with_capacity(SCORING_SEQUENCES),
                next_sequence: 1,
                capacity_blocked: false,
                memory: MemoryCondition::Normal,
                observations: ObservationCadence::default(),
                memory_wait: false,
            }),
        }
    }

    /// Move the execution out for a transition. The placeholder is truthful:
    /// a transition that panics has stopped execution.
    fn take(&mut self) -> Execution<D> {
        std::mem::replace(
            &mut self.execution,
            Execution::Stopped(UnloadCause::Internal {
                reason: "execution transition interrupted".into(),
            }),
        )
    }
}

/// The scoring owner does not generate: it refuses the admission.
fn refuse_generation() -> WorkerReply {
    WorkerReply::AdmissionRefused(AdmissionError::Refused(RequestError::Input(
        "this load serves scoring, not generation".into(),
    )))
}

/// How a score that reaches a stopped owner fails, as the generation owner
/// refuses an admission then.
fn stopped(cause: &UnloadCause) -> RequestError {
    match cause {
        UnloadCause::MemoryPressure => RequestError::ModelUnloaded {
            cause: ModelUnloadCause::MemoryPressure,
        },
        UnloadCause::Shutdown | UnloadCause::DeviceLost { .. } | UnloadCause::Internal { .. } => {
            RequestError::WorkerClosed
        }
    }
}

impl<D: ScoringDomain> Driven for ScoringOwner<D> {
    type Stopped = UnloadCause;

    fn command(&mut self, command: WorkerCommand, _: u64) -> Result<WorkerReply, String> {
        match command {
            WorkerCommand::Admit(_) => Ok(refuse_generation()),
            WorkerCommand::Score(request) => {
                match &mut self.execution {
                    Execution::Serving(scorer) => match scorer.bounds.refusal(&request.tokens) {
                        Some(error) => (request.done)(Err(error)),
                        None => scorer.waiting.push_back(request),
                    },
                    Execution::Stopped(cause) => (request.done)(Err(stopped(cause))),
                }
                Ok(WorkerReply::Acknowledged)
            }
            // A score is not stoppable, and no generation is live here.
            WorkerCommand::Stop { .. } => Ok(WorkerReply::Acknowledged),
            WorkerCommand::Status { .. } => Ok(WorkerReply::Status(None)),
            WorkerCommand::Observe => match &self.execution {
                Execution::Serving(scorer) => scorer.domain.observe_memory(),
                Execution::Stopped(cause) => Err(format!("execution stopped: {cause}")),
            },
            WorkerCommand::Throughput => match &self.execution {
                // A scoring load emits no output tokens: its output throughput is zero over the
                // whole window, with nothing to warm up.
                Execution::Serving(_) => Ok(WorkerReply::Throughput(ThroughputSnapshot {
                    window_ms: WINDOW_MS,
                    ready: true,
                    accepted_tokens: 0,
                })),
                Execution::Stopped(cause) => Err(format!("execution stopped: {cause}")),
            },
        }
    }

    fn event(&mut self, event: Event, _: u64) {
        match event {
            // A step waits for its own flight, nothing publishes, and a score
            // is not cancellable.
            Event::Completion | Event::Publication(..) | Event::Cancel(_) => {}
            Event::Close => {
                self.execution = match self.take() {
                    Execution::Serving(scorer) => {
                        scorer.fail_all(&RequestError::WorkerClosed);
                        Execution::Stopped(UnloadCause::Shutdown)
                    }
                    stopped @ Execution::Stopped(_) => stopped,
                }
            }
        }
    }

    fn run(&mut self, now: u64) -> Result<u64, UnloadCause> {
        match self.take() {
            Execution::Serving(mut scorer) => match scorer.run(now) {
                Ok(due) => {
                    self.execution = Execution::Serving(scorer);
                    Ok(due)
                }
                Err(Fatal { error, cause }) => {
                    tracing::error!(%error, "scoring stopped");
                    scorer.fail_all(&error);
                    self.execution = Execution::Stopped(cause.clone());
                    Err(cause)
                }
            },
            Execution::Stopped(cause) => {
                self.execution = Execution::Stopped(cause.clone());
                Err(cause)
            }
        }
    }
}

/// An error that ends execution: every score still held fails with `error`,
/// and the owner stops with `cause`.
struct Fatal {
    error: RequestError,
    cause: UnloadCause,
}

impl Fatal {
    /// Execution ends on `error` as the generation owner's does.
    fn stop(error: DomainError) -> Self {
        Self::request(RequestError::from(error))
    }

    fn request(error: RequestError) -> Self {
        let cause = UnloadCause::from(&error);
        Self { error, cause }
    }

    fn invariant(detail: impl Into<String>) -> Self {
        Self::stop(invariant(detail))
    }

    /// Reclaim persisted after releases were exhausted.
    fn memory_pressure() -> Self {
        Self {
            error: RequestError::ModelUnloaded {
                cause: ModelUnloadCause::MemoryPressure,
            },
            cause: UnloadCause::MemoryPressure,
        }
    }

    fn bindings_lost() -> Self {
        Self::invariant("the state bindings are absent between steps")
    }
}

/// How the scorer meets a domain error.
enum Setback {
    /// The domain's capacity is short: fewer walks, or the one alone fails.
    Capacity(RequestError),
    /// Memory is in Reclaim or cannot be observed: growth waits for Normal.
    Memory(MemoryObservation),
    /// The scores the error concerns fail; execution continues.
    Request(RequestError),
    Fatal(Fatal),
}

impl Setback {
    fn of(error: DomainError) -> Self {
        match error {
            DomainError::Reclaim => Self::Memory(MemoryObservation::Reclaim),
            DomainError::Blind(_) => Self::Memory(MemoryObservation::Blind),
            error @ DomainError::Capacity(_) => Self::Capacity(RequestError::from(error)),
            error if is_fatal(&error) => Self::Fatal(Fatal::stop(error)),
            other => Self::Request(RequestError::from(other)),
        }
    }
}

/// What a step that met a setback does next.
enum Retry {
    /// Step again at once: surplus state was released, or a walk deferred.
    Now,
    /// Step again on a later run.
    Later,
}

/// The domain and the scores it walks.
struct Scorer<D: ScoringDomain> {
    domain: D,
    /// The domain's state bindings while no step holds them.
    bindings: Option<D::Bindings>,
    bounds: Bounds,
    waiting: VecDeque<ScoreRequest>,
    walks: Vec<Walk>,
    next_sequence: u64,
    /// A walk met the domain's capacity: admission waits for one to end.
    capacity_blocked: bool,
    memory: MemoryCondition,
    observations: ObservationCadence,
    /// A growth was refused for memory: steps wait until it is Normal.
    memory_wait: bool,
}

impl<D: ScoringDomain> Scorer<D> {
    /// Observe memory when due, admit waiting scores while fewer than S walk
    /// and capacity allows, then take one step of every walk. Returns when
    /// the owner must run again without an event.
    fn run(&mut self, now: u64) -> Result<u64, Fatal> {
        self.observe(now)?;
        if self.memory.should_unload(now) {
            tracing::error!(
                scores = self.walks.len() + self.waiting.len(),
                "unloading the model: memory stayed below the reclaim reserve"
            );
            return Err(Fatal::memory_pressure());
        }
        self.admit(now)?;
        if !self.walks.is_empty() && !self.paused() {
            self.step(now)?;
        }
        if self.walks.is_empty() {
            // Waiting for work is no launch's service.
            self.domain.idle();
        }
        let ready = !self.paused() && (!self.walks.is_empty() || self.admissible());
        Ok(if ready { now } else { self.observations.next() })
    }

    /// Growth pauses in Reclaim, and after a growth refused for memory.
    fn paused(&self) -> bool {
        self.memory_wait || matches!(self.memory, MemoryCondition::Reclaim { .. })
    }

    /// Whether a waiting score may be opened: a walk is free, and no walk
    /// short of capacity waits for another to end.
    fn admissible(&self) -> bool {
        !self.waiting.is_empty()
            && self.walks.len() < SCORING_SEQUENCES
            && (!self.capacity_blocked || self.walks.is_empty())
    }

    /// Observe memory when it is due. In Reclaim, release what the scorer
    /// holds; a release that freed memory restarts the unload clock.
    fn observe(&mut self, now: u64) -> Result<(), Fatal> {
        if !self.observations.take(now) {
            return Ok(());
        }
        let observation = self.probe()?;
        self.enter(self.memory.observe(observation, now));
        if observation == MemoryObservation::Reclaim && self.release()? {
            self.memory = self.memory.released(now);
            let after = self.probe()?;
            self.enter(self.memory.observe(after, now));
        }
        Ok(())
    }

    fn probe(&mut self) -> Result<MemoryObservation, Fatal> {
        MemoryObservation::of(self.domain.probe_memory())
            .map_err(|error| Fatal::invariant(format!("memory observation: {error}")))
    }

    /// Release in the Reclaim order while memory stays in Reclaim: surplus
    /// state backing, then the newest walks, deferred. Returns whether
    /// anything was released.
    fn release(&mut self) -> Result<bool, Fatal> {
        let mut released = self.release_surplus()?;
        while !self.walks.is_empty() && self.probe()? == MemoryObservation::Reclaim {
            self.defer_newest()?;
            self.release_surplus()?;
            released = true;
        }
        Ok(released)
    }

    fn release_surplus(&mut self) -> Result<bool, Fatal> {
        let bindings = self.bindings.as_mut().ok_or_else(Fatal::bindings_lost)?;
        release_surplus(&mut self.domain, bindings)
    }

    /// Enter `next`. Memory returning to Normal is an availability change:
    /// every wait on memory or capacity ends.
    fn enter(&mut self, next: MemoryCondition) {
        if std::mem::discriminant(&self.memory) != std::mem::discriminant(&next) {
            tracing::warn!(from = ?self.memory, to = ?next, "memory condition changed");
        }
        if next == MemoryCondition::Normal && self.memory != MemoryCondition::Normal {
            self.memory_wait = false;
            self.capacity_blocked = false;
        }
        self.memory = next;
    }

    /// A growth was refused for memory: that refusal is an observation, and
    /// growth waits until memory is Normal.
    fn refused_for_memory(&mut self, observation: MemoryObservation, now: u64) {
        self.enter(self.memory.observe(observation, now));
        self.memory_wait = true;
    }

    fn admit(&mut self, now: u64) -> Result<(), Fatal> {
        while self.admissible() && !self.paused() {
            let Some(request) = self.waiting.pop_front() else {
                break;
            };
            let sequence = RequestId(self.next_sequence);
            self.next_sequence += 1;
            let opened = match self.open(sequence) {
                Ok(opened) => opened,
                Err(fatal) => {
                    (request.done)(Err(fatal.error.clone()));
                    return Err(fatal);
                }
            };
            let Err(error) = opened else {
                self.walks.push(Walk::new(request, sequence));
                continue;
            };
            match Setback::of(error) {
                // It cannot be opened even alone.
                Setback::Capacity(error) if self.walks.is_empty() => (request.done)(Err(error)),
                Setback::Capacity(_) => {
                    self.waiting.push_front(request);
                    self.capacity_blocked = true;
                }
                Setback::Memory(observation) => {
                    self.waiting.push_front(request);
                    self.refused_for_memory(observation, now);
                }
                Setback::Request(error) => (request.done)(Err(error)),
                Setback::Fatal(fatal) => {
                    (request.done)(Err(fatal.error.clone()));
                    return Err(fatal);
                }
            }
        }
        Ok(())
    }

    /// Open `sequence`. A capacity shortage first releases surplus state
    /// backing and opens again, as the generation owner's first rung does.
    fn open(&mut self, sequence: RequestId) -> Result<Result<(), DomainError>, Fatal> {
        let bindings = self.bindings.as_mut().ok_or_else(Fatal::bindings_lost)?;
        let opened = self.domain.open_sequence(bindings, sequence);
        if matches!(opened, Err(DomainError::Capacity(_)))
            && release_surplus(&mut self.domain, bindings)?
        {
            return Ok(self.domain.open_sequence(bindings, sequence));
        }
        Ok(opened)
    }

    /// One launch of every walk's next rows, waited for and committed; each
    /// walk takes its rows, and every walk that ended is closed and answered.
    /// A step the domain pushes back on is retried with fewer walks, or on a
    /// later run.
    fn step(&mut self, now: u64) -> Result<(), Fatal> {
        let mut bindings = self.bindings.take().ok_or_else(Fatal::bindings_lost)?;
        // Whether surplus state was released since the walks last changed.
        let mut released = false;
        let flight = loop {
            let operations = self.walks.iter().map(Walk::forward).collect::<Vec<_>>();
            let refusal = match self.domain.reserve_step(&mut bindings, &operations) {
                Ok(reservation) => {
                    match self.domain.submit_step(bindings, &operations, reservation) {
                        Ok(flight) => break flight,
                        Err(Unsubmitted::Refused(error, returned)) => {
                            bindings = returned;
                            error
                        }
                        Err(Unsubmitted::Failed(error)) => return Err(Fatal::stop(error)),
                    }
                }
                Err(error) => error,
            };
            // Every refusal returns the bindings.
            let retry = self.setback(refusal, now, &mut bindings, &mut released);
            if !matches!(retry, Ok(Retry::Now)) {
                self.bindings = Some(bindings);
                return retry.map(|_| ());
            }
        };
        let (pending, bindings) = self.domain.finish_step(flight).map_err(Fatal::stop)?;
        self.bindings = Some(bindings);
        if pending.len() != self.walks.len() {
            return Err(Fatal::invariant("a step's outcomes differ from its walks"));
        }
        let mut committed = Vec::with_capacity(pending.len());
        for (pending, walk) in pending.into_iter().zip(&self.walks) {
            let (sequence, rows) = self.domain.commit(pending);
            if sequence != walk.sequence() {
                return Err(Fatal::invariant(
                    "a step's outcome order differs from its walks",
                ));
            }
            committed.push(rows);
        }
        let mut verdicts = Vec::with_capacity(committed.len());
        for (walk, rows) in self.walks.iter_mut().zip(committed) {
            verdicts.push(match rows {
                Ok(rows) => walk
                    .absorb(host_rows(&self.domain, walk, rows, self.bounds.vocabulary)?)
                    .map_err(Fatal::invariant)?,
                // A sequence whose rows could not be committed fails alone:
                // every non-fatal reconcile failure leaves the request
                // resolved (its state aborted or reinserted), so `end` closes
                // it. Were it left unresolved, the refused close would stop
                // the owner with that walk answered.
                Err(error) if is_fatal(&error) => return Err(Fatal::stop(error)),
                Err(error) => Absorbed::Ended(Err(RequestError::from(error))),
            });
        }
        let mut ended = Vec::new();
        for (walk, verdict) in std::mem::take(&mut self.walks).into_iter().zip(verdicts) {
            match verdict {
                Absorbed::Continues => self.walks.push(walk),
                Absorbed::Ended(result) => ended.push((walk, result)),
            }
        }
        self.end(ended)
    }

    /// Meet a refused step. A capacity shortage first releases surplus state
    /// backing, once per set of walks, and steps again; still short, it
    /// defers the newest walk when there are more, and fails the walk that
    /// cannot be stepped alone. A memory refusal waits for Normal; other
    /// errors fail the step's walks.
    fn setback(
        &mut self,
        error: DomainError,
        now: u64,
        bindings: &mut D::Bindings,
        released: &mut bool,
    ) -> Result<Retry, Fatal> {
        let failed = match Setback::of(error) {
            Setback::Capacity(error) => {
                if !*released {
                    *released = true;
                    if release_surplus(&mut self.domain, bindings)? {
                        return Ok(Retry::Now);
                    }
                }
                if self.walks.len() > 1 {
                    self.defer_newest()?;
                    self.capacity_blocked = true;
                    // The deferred walk's backing is surplus now.
                    *released = false;
                    return Ok(Retry::Now);
                }
                error
            }
            Setback::Memory(observation) => {
                self.refused_for_memory(observation, now);
                return Ok(Retry::Later);
            }
            Setback::Fatal(fatal) => return Err(fatal),
            Setback::Request(error) => error,
        };
        let ended = self
            .walks
            .drain(..)
            .map(|walk| (walk, Err(failed.clone())))
            .collect();
        self.end(ended)?;
        Ok(Retry::Later)
    }

    /// Close the newest walk's sequence and put its request back at the
    /// front of the queue, its progress dropped.
    fn defer_newest(&mut self) -> Result<(), Fatal> {
        let walk = self
            .walks
            .pop()
            .ok_or_else(|| Fatal::invariant("no walk to defer"))?;
        let sequence = walk.sequence();
        self.waiting.push_front(walk.into_request());
        self.domain
            .close_sequence(sequence)
            .map_err(Fatal::invariant)
    }

    /// Close every ended walk's sequence and answer it. Every walk is
    /// answered before a failed close stops the owner.
    fn end(&mut self, ended: Vec<(Walk, Result<ScoreOutcome, RequestError>)>) -> Result<(), Fatal> {
        let mut fault = None;
        for (walk, result) in ended {
            let result = match self.domain.close_sequence(walk.sequence()) {
                Ok(()) => result,
                Err(detail) => {
                    let fatal = Fatal::invariant(detail);
                    let error = fatal.error.clone();
                    fault.get_or_insert(fatal);
                    Err(error)
                }
            };
            walk.answer(result);
            // Its sequence's capacity is free again.
            self.capacity_blocked = false;
        }
        fault.map_or(Ok(()), Err)
    }

    /// Fail every score still held with `error`: the owner is stopping, and
    /// the domain goes with it.
    fn fail_all(self, error: &RequestError) {
        for walk in self.walks {
            walk.answer(Err(error.clone()));
        }
        for request in self.waiting {
            (request.done)(Err(error.clone()));
        }
    }
}

/// Release state backing beyond what the sequences need; whether any was
/// released. A failed release stops the load, as a failed `shrink_state`
/// stops the generation owner's provisioning (its open path classifies it as
/// the open's own error instead).
fn release_surplus<D: ScoringDomain>(
    domain: &mut D,
    bindings: &mut D::Bindings,
) -> Result<bool, Fatal> {
    domain.release_surplus(bindings).map_err(Fatal::stop)
}

/// The step's rows of `walk` on the host, in its step's order. Logits are
/// read only for rows that predict one of its tokens.
fn host_rows<D: ScoringDomain>(
    domain: &D,
    walk: &Walk,
    rows: Vec<Forwarded<D::Logits>>,
    vocabulary: usize,
) -> Result<Vec<Row>, Fatal> {
    let start = walk.rows().start;
    rows.into_iter()
        .enumerate()
        .map(|(offset, row)| {
            let logits = match &row.logits {
                Some(logits) if walk.predicts(start + offset) => {
                    let read = domain.read_logits(logits).map_err(|error| {
                        Fatal::request(RequestError::Device(DeviceError::Transfer(format!(
                            "reading a logits row: {error}"
                        ))))
                    })?;
                    if read.len() != vocabulary {
                        return Err(Fatal::invariant(format!(
                            "a logits row has {} entries for a vocabulary of {vocabulary}",
                            read.len()
                        )));
                    }
                    Some(read)
                }
                _ => None,
            };
            Ok(Row {
                logits,
                selected: row.selected,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::walk::tests::{request, tokens, Outcomes};
    use super::*;
    use crate::options::SCORING_WIDTH;
    use magnitude_executor::{CapacityError, Operation, ResourceKind, TokenId};
    use magnitude_scheduler::memory_condition::{MEMORY_ESCALATION_NS, OBSERVATION_INTERVAL_NS};
    use std::collections::{BTreeMap, BTreeSet};

    const VOCABULARY: usize = 8;
    /// The token every fake logits row ranks first.
    const LIKELY: u32 = 1;

    /// The fake's bindings: neither `Clone` nor `Copy`, so a scorer that
    /// loses them cannot step again.
    struct Bindings;

    /// A domain that steps every walk on the host. Each row's logits rank
    /// [`LIKELY`] first, and each row selects it.
    #[derive(Default)]
    struct Fake {
        /// The error opening a sequence meets, once.
        open_errors: BTreeMap<RequestId, DomainError>,
        /// The most walks a step reserves before it meets capacity.
        capacity: Option<usize>,
        /// A step carrying this token meets capacity, alone or not.
        oversized: Option<TokenId>,
        /// The error the next step's reservation meets, once.
        reserve_refusal: Option<DomainError>,
        /// The refusal the next step's submission meets, once.
        submit_refusal: Option<DomainError>,
        /// The error the next step's completion meets.
        finish_error: Option<DomainError>,
        /// The error committing a sequence's rows meets, once. The sequence
        /// stays resolved, as the executor leaves it.
        commit_errors: BTreeMap<RequestId, DomainError>,
        /// The domain holds surplus state backing: releasing it frees memory
        /// and lifts `capacity`.
        surplus: bool,
        /// How often surplus state was released.
        releases: usize,
        /// Sequences whose logits rows hold a NaN.
        poisoned: BTreeSet<RequestId>,
        /// What every memory probe reads.
        memory: Option<DomainError>,
        open: BTreeSet<RequestId>,
        closed: Vec<RequestId>,
        /// The sequences of every submitted step.
        submitted: Vec<Vec<RequestId>>,
    }

    /// The forwards of a step: sequence, position, rows.
    type Forwards = Vec<(RequestId, usize, usize)>;

    fn capacity_error() -> DomainError {
        DomainError::Capacity(CapacityError {
            resource: ResourceKind::StateRows,
            required: 2,
            available: 1,
        })
    }

    impl ScoringDomain for Fake {
        type Bindings = Bindings;
        type Reservation = Forwards;
        type Flight = (Bindings, Forwards);
        type Pending = (RequestId, usize, usize);
        type Logits = RequestId;

        fn open_sequence(
            &mut self,
            _: &mut Bindings,
            sequence: RequestId,
        ) -> Result<(), DomainError> {
            if let Some(error) = self.open_errors.remove(&sequence) {
                return Err(error);
            }
            assert!(self.open.insert(sequence), "a sequence opens once");
            Ok(())
        }

        fn reserve_step(
            &mut self,
            _: &mut Bindings,
            operations: &[Operation],
        ) -> Result<Forwards, DomainError> {
            if let Some(error) = self.reserve_refusal.take() {
                return Err(error);
            }
            if self
                .capacity
                .is_some_and(|capacity| operations.len() > capacity)
            {
                return Err(capacity_error());
            }
            let mut forwards = Vec::new();
            for operation in operations {
                let Operation::Forward {
                    request,
                    tokens,
                    position,
                    ..
                } = operation
                else {
                    panic!("a scorer forwards");
                };
                assert!(
                    self.open.contains(request),
                    "a step forwards open sequences"
                );
                if self.oversized.is_some_and(|token| tokens.contains(&token)) {
                    return Err(capacity_error());
                }
                forwards.push((*request, *position, tokens.len()));
            }
            Ok(forwards)
        }

        fn submit_step(
            &mut self,
            bindings: Bindings,
            _: &[Operation],
            forwards: Forwards,
        ) -> Result<(Bindings, Forwards), Unsubmitted<Bindings>> {
            if let Some(error) = self.submit_refusal.take() {
                return Err(Unsubmitted::Refused(error, bindings));
            }
            self.submitted
                .push(forwards.iter().map(|(sequence, ..)| *sequence).collect());
            Ok((bindings, forwards))
        }

        fn finish_step(
            &mut self,
            (bindings, forwards): (Bindings, Forwards),
        ) -> Result<(Vec<(RequestId, usize, usize)>, Bindings), DomainError> {
            match self.finish_error.take() {
                Some(error) => Err(error),
                None => Ok((forwards, bindings)),
            }
        }

        fn commit(
            &mut self,
            (sequence, _, rows): (RequestId, usize, usize),
        ) -> (RequestId, Result<Vec<Forwarded<RequestId>>, DomainError>) {
            if let Some(error) = self.commit_errors.remove(&sequence) {
                return (sequence, Err(error));
            }
            let row = || Forwarded {
                logits: Some(sequence),
                selected: Some(TokenId(LIKELY)),
            };
            (sequence, Ok((0..rows).map(|_| row()).collect()))
        }

        fn close_sequence(&mut self, sequence: RequestId) -> Result<(), String> {
            if !self.open.remove(&sequence) {
                return Err(format!("sequence {} is not open", sequence.0));
            }
            self.closed.push(sequence);
            Ok(())
        }

        fn read_logits(&self, sequence: &RequestId) -> Result<Vec<f32>, String> {
            let mut row = vec![0.0; VOCABULARY];
            row[LIKELY as usize] = 4.0;
            if self.poisoned.contains(sequence) {
                row[3] = f32::NAN;
            }
            Ok(row)
        }

        fn probe_memory(&mut self) -> Result<(), DomainError> {
            self.memory.clone().map_or(Ok(()), Err)
        }

        fn release_surplus(&mut self, _: &mut Bindings) -> Result<bool, DomainError> {
            self.releases += 1;
            if !std::mem::take(&mut self.surplus) {
                return Ok(false);
            }
            self.capacity = None;
            Ok(true)
        }

        fn idle(&mut self) {}

        fn observe_memory(&self) -> Result<WorkerReply, String> {
            Err("the fake reads no memory".into())
        }
    }

    fn owner(fake: Fake) -> ScoringOwner<Fake> {
        ScoringOwner::serving(
            fake,
            Bindings,
            Bounds {
                context: 256,
                vocabulary: VOCABULARY,
            },
        )
    }

    fn fake(owner: &ScoringOwner<Fake>) -> &Fake {
        match &owner.execution {
            Execution::Serving(scorer) => &scorer.domain,
            Execution::Stopped(cause) => panic!("the owner stopped: {cause}"),
        }
    }

    fn scorer(owner: &mut ScoringOwner<Fake>) -> &mut Scorer<Fake> {
        match &mut owner.execution {
            Execution::Serving(scorer) => scorer,
            Execution::Stopped(cause) => panic!("the owner stopped: {cause}"),
        }
    }

    /// Queue a score of `length` copies of `token`.
    fn score(owner: &mut ScoringOwner<Fake>, token: u32, length: usize) -> Outcomes {
        let (request, outcomes) = request(tokens(std::iter::repeat_n(token, length)), false, true);
        let reply = owner.command(WorkerCommand::Score(request), 0).unwrap();
        assert!(matches!(reply, WorkerReply::Acknowledged));
        outcomes
    }

    /// Run the owner at `now` until it has nothing to do at once.
    fn settle(owner: &mut ScoringOwner<Fake>, now: u64) -> Result<u64, UnloadCause> {
        for _ in 0..1000 {
            let due = owner.run(now)?;
            if due != now {
                return Ok(due);
            }
        }
        panic!("the owner never settled");
    }

    /// The one answer a score received.
    fn answer(outcomes: &Outcomes) -> Result<ScoreOutcome, RequestError> {
        let answer = outcomes.try_recv().expect("the score was answered");
        assert!(outcomes.try_recv().is_err(), "a score is answered once");
        answer
    }

    fn scored(outcomes: &Outcomes, length: usize) {
        let outcome = answer(outcomes).unwrap();
        assert_eq!(outcome.positions.len(), length - 1);
        assert!(outcome.positions.iter().all(|position| position.rank == 1));
        assert_eq!(outcome.next, Some(TokenId(LIKELY)));
    }

    #[test]
    fn walks_score_in_steps_and_close_their_sequences() {
        let mut owner = owner(Fake::default());
        let length = SCORING_WIDTH + 5;
        let scores = (0..3)
            .map(|_| score(&mut owner, LIKELY, length))
            .collect::<Vec<_>>();
        assert_eq!(settle(&mut owner, 0), Ok(OBSERVATION_INTERVAL_NS));
        for outcomes in &scores {
            scored(outcomes, length);
        }
        let fake = fake(&owner);
        assert_eq!(
            fake.submitted.len(),
            2,
            "two steps of W rows walk every score"
        );
        assert!(fake.open.is_empty());
        assert_eq!(fake.closed.len(), 3);
    }

    #[test]
    fn an_admission_failure_answers_only_that_request() {
        let mut owner = owner(Fake {
            open_errors: BTreeMap::from([(RequestId(2), DomainError::Input("unopenable".into()))]),
            ..Fake::default()
        });
        let scores = (0..3)
            .map(|_| score(&mut owner, LIKELY, 4))
            .collect::<Vec<_>>();
        settle(&mut owner, 0).unwrap();
        scored(&scores[0], 4);
        assert_eq!(
            answer(&scores[1]),
            Err(RequestError::Input("unopenable".into()))
        );
        scored(&scores[2], 4);
    }

    #[test]
    fn a_capacity_shortage_at_reserve_defers_the_newest_walk_and_the_others_complete() {
        let mut owner = owner(Fake {
            capacity: Some(2),
            ..Fake::default()
        });
        let length = SCORING_WIDTH + 5;
        let scores = (0..3)
            .map(|_| score(&mut owner, LIKELY, length))
            .collect::<Vec<_>>();
        settle(&mut owner, 0).unwrap();
        for outcomes in &scores {
            scored(outcomes, length);
        }
        let fake = fake(&owner);
        assert!(fake.submitted.iter().all(|step| step.len() <= 2));
        assert_eq!(fake.submitted[0], vec![RequestId(1), RequestId(2)]);
        // The deferred walk's sequence closed unscored; its score walked again
        // from its first token on a fresh sequence.
        assert_eq!(fake.closed[0], RequestId(3));
        assert!(fake
            .submitted
            .iter()
            .any(|step| step.contains(&RequestId(4))));
        assert!(fake.open.is_empty());
    }

    /// The first rung is the generation owner's: release surplus state
    /// backing and step again. No walk is deferred.
    #[test]
    fn a_capacity_shortage_that_a_release_clears_defers_no_walk() {
        let mut owner = owner(Fake {
            capacity: Some(2),
            surplus: true,
            ..Fake::default()
        });
        let scores = (0..3)
            .map(|_| score(&mut owner, LIKELY, 4))
            .collect::<Vec<_>>();
        settle(&mut owner, 0).unwrap();
        for outcomes in &scores {
            scored(outcomes, 4);
        }
        let fake = fake(&owner);
        assert_eq!(fake.releases, 1);
        assert_eq!(
            fake.submitted,
            vec![vec![RequestId(1), RequestId(2), RequestId(3)]]
        );
        assert_eq!(fake.closed, vec![RequestId(1), RequestId(2), RequestId(3)]);
    }

    #[test]
    fn a_capacity_shortage_at_open_releases_before_it_refuses() {
        {
            let mut owner = owner(Fake {
                open_errors: BTreeMap::from([(RequestId(1), capacity_error())]),
                surplus: true,
                ..Fake::default()
            });
            let opened = score(&mut owner, LIKELY, 4);
            settle(&mut owner, 0).unwrap();
            scored(&opened, 4);
            assert_eq!(fake(&owner).releases, 1);
        }
        // With nothing to release, a score that cannot be opened alone fails.
        let mut owner = owner(Fake {
            open_errors: BTreeMap::from([(RequestId(1), capacity_error())]),
            ..Fake::default()
        });
        let refused = score(&mut owner, LIKELY, 4);
        settle(&mut owner, 0).unwrap();
        assert!(matches!(answer(&refused), Err(RequestError::Capacity(_))));
        assert_eq!(fake(&owner).releases, 1);
    }

    /// A failed commit fails its walk alone; the executor leaves the request
    /// resolved, so its sequence closes.
    #[test]
    fn a_walk_whose_rows_cannot_be_committed_fails_alone_and_closes() {
        let unreadable = DomainError::TableRead(magnitude_executor::TableReadError {
            detail: "short read".into(),
        });
        let mut owner = owner(Fake {
            commit_errors: BTreeMap::from([(RequestId(2), unreadable)]),
            ..Fake::default()
        });
        let length = SCORING_WIDTH + 5;
        let scores = (0..3)
            .map(|_| score(&mut owner, LIKELY, length))
            .collect::<Vec<_>>();
        settle(&mut owner, 0).unwrap();
        scored(&scores[0], length);
        assert!(matches!(
            answer(&scores[1]),
            Err(RequestError::TableRead(_))
        ));
        scored(&scores[2], length);
        let fake = fake(&owner);
        assert_eq!(fake.closed[0], RequestId(2));
        assert!(fake.open.is_empty());
    }

    #[test]
    fn a_score_that_cannot_be_stepped_alone_fails_with_the_shortage() {
        let oversized = 7;
        let mut owner = owner(Fake {
            oversized: Some(TokenId(oversized)),
            ..Fake::default()
        });
        let fits = score(&mut owner, LIKELY, 6);
        let too_large = score(&mut owner, oversized, 6);
        settle(&mut owner, 0).unwrap();
        scored(&fits, 6);
        assert!(matches!(answer(&too_large), Err(RequestError::Capacity(_))));
        assert!(fake(&owner).open.is_empty());
    }

    #[test]
    fn a_walk_whose_row_the_reduction_refuses_ends_alone() {
        let mut owner = owner(Fake {
            poisoned: BTreeSet::from([RequestId(2)]),
            ..Fake::default()
        });
        let length = SCORING_WIDTH + 5;
        let scores = (0..3)
            .map(|_| score(&mut owner, LIKELY, length))
            .collect::<Vec<_>>();
        settle(&mut owner, 0).unwrap();
        scored(&scores[0], length);
        assert!(matches!(
            answer(&scores[1]),
            Err(RequestError::Invariant(_))
        ));
        scored(&scores[2], length);
        assert!(fake(&owner).open.is_empty());
    }

    #[test]
    fn close_while_serving_answers_every_waiting_and_active_score_once() {
        let mut owner = owner(Fake::default());
        let scores = (0..SCORING_SEQUENCES + 2)
            .map(|_| score(&mut owner, LIKELY, SCORING_WIDTH + 5))
            .collect::<Vec<_>>();
        // One step: S walks are active, two scores wait.
        assert_eq!(owner.run(0), Ok(0));
        assert_eq!(scorer(&mut owner).walks.len(), SCORING_SEQUENCES);
        owner.event(Event::Close, 0);
        for outcomes in &scores {
            assert_eq!(answer(outcomes), Err(RequestError::WorkerClosed));
        }
        assert_eq!(owner.run(0), Err(UnloadCause::Shutdown));
    }

    #[test]
    fn a_device_failure_fails_every_score_and_stops() {
        let lost = DomainError::Device(DeviceError::Lost("gone".into()));
        let mut owner = owner(Fake {
            finish_error: Some(lost),
            ..Fake::default()
        });
        let scores = (0..SCORING_SEQUENCES + 1)
            .map(|_| score(&mut owner, LIKELY, 4))
            .collect::<Vec<_>>();
        assert_eq!(
            owner.run(0),
            Err(UnloadCause::DeviceLost {
                reason: "gone".into()
            })
        );
        for outcomes in &scores {
            assert_eq!(
                answer(outcomes),
                Err(RequestError::Device(DeviceError::Lost("gone".into())))
            );
        }
        assert_eq!(
            owner.run(1),
            Err(UnloadCause::DeviceLost {
                reason: "gone".into()
            })
        );
    }

    #[test]
    fn failed_memory_observations_are_tolerated() {
        let mut owner = owner(Fake {
            memory: Some(DomainError::Blind("no reading".into())),
            ..Fake::default()
        });
        let outcomes = score(&mut owner, LIKELY, 4);
        settle(&mut owner, 0).unwrap();
        scored(&outcomes, 4);
    }

    #[test]
    fn reclaim_defers_the_walks_pauses_and_resumes_once_memory_is_normal() {
        let mut owner = owner(Fake::default());
        let length = SCORING_WIDTH + 5;
        let scores = (0..2)
            .map(|_| score(&mut owner, LIKELY, length))
            .collect::<Vec<_>>();
        assert_eq!(owner.run(0), Ok(0), "one step of both walks");
        scorer(&mut owner).domain.memory = Some(DomainError::Reclaim);
        let paused = OBSERVATION_INTERVAL_NS;
        assert_eq!(settle(&mut owner, paused), Ok(2 * paused));
        let held = scorer(&mut owner);
        assert!(held.walks.is_empty(), "Reclaim releases the walks");
        assert_eq!(held.waiting.len(), 2);
        assert!(held.domain.open.is_empty());
        for outcomes in &scores {
            assert!(outcomes.try_recv().is_err(), "a deferred score waits");
        }
        held.domain.memory = None;
        settle(&mut owner, 2 * paused).unwrap();
        for outcomes in &scores {
            scored(outcomes, length);
        }
    }

    #[test]
    fn reclaim_that_persists_unloads_the_model() {
        let mut owner = owner(Fake {
            memory: Some(DomainError::Reclaim),
            ..Fake::default()
        });
        let outcomes = score(&mut owner, LIKELY, 4);
        assert_eq!(settle(&mut owner, 0), Ok(OBSERVATION_INTERVAL_NS));
        assert_eq!(
            settle(&mut owner, MEMORY_ESCALATION_NS - 1),
            Ok(MEMORY_ESCALATION_NS - 1 + OBSERVATION_INTERVAL_NS)
        );
        assert!(
            outcomes.try_recv().is_err(),
            "the score waits while memory may recover"
        );
        assert_eq!(
            owner.run(MEMORY_ESCALATION_NS + OBSERVATION_INTERVAL_NS),
            Err(UnloadCause::MemoryPressure)
        );
        assert_eq!(
            answer(&outcomes),
            Err(RequestError::ModelUnloaded {
                cause: ModelUnloadCause::MemoryPressure
            })
        );
    }

    /// One escalation interval of failed observations counts as Reclaim
    /// whose unload clock has expired, idle or not.
    #[test]
    fn failed_observations_that_persist_unload_the_model() {
        let mut owner = owner(Fake {
            memory: Some(DomainError::Blind("no reading".into())),
            ..Fake::default()
        });
        assert_eq!(settle(&mut owner, 0), Ok(OBSERVATION_INTERVAL_NS));
        let before = MEMORY_ESCALATION_NS - 1;
        assert_eq!(
            settle(&mut owner, before),
            Ok(before + OBSERVATION_INTERVAL_NS)
        );
        let outcomes = score(&mut owner, LIKELY, 4);
        assert_eq!(
            owner.run(MEMORY_ESCALATION_NS + OBSERVATION_INTERVAL_NS),
            Err(UnloadCause::MemoryPressure)
        );
        assert_eq!(
            answer(&outcomes),
            Err(RequestError::ModelUnloaded {
                cause: ModelUnloadCause::MemoryPressure
            })
        );
    }

    #[test]
    fn a_growth_refused_for_memory_waits_for_normal_holding_its_walks() {
        let mut owner = owner(Fake::default());
        let outcomes = score(&mut owner, LIKELY, 4);
        scorer(&mut owner)
            .domain
            .open_errors
            .insert(RequestId(1), DomainError::Reclaim);
        assert_eq!(settle(&mut owner, 0), Ok(OBSERVATION_INTERVAL_NS));
        assert!(outcomes.try_recv().is_err());
        assert_eq!(scorer(&mut owner).waiting.len(), 1);
        settle(&mut owner, OBSERVATION_INTERVAL_NS).unwrap();
        scored(&outcomes, 4);
    }

    /// The step keeps its walks and the bindings come back: once memory is
    /// Normal the same walks step on.
    #[test]
    fn a_step_refused_for_memory_holds_its_walks_until_memory_is_normal() {
        let mut owner = owner(Fake {
            reserve_refusal: Some(DomainError::Reclaim),
            ..Fake::default()
        });
        let scores = (0..2)
            .map(|_| score(&mut owner, LIKELY, 4))
            .collect::<Vec<_>>();
        assert_eq!(settle(&mut owner, 0), Ok(OBSERVATION_INTERVAL_NS));
        let held = scorer(&mut owner);
        assert_eq!(held.walks.len(), 2, "the walks are held, not deferred");
        assert!(held.domain.closed.is_empty());
        settle(&mut owner, OBSERVATION_INTERVAL_NS).unwrap();
        for outcomes in &scores {
            scored(outcomes, 4);
        }
        assert_eq!(
            fake(&owner).submitted,
            vec![vec![RequestId(1), RequestId(2)]]
        );
    }

    #[test]
    fn a_submission_refused_for_capacity_defers_the_newest_walk() {
        let mut owner = owner(Fake {
            submit_refusal: Some(capacity_error()),
            ..Fake::default()
        });
        let scores = (0..2)
            .map(|_| score(&mut owner, LIKELY, 4))
            .collect::<Vec<_>>();
        settle(&mut owner, 0).unwrap();
        for outcomes in &scores {
            scored(outcomes, 4);
        }
        let fake = fake(&owner);
        assert_eq!(fake.closed[0], RequestId(2), "the newest walk was deferred");
        assert_eq!(fake.submitted[0], vec![RequestId(1)]);
    }

    #[test]
    fn a_serving_scoring_owner_reports_zero_output_throughput() {
        let mut owner = owner(Fake::default());
        assert!(matches!(
            owner.command(WorkerCommand::Throughput, 0),
            Ok(WorkerReply::Throughput(ThroughputSnapshot {
                window_ms: WINDOW_MS,
                ready: true,
                accepted_tokens: 0,
            }))
        ));
    }

    #[test]
    fn a_scoring_owner_refuses_generation() {
        let WorkerReply::AdmissionRefused(AdmissionError::Refused(RequestError::Input(reason))) =
            refuse_generation()
        else {
            panic!("the scoring owner refuses an admission as invalid input");
        };
        assert_eq!(reason, "this load serves scoring, not generation");
    }

    #[test]
    fn domain_errors_are_met_as_the_generation_owner_meets_them() {
        assert!(matches!(
            Setback::of(DomainError::Reclaim),
            Setback::Memory(MemoryObservation::Reclaim)
        ));
        assert!(matches!(
            Setback::of(DomainError::Blind("no reading".into())),
            Setback::Memory(MemoryObservation::Blind)
        ));
        assert!(matches!(
            Setback::of(capacity_error()),
            Setback::Capacity(RequestError::Capacity(_))
        ));
        assert!(matches!(
            Setback::of(DomainError::Input("bad launch".into())),
            Setback::Request(RequestError::Input(_))
        ));
        let Setback::Fatal(lost) =
            Setback::of(DomainError::Device(DeviceError::Lost("gone".into())))
        else {
            panic!("a lost device stops execution");
        };
        assert_eq!(
            lost.cause,
            UnloadCause::DeviceLost {
                reason: "gone".into()
            }
        );
        let Setback::Fatal(broken) = Setback::of(invariant("broken")) else {
            panic!("a broken invariant stops execution");
        };
        assert!(matches!(broken.cause, UnloadCause::Internal { .. }));
    }

    fn stopped_owner(cause: UnloadCause) -> ScoringOwner<Fake> {
        ScoringOwner {
            execution: Execution::Stopped(cause),
        }
    }

    /// A stopped owner answers every score through its callback, once, and
    /// acknowledges the command.
    #[test]
    fn a_stopped_owner_fails_each_score_through_its_callback() {
        for (cause, expected) in [
            (UnloadCause::Shutdown, RequestError::WorkerClosed),
            (
                UnloadCause::MemoryPressure,
                RequestError::ModelUnloaded {
                    cause: ModelUnloadCause::MemoryPressure,
                },
            ),
        ] {
            let mut owner = stopped_owner(cause.clone());
            let outcomes = score(&mut owner, LIKELY, 2);
            assert_eq!(answer(&outcomes), Err(expected));
            assert_eq!(owner.run(0), Err(cause));
        }
    }

    #[test]
    fn a_stopped_owner_keeps_its_cause_through_close_and_has_no_status() {
        let mut owner = stopped_owner(UnloadCause::MemoryPressure);
        owner.event(Event::Close, 0);
        assert_eq!(owner.run(0), Err(UnloadCause::MemoryPressure));
        assert!(matches!(
            owner.command(
                WorkerCommand::Status {
                    request: RequestId(1)
                },
                0
            ),
            Ok(WorkerReply::Status(None))
        ));
        assert!(owner.command(WorkerCommand::Observe, 0).is_err());
        assert!(owner.command(WorkerCommand::Throughput, 0).is_err());
    }
}
