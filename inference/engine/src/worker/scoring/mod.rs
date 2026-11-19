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
//!   A shortage first takes the relief the ledger offers of the load's
//!   cache, as the generation owner's does;
//! - memory is read where the generation owner reads it: at each step's
//!   end (a flight boundary), at an OS pressure event, at the worker's timed
//!   reading, and after a reading a change beside deferred. A reading's
//!   relief defers the walks it names, new work waits while it says so, and
//!   a shortage that persists with nothing to give unloads the model.
//!
//! Execution stops only on the errors that stop the generation owner (device,
//! state and invariant failures) and on a logits row that cannot be read.
//! Other errors fail the scores they concern. A step runs to its end within
//! one `run`, so nothing is in flight between calls: while walks remain, the
//! owner wakes itself, and commands and events that arrive during a step
//! apply before the next one.

mod domain;
mod walk;

use crate::error::UnloadCause;
use crate::options::SCORING_SEQUENCES;
use domain::{invariant, Forwarded, Freed, Read, ScoringDomain, Unsubmitted};
use magnitude_executor::{CapacityError, DeviceError, DomainError, ExecutorDomain, RequestId};
use magnitude_family_contracts::ModelDefinition;
use magnitude_scheduler::{
    owner::{is_fatal, AdmissionError},
    protocol::{RoomGrant, RoomRefusal, ScoreRequest, WorkerCommand, WorkerReply},
    publication::{ModelUnloadCause, RequestError},
    scoring::ScoreOutcome,
    throughput::{ThroughputSnapshot, WINDOW_MS},
    worker::{Driven, Event, Reread, Wakes},
};
use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;
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
    /// and vocabulary of the load's `definition`, woken through `wakes`.
    pub fn new(
        domain: D,
        bindings: D::Bindings,
        definition: &ModelDefinition,
        wakes: Arc<dyn Wakes>,
    ) -> Result<Self, String> {
        Ok(Self::serving(domain, bindings, Bounds::of(definition)?, wakes))
    }

    fn serving(domain: D, bindings: D::Bindings, bounds: Bounds, wakes: Arc<dyn Wakes>) -> Self {
        Self {
            execution: Execution::Serving(Scorer {
                domain,
                bindings: Some(bindings),
                bounds,
                waiting: VecDeque::new(),
                walks: Vec::with_capacity(SCORING_SEQUENCES),
                next_sequence: 1,
                capacity_blocked: false,
                admission_waits: false,
                // The load's first reading is owed.
                read_owed: true,
                held: false,
                wakes,
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
            WorkerCommand::ClaimMeasurementRoom { bytes } => Ok(WorkerReply::MeasurementRoom(match &mut self.execution {
                Execution::Serving(scorer) => RoomGrant::decided(
                    scorer
                        .domain
                        .claim_measurement_room(bytes)
                        .map_err(|bytes| RoomRefusal::Short { bytes }),
                ),
                // An ending model holds no claim; its end answers the host.
                Execution::Stopped(_) => RoomGrant::released(),
            })),
            WorkerCommand::AttachMeasurementProcess { process } => {
                if let Execution::Serving(scorer) = &mut self.execution {
                    scorer.domain.attach_measurement_process(process);
                }
                Ok(WorkerReply::Acknowledged)
            }
            WorkerCommand::ReleaseMeasurementRoom => {
                if let Execution::Serving(scorer) = &mut self.execution {
                    scorer.domain.release_measurement_room();
                }
                Ok(WorkerReply::Acknowledged)
            }
            // A scoring load prices no decode: newly stored costs change
            // nothing it does.
            WorkerCommand::Reprice => Ok(WorkerReply::Acknowledged),
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
            // is not cancellable. A completion ends a release a step or an
            // open waited on, or is the owner's own wake: its run follows.
            Event::Completion | Event::Publication(..) | Event::Cancel(_) => {}
            // The run that follows reads memory.
            Event::MemoryPressure | Event::Reread => {
                if let Execution::Serving(scorer) = &mut self.execution {
                    scorer.read_owed = true;
                }
            }
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

    fn run(&mut self, _: u64) -> Result<Reread, UnloadCause> {
        match self.take() {
            Execution::Serving(mut scorer) => match scorer.run() {
                Ok(reread) => {
                    self.execution = Execution::Serving(scorer);
                    Ok(reread)
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

    /// A shortage persists with nothing to give.
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
    /// The domain's capacity is short, memory among it: relief, then fewer
    /// walks, or the one alone fails with `error`.
    Capacity { capacity: CapacityError, error: RequestError },
    /// The scores the error concerns fail; execution continues.
    Request(RequestError),
    Fatal(Fatal),
}

impl Setback {
    fn of(error: DomainError) -> Self {
        match error {
            DomainError::Capacity(capacity) => Self::Capacity {
                error: RequestError::from(DomainError::Capacity(capacity.clone())),
                capacity,
            },
            error if is_fatal(&error) => Self::Fatal(Fatal::stop(error)),
            other => Self::Request(RequestError::from(other)),
        }
    }
}

/// What a step that met a setback does next.
enum Retry {
    /// Step again at once: relief freed memory, or a walk was deferred.
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
    /// The last reading holds new work back.
    admission_waits: bool,
    /// The next run reads memory.
    read_owed: bool,
    /// What this run would do waits on a release off this thread, whose
    /// end wakes the owner.
    held: bool,
    wakes: Arc<dyn Wakes>,
}

impl<D: ScoringDomain> Scorer<D> {
    /// Read memory where owed, admit waiting scores while fewer than S walk
    /// and capacity and memory allow, then take one step of every walk,
    /// whose end reads memory. While there is work to do at once (walks to
    /// step, or a score to admit) the owner wakes itself, so what arrived
    /// meanwhile applies first; what waits on a release is woken by its end
    /// instead.
    fn run(&mut self) -> Result<Reread, Fatal> {
        self.held = false;
        let mut reread = Reread::Unread;
        if self.domain.pressured() | self.read_owed {
            reread = self.observe()?;
        }
        self.admit()?;
        if !self.walks.is_empty() {
            self.step()?;
            reread = self.observe()?;
        }
        if self.walks.is_empty() {
            // Waiting for work is no launch's service.
            self.domain.idle();
        }
        if self.ready() {
            self.wakes.completion().complete();
            // The next run's step reads memory at its end.
            return Ok(Reread::Settled);
        }
        Ok(reread)
    }

    /// Whether the owner has work to do at once: walks to step or a score
    /// to admit, waiting on no release.
    fn ready(&self) -> bool {
        !self.held && (!self.walks.is_empty() || self.admissible())
    }

    /// Whether a waiting score may be opened: memory lets new work in, a
    /// walk is free, and no walk short of capacity waits for another to end.
    fn admissible(&self) -> bool {
        !self.admission_waits
            && !self.waiting.is_empty()
            && self.walks.len() < SCORING_SEQUENCES
            && (!self.capacity_blocked || self.walks.is_empty())
    }

    /// One reading of memory and the scorer's response, in the generation
    /// owner's order: what the reading took of the cache is freed, a
    /// shortage with nothing to give unloads, the walks a relief names are
    /// deferred, and new work waits while the reading says so.
    fn observe(&mut self) -> Result<Reread, Fatal> {
        // Any reading answers the one owed.
        self.read_owed = false;
        let walks = self.walks.iter().map(Walk::sequence).collect::<BTreeSet<_>>();
        let read = self.domain.read_memory(&walks).map_err(Fatal::stop)?;
        self.follow()?;
        let Some(read) = read else {
            // A change of ours stood beside the sample: its completion wakes
            // the owner, whose run reads then.
            self.read_owed = true;
            return Ok(Reread::Unread);
        };
        let Read { unload, preempt, admission_waits, moved, cost, stands, ceiling } = read;
        if unload {
            tracing::error!(
                scores = self.walks.len() + self.waiting.len(),
                "unloading the model: a real memory shortage persists with nothing to give"
            );
            return Err(Fatal::memory_pressure());
        }
        let preempted = !preempt.is_empty();
        if preempted {
            tracing::info!(walks = ?preempt, "a real memory shortage defers the walks worth least");
            self.defer(&preempt)?;
            self.follow()?;
        }
        let waited = std::mem::replace(&mut self.admission_waits, admission_waits);
        if waited && !admission_waits {
            // Memory lets new work in again: so does capacity.
            self.capacity_blocked = false;
        }
        let changed = preempted || waited != admission_waits || moved;
        Ok(match stands {
            true => Reread::Stands { cost, changed, ceiling },
            false => Reread::Settled,
        })
    }

    /// Free what the ledger took of the load's cache.
    fn follow(&mut self) -> Result<Freed, Fatal> {
        let bindings = self.bindings.as_mut().ok_or_else(Fatal::bindings_lost)?;
        let wakes = Arc::clone(&self.wakes);
        self.domain
            .follow_memory(bindings, &move || wakes.completion())
            .map_err(Fatal::stop)
    }

    /// The relief for a claim `capacity` refused ([`ScoringDomain::relieve`]).
    /// One that frees memory only later holds what claimed it until its end
    /// wakes the owner.
    fn relieve(&mut self, bindings: &mut D::Bindings, capacity: &CapacityError) -> Result<Freed, Fatal> {
        let wakes = Arc::clone(&self.wakes);
        let freed = self
            .domain
            .relieve(bindings, capacity, &move || wakes.completion())
            .map_err(Fatal::stop)?;
        if freed == Freed::Later {
            self.held = true;
        }
        Ok(freed)
    }

    fn admit(&mut self) -> Result<(), Fatal> {
        while self.admissible() && !self.held {
            let Some(request) = self.waiting.pop_front() else {
                break;
            };
            let sequence = RequestId(self.next_sequence);
            self.next_sequence += 1;
            // The walk writes a row for each of its tokens.
            let rows = request.tokens.len() as u64;
            let opened = match self.open(sequence, rows) {
                Ok(opened) => opened,
                Err(fatal) => {
                    (request.done)(Err(fatal.error.clone()));
                    return Err(fatal);
                }
            };
            let error = match opened {
                Opened::Open => {
                    self.walks.push(Walk::new(request, sequence));
                    continue;
                }
                // It opens once the release it waits on ends.
                Opened::Held => {
                    self.waiting.push_front(request);
                    break;
                }
                Opened::Refused(error) => error,
            };
            match Setback::of(error) {
                // It cannot be opened even alone.
                Setback::Capacity { error, .. } if self.walks.is_empty() => (request.done)(Err(error)),
                Setback::Capacity { .. } => {
                    self.waiting.push_front(request);
                    self.capacity_blocked = true;
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

    /// Open `sequence`. A capacity shortage first takes the relief the
    /// ledger offers and opens again, as the generation owner's claims do.
    fn open(&mut self, sequence: RequestId, rows: u64) -> Result<Opened, Fatal> {
        let mut bindings = self.bindings.take().ok_or_else(Fatal::bindings_lost)?;
        let opened = self.open_with(&mut bindings, sequence, rows);
        self.bindings = Some(bindings);
        opened
    }

    fn open_with(&mut self, bindings: &mut D::Bindings, sequence: RequestId, rows: u64) -> Result<Opened, Fatal> {
        let opened = self.domain.open_sequence(bindings, sequence, rows);
        let Err(DomainError::Capacity(capacity)) = &opened else {
            return Ok(Opened::of(opened));
        };
        Ok(match self.relieve(bindings, &capacity.clone())? {
            Freed::Now => Opened::of(self.domain.open_sequence(bindings, sequence, rows)),
            Freed::Later => Opened::Held,
            Freed::Nothing => Opened::of(opened),
        })
    }

    /// One launch of every walk's next rows, waited for and committed; each
    /// walk takes its rows, and every walk that ended is closed and answered.
    /// A step the domain pushes back on is retried with fewer walks, or on a
    /// later run.
    fn step(&mut self) -> Result<(), Fatal> {
        let mut bindings = self.bindings.take().ok_or_else(Fatal::bindings_lost)?;
        // Whether relief was asked for since the walks last changed.
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
            let retry = self.setback(refusal, &mut bindings, &mut released);
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

    /// Meet a refused step. A capacity shortage first takes the relief the
    /// ledger offers, once per set of walks, and steps again, or waits for a
    /// release under way; still short, it defers the newest walk when there
    /// are more, and fails the walk that cannot be stepped alone. Other
    /// errors fail the step's walks.
    fn setback(
        &mut self,
        error: DomainError,
        bindings: &mut D::Bindings,
        released: &mut bool,
    ) -> Result<Retry, Fatal> {
        let failed = match Setback::of(error) {
            Setback::Capacity { capacity, error } => {
                if !*released {
                    *released = true;
                    match self.relieve(bindings, &capacity)? {
                        Freed::Now => return Ok(Retry::Now),
                        Freed::Later => return Ok(Retry::Later),
                        Freed::Nothing => {}
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

    /// Defer every walk of `sequences`, the newest first, so they queue in
    /// their order at the front.
    fn defer(&mut self, sequences: &BTreeSet<RequestId>) -> Result<(), Fatal> {
        for index in (0..self.walks.len()).rev() {
            if sequences.contains(&self.walks[index].sequence()) {
                let walk = self.walks.remove(index);
                self.defer_walk(walk)?;
            }
        }
        Ok(())
    }

    /// Close the newest walk's sequence and put its request back at the
    /// front of the queue, its progress dropped.
    fn defer_newest(&mut self) -> Result<(), Fatal> {
        let walk = self
            .walks
            .pop()
            .ok_or_else(|| Fatal::invariant("no walk to defer"))?;
        self.defer_walk(walk)
    }

    fn defer_walk(&mut self, walk: Walk) -> Result<(), Fatal> {
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

/// What opening a sequence came to.
enum Opened {
    Open,
    /// Short of memory a release under way frees: it opens once that ends.
    Held,
    Refused(DomainError),
}

impl Opened {
    fn of(opened: Result<(), DomainError>) -> Self {
        match opened {
            Ok(()) => Self::Open,
            Err(error) => Self::Refused(error),
        }
    }
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
    use magnitude_executor::{CompletionWake, Operation, ResourceKind, TokenId};
    use magnitude_scheduler::publication::PublicationWake;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
        /// The ledger offers relief of the domain's cache: taking it frees
        /// memory and lifts `capacity`.
        surplus: bool,
        /// How often relief was asked for.
        releases: usize,
        /// The next relief frees memory only later, once.
        relief_later: bool,
        /// Sequences whose logits rows hold a NaN.
        poisoned: BTreeSet<RequestId>,
        /// What every reading of memory reads.
        reading: Read,
        /// The next reading meets a change of ours beside it, once.
        beside: bool,
        /// The readings taken.
        readings: usize,
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
            _: u64,
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

        fn follow_memory(
            &mut self,
            _: &mut Bindings,
            _: &dyn Fn() -> CompletionWake,
        ) -> Result<Freed, DomainError> {
            Ok(Freed::Nothing)
        }

        fn relieve(
            &mut self,
            _: &mut Bindings,
            _: &CapacityError,
            _: &dyn Fn() -> CompletionWake,
        ) -> Result<Freed, DomainError> {
            self.releases += 1;
            if std::mem::take(&mut self.relief_later) {
                return Ok(Freed::Later);
            }
            if !std::mem::take(&mut self.surplus) {
                return Ok(Freed::Nothing);
            }
            self.capacity = None;
            Ok(Freed::Now)
        }

        fn read_memory(&mut self, _: &BTreeSet<RequestId>) -> Result<Option<Read>, DomainError> {
            self.readings += 1;
            if std::mem::take(&mut self.beside) {
                return Ok(None);
            }
            Ok(Some(self.reading.clone()))
        }

        fn pressured(&self) -> bool {
            false
        }

        fn idle(&mut self) {}

        fn observe_memory(&self) -> Result<WorkerReply, String> {
            Err("the fake reads no memory".into())
        }

        fn claim_measurement_room(&mut self, _: u64) -> Result<(), u64> {
            Ok(())
        }

        fn attach_measurement_process(&mut self, _: magnitude_executor::ProcessId) {}

        fn release_measurement_room(&mut self) {}
    }

    /// The owner's wakes: how often it woke itself.
    #[derive(Default)]
    struct Woken(Arc<AtomicUsize>);

    impl Wakes for Woken {
        fn completion(&self) -> CompletionWake {
            let woken = Arc::clone(&self.0);
            CompletionWake::new(move || {
                woken.fetch_add(1, Ordering::SeqCst);
            })
        }

        fn publication(&self, _: RequestId, _: PublicationWake) {}

        fn pressure(&self) -> Box<dyn Fn() + Send + Sync> {
            Box::new(|| {})
        }
    }

    fn owner(fake: Fake) -> ScoringOwner<Fake> {
        woken_owner(fake).0
    }

    /// An owner, and how often it woke itself.
    fn woken_owner(fake: Fake) -> (ScoringOwner<Fake>, Arc<AtomicUsize>) {
        let woken = Woken::default();
        let count = Arc::clone(&woken.0);
        let owner = ScoringOwner::serving(
            fake,
            Bindings,
            Bounds {
                context: 256,
                vocabulary: VOCABULARY,
            },
            Arc::new(woken),
        );
        (owner, count)
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

    /// Whether the owner woke itself ([`Scorer::ready`]).
    fn wakes_itself(owner: &ScoringOwner<Fake>) -> bool {
        matches!(&owner.execution, Execution::Serving(scorer) if scorer.ready())
    }

    /// Run the owner at `now` until it no longer wakes itself; the last
    /// run's reading.
    fn settle(owner: &mut ScoringOwner<Fake>, now: u64) -> Result<Reread, UnloadCause> {
        for _ in 0..1000 {
            let reread = owner.run(now)?;
            if !wakes_itself(owner) {
                return Ok(reread);
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
        assert_eq!(settle(&mut owner, 0), Ok(Reread::Settled));
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
        let unreadable = DomainError::Input("short read".into());
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
        assert!(matches!(answer(&scores[1]), Err(RequestError::Input(_))));
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
        assert_eq!(owner.run(0), Ok(Reread::Settled));
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

    /// A reading a change of ours stood beside is owed to the next run,
    /// which reads; any later reading answers it.
    #[test]
    fn a_reading_beside_a_change_is_taken_on_the_next_run() {
        let mut owner = owner(Fake { beside: true, ..Fake::default() });
        assert_eq!(owner.run(0), Ok(Reread::Unread), "the first reading met a change beside");
        assert!(scorer(&mut owner).read_owed);
        assert_eq!(owner.run(0), Ok(Reread::Settled), "the next run reads");
        assert!(!scorer(&mut owner).read_owed);
        let outcomes = score(&mut owner, LIKELY, 4);
        settle(&mut owner, 0).unwrap();
        scored(&outcomes, 4);
        assert_eq!(fake(&owner).readings, 3, "the first, the one owed, the step's end");
    }

    /// A walk steps once per run and the owner wakes itself while walks
    /// remain, so a command that arrives between steps applies before the
    /// next.
    #[test]
    fn a_scorer_with_walks_left_wakes_itself_after_each_step() {
        let (mut owner, woken) = woken_owner(Fake::default());
        let outcomes = score(&mut owner, LIKELY, SCORING_WIDTH + 5);
        assert_eq!(owner.run(0), Ok(Reread::Settled));
        assert_eq!(woken.load(Ordering::SeqCst), 1, "walks remain");
        assert!(outcomes.try_recv().is_err());
        owner.run(0).unwrap();
        scored(&outcomes, SCORING_WIDTH + 5);
        assert_eq!(woken.load(Ordering::SeqCst), 1, "nothing remains");
    }

    /// The step that ends the last walk wakes the owner for a score deferred
    /// for capacity, which no arrival would admit.
    #[test]
    fn the_end_of_the_last_walk_wakes_the_owner_for_a_deferred_score() {
        let (mut owner, woken) = woken_owner(Fake {
            capacity: Some(1),
            ..Fake::default()
        });
        let scores = (0..2)
            .map(|_| score(&mut owner, LIKELY, 4))
            .collect::<Vec<_>>();
        owner.run(0).unwrap();
        scored(&scores[0], 4);
        assert_eq!(scorer(&mut owner).waiting.len(), 1, "the newest was deferred");
        assert_eq!(woken.load(Ordering::SeqCst), 1, "the deferred score is admissible");
        settle(&mut owner, 0).unwrap();
        scored(&scores[1], 4);
    }

    /// A reading's relief defers the walks it names, newest first, their
    /// sequences closed; they queue in order and score once memory allows.
    #[test]
    fn a_relief_that_names_walks_defers_them_and_they_score_later() {
        let mut owner = owner(Fake::default());
        // Three steps long: the relief comes at the second step's end.
        let length = 2 * SCORING_WIDTH + 5;
        let scores = (0..2)
            .map(|_| score(&mut owner, LIKELY, length))
            .collect::<Vec<_>>();
        assert_eq!(owner.run(0), Ok(Reread::Settled), "one step of both walks");
        scorer(&mut owner).domain.reading = Read {
            preempt: [RequestId(1), RequestId(2)].into(),
            admission_waits: true,
            stands: true,
            ..Read::default()
        };
        assert!(matches!(owner.run(0), Ok(Reread::Stands { changed: true, .. })), "a step, then the relief");
        let held = scorer(&mut owner);
        assert!(held.walks.is_empty(), "the relief defers the walks");
        assert_eq!(held.waiting.len(), 2);
        assert!(held.domain.open.is_empty());
        assert_eq!(held.domain.closed, vec![RequestId(2), RequestId(1)], "newest first");
        for outcomes in &scores {
            assert!(outcomes.try_recv().is_err(), "a deferred score waits");
        }
        // The worker's timed reading finds memory free again.
        held.domain.reading = Read::default();
        owner.event(Event::Reread, 0);
        settle(&mut owner, 0).unwrap();
        for outcomes in &scores {
            scored(outcomes, length);
        }
    }

    /// New work waits while a reading says so; the reading that lets it in
    /// admits it at once.
    #[test]
    fn new_work_waits_while_a_reading_holds_admission() {
        let mut owner = owner(Fake {
            reading: Read { admission_waits: true, stands: true, ..Read::default() },
            ..Fake::default()
        });
        let outcomes = score(&mut owner, LIKELY, 4);
        assert!(matches!(owner.run(0), Ok(Reread::Stands { .. })));
        assert!(outcomes.try_recv().is_err());
        assert_eq!(scorer(&mut owner).waiting.len(), 1);
        scorer(&mut owner).domain.reading = Read::default();
        owner.event(Event::MemoryPressure, 0);
        settle(&mut owner, 0).unwrap();
        scored(&outcomes, 4);
    }

    #[test]
    fn a_shortage_with_nothing_to_give_unloads_the_model() {
        let mut owner = owner(Fake {
            reading: Read { unload: true, ..Read::default() },
            ..Fake::default()
        });
        let outcomes = score(&mut owner, LIKELY, 4);
        assert_eq!(owner.run(0), Err(UnloadCause::MemoryPressure));
        assert_eq!(
            answer(&outcomes),
            Err(RequestError::ModelUnloaded {
                cause: ModelUnloadCause::MemoryPressure
            })
        );
    }

    /// The step keeps its walks and the bindings come back while a release
    /// it waits on runs: the owner does not wake itself, and the release's
    /// end steps the same walks on.
    #[test]
    fn a_step_whose_relief_frees_later_holds_its_walks_until_the_release_ends() {
        let (mut owner, woken) = woken_owner(Fake {
            reserve_refusal: Some(capacity_error()),
            relief_later: true,
            ..Fake::default()
        });
        let scores = (0..2)
            .map(|_| score(&mut owner, LIKELY, 4))
            .collect::<Vec<_>>();
        owner.run(0).unwrap();
        let held = scorer(&mut owner);
        assert_eq!(held.walks.len(), 2, "the walks are held, not deferred");
        assert!(held.held);
        assert!(held.domain.closed.is_empty());
        assert_eq!(woken.load(Ordering::SeqCst), 0, "the release's end wakes the owner");
        owner.event(Event::Completion, 0);
        settle(&mut owner, 0).unwrap();
        for outcomes in &scores {
            scored(outcomes, 4);
        }
        assert_eq!(fake(&owner).submitted, vec![vec![RequestId(1), RequestId(2)]]);
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
            Setback::of(capacity_error()),
            Setback::Capacity { error: RequestError::Capacity(_), .. }
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
