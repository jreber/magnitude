//! Serialized ownership of logical generation and one executor resource domain.
//!
//! The owner is the one schedule. Admission is logical: it records a request
//! and replies. Residency, rounds, reclamation and eviction all happen in
//! [`Owner::run`], between flights, where the owner holds the
//! [`StateBindings`] that every binding change requires. The pipeline waits
//! only on its own flight; a request that cannot proceed waits alone, in
//! exactly one [`State`]. Teacher-forced scores are the owner's work too:
//! their walks' flights are flights of the same pipeline ([`Owner::score`]).
use super::{
    domain::{
        DomainFlight, ExecutorDomain, FlightOutcomes, OperationGroup, ResumeState, finish, group,
        requirements, submit_group,
    },
    policy::{
        AvailabilityEpoch, Operation as ScheduledOperation, Phase, Scheduler, Selection,
        ServiceLimits,
    },
    prefix_cache::{
        CachedPrefixKeyInfo, CachePoint, MIN_BRANCH_GAIN, MIN_PREFIX_HIT, PrefixCache,
        PrefixCacheCapacity, PrefixPath, PrefixRetention,
    },
    protocol::{RequestSnapshot, RoomGrant, RoomRefusal, ScoreRequest, WorkerReply},
    publication::{
        CapacityResource, ModelUnloadCause, PublicationPermit, PublicationQueue,
        PublicationReceiver, PublicationSender, PublicationWake, PublicationWakeKind, PublishError,
        RequestError, RequestTimings, Retirement, ServiceCapacityError,
    },
    round_driver::{RoundError, lower_round, reconcile_forward},
    scoring::{
        ScoreOutcome,
        walk::{Absorbed, Bounds, Shape, Turn, Walk, Walks},
    },
    serving::ServingCensus,
    throughput::{ThroughputAccumulator, ThroughputSnapshot},
    worker::{Reread, Wakes},
};
use magnitude_executor::{
    BankResize, BankResized, BankStep, BankTier, DeviceError, DomainError, Followed, LoadMeasures, InvariantError,
    MemoryReading, StateBytes, StoreAllocation, NativeFamily, SizingTerm, SizingTerms,
    Operation, Outcome, PendingOperationOutcome, PhysicalDecision,
    ProgramFamily, RequestId, ResourceKind, ResourcePlan, RowResult, StateBindings, SubmitError,
    BuiltForm, FrontGrant, FrontPass, GuessKind, SubmitFailure, TargetContinuation, WorkKind, Yielded,
};
use magnitude_executor::memory::ledger::Relief;
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::{
    DraftCheckpoint, FinishReason, Generation, InputLayout, MethodCheckpoint, OutputToken,
    RoundStart, StartedRound, TimedLaunch,
};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    cell::RefCell,
    num::NonZeroU64,
    path::PathBuf,
    rc::Rc,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::cold_tier::{
    ColdTier, EntryKey, MethodTag, PointTag, SpilledCheckpoint, decode_spill, encode_spill,
    entry_key, prefer_disk,
};
use magnitude_state::ExportedState;

/// Where a newly admitted request stops its prefill to retain branch
/// points, and the shared prefix it waits for a live peer to retain.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct BranchPlan {
    branches: BTreeSet<usize>,
    awaited: Option<(RequestId, usize)>,
}

/// Whether retaining a state at `point` is worth a split prefill chunk and a
/// retained state: a hit, inside the prompt, and at least
/// [`MIN_BRANCH_GAIN`] rows beyond `from`, where the request resumes.
fn worthwhile(point: usize, from: usize, prompt: usize) -> bool {
    point >= MIN_PREFIX_HIT && point >= from + MIN_BRANCH_GAIN && point < prompt
}

/// Plan a branch point at `point` when it is worthwhile over both `from`
/// and the deepest branch point already planned below it.
fn plan_branch(branches: &mut BTreeSet<usize>, point: usize, from: usize, prompt: usize) {
    let below = branches.range(..point).next_back().copied().unwrap_or(0);
    if !branches.contains(&point) && worthwhile(point, from.max(below), prompt) {
        branches.insert(point);
    }
}

/// Plan a declared cache point at `point` when it is a hit inside the prompt
/// beyond `from`, where the request resumes. Unlike a discovered divergence,
/// a declared point is not held to [`MIN_BRANCH_GAIN`]: the host declares it
/// where later requests are expected to diverge, and each turn of a
/// conversation resumes at the point its previous turn declared. A turn
/// shorter than the gain would otherwise retain nothing beyond where it
/// resumed, so every following turn would resume there too and recompute
/// all the turns since.
fn plan_declared(branches: &mut BTreeSet<usize>, point: usize, from: usize, prompt: usize) {
    if point >= MIN_PREFIX_HIT && point > from && point < prompt {
        branches.insert(point);
    }
}

/// Where a peer's prefill stops within the prefix it shares with a new
/// request, up to `shared` rows: its deepest planned point there. That is
/// `shared` itself, or a point fewer than [`MIN_BRANCH_GAIN`] rows below it
/// (such as a declared cache point) that kept `shared` from being planned.
fn peer_stop(branches: &BTreeSet<usize>, shared: usize) -> Option<usize> {
    branches.range(..=shared).next_back().copied()
}

/// The prefix a new request waits for: `in_flight`, the deepest stop a peer
/// is still prefilling toward, when it is worth resuming from over the
/// deepest cached entry (`resumed`) and within [`MIN_BRANCH_GAIN`] rows of
/// the deepest prefix the request shares with a cached path or a peer
/// (`deepest`); otherwise the request plans its own branch. Only an entry is
/// a resume point: a cached path sharing the prefix without an entry ending
/// in it saves the request nothing, so it never stands in for the peer.
fn awaited_prefix(in_flight: usize, deepest: usize, resumed: usize, prompt: usize) -> Option<usize> {
    (worthwhile(in_flight, resumed, prompt) && in_flight + MIN_BRANCH_GAIN > deepest)
        .then_some(in_flight)
}

/// Whether a request outside a round may take the device before a member's
/// queued successor chunk, which would then be discarded: it is unfinished
/// (or still owes its reconciliation) and is not waiting on a prefix that a
/// member computes.
fn competes(request: &Request, status: Status) -> bool {
    !matches!(status, Status::Terminal(_))
        && !matches!(request.state, State::Between(_, Stage::AwaitingPrefix { .. }))
}

/// Where a request's prompt is retained: one row before the prompt end, the
/// deepest point a request with the same prompt can resume from (it must
/// compute the row it samples from). None when that is not an exact input
/// boundary, too short to be a hit, or already retained.
fn prompt_boundary(record: &Record, generation: &Generation) -> Option<usize> {
    let prefix = record.prefix.as_ref()?;
    let boundary = generation.prompt().len().checked_sub(1)?;
    (!prefix.prefill_retained
        && boundary >= MIN_PREFIX_HIT
        && generation.layout().boundary(boundary))
    .then_some(boundary)
}

/// Whether a stopped request's work is retained at `position`, the rows its
/// state reached. A stop lands anywhere in a round, so it is retained only
/// where a request can resume from it: deep enough to be a hit, at an exact
/// input boundary of `layout` (never inside an indivisible media span), and
/// beside a method checkpoint that accounts for the same rows.
fn resumes_from_stop(layout: &InputLayout, position: usize, method: &MethodCheckpoint) -> bool {
    position >= MIN_PREFIX_HIT && layout.boundary(position) && method.accounts_for(position)
}

/// The first position after `position` where the request's prefill stops
/// for the prefix cache: a planned branch point or its prompt boundary.
fn next_stop(record: &Record, generation: &Generation, position: usize) -> Option<usize> {
    let prefix = record.prefix.as_ref()?;
    let branch = prefix.branches.range(position + 1..).next().copied();
    [branch, prompt_boundary(record, generation)]
        .into_iter()
        .flatten()
        .filter(|&stop| stop > position)
        .min()
}

/// The prompt chunks, at most `chunks` of at most `chunk_rows` rows, a lone
/// prefill group's request submits next, when they are known now, so the
/// domain queues them behind this group (a chain, or a pipelined successor;
/// `ExecutorDomain::plan_prompt`). Planning stops at a planned branch or
/// retention point: the chunk ending there is followed by that retention,
/// not planned past.
fn planned_prefill(
    round: &Round,
    operations: &[Operation],
    chunk_rows: usize,
    chunks: usize,
) -> Result<Vec<Operation>, RequestError> {
    let [Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens,
        position,
        ..
    }] = operations
    else {
        return Ok(Vec::new());
    };
    let Some(Member {
        record,
        state: Membership::Started(started),
    }) = round.members.get(request)
    else {
        return Ok(Vec::new());
    };
    let end = position + tokens.len();
    let stop = next_stop(record, started.generation(), *position);
    if stop.is_some_and(|stop| stop <= end) {
        return Ok(Vec::new());
    }
    started
        .planned_prefill(*request, chunk_rows, stop, chunks)
        .map_err(invariant("planned prefill"))
}

/// Whether `operations` are one primed prompt chunk that closes its request's
/// prompt, so its target result publishes before its drafter entry finishes.
/// The prompt's end says so, never the entry's shape: a drafter that enters
/// each row with its successor enters every row of a chunk but a prompt's
/// first.
fn closes_prompt(round: &Round, operations: &[Operation]) -> bool {
    let [Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens,
        position,
        prime: Some(_),
        ..
    }] = operations
    else {
        return false;
    };
    let member = round
        .members
        .get(request)
        .unwrap_or_else(|| panic!("a primed prefill of request {request:?} is not a member of its round"));
    position + tokens.len() == member.state.generation().prompt().len()
}

/// The selection bound `operations` spend: every selected target row, and one
/// per drafting request.
fn selection_charge(operations: &[Operation]) -> usize {
    operations
        .iter()
        .map(|operation| match operation {
            Operation::Forward { .. } => operation.selected_rows(),
            Operation::Head { proposals, .. } => usize::from(!proposals.is_empty()),
            Operation::Encode { .. } => 0,
        })
        .sum()
}

/// Whether `candidate`'s next round is a pass in the form its launch grants
/// (`Generation::chains`): a front pass, entering the MTP head behind it,
/// or a verify pass, entering nothing. A launch's forwards chain all or
/// none (`ExecutorDomain::shares_launch`), and its passes are of the one
/// form its regime prices ([`FrontRows`]), within a chained launch's rows
/// (`ExecutorDomain::chained_rows`), so a decode launch takes the selected
/// requests alike in this with the first.
fn chained(state: &State) -> bool {
    matches!(state, State::Between(generation, Stage::Ready) if generation.chains())
}

/// A decode launch's guessed rows (front-row decode): front rows, or a
/// verify pass's acceptable rows on the chain. A launch may have them only
/// where every request it selected decodes in front passes ([`chained`]),
/// so no request's rows, a drafting request's verify rows among them, are
/// ever held for a guessed row. Such a launch reserves every request
/// before it is priced ([`ReservedRound::reserve`]), so a request turned
/// back, its publication full or closed, is neither counted nor granted:
/// the launch is priced for the requests it builds, by
/// `ExecutorDomain::front_pass`, which owns each pass's form and depth.
/// Every request's round is held to its exact row and its guessed rows, so
/// the launch never holds more rows than its grant counted.
enum FrontRows {
    /// A pass the host builds, every request's in the one `form` its regime
    /// prices: of the requests whose round would run guessed rows
    /// (`Generation::front_guess`; one that would fall back to one row runs
    /// none, and is not counted), which run them and up to how deep, held
    /// to the launch's decode row budget past a row a request
    /// ([`Self::of`]); every other request runs its exact row alone in that
    /// form, so a verify launch's forwards chain none, whether or not any
    /// request is granted a guessed row.
    /// Only a request holding a guess, or drafting its chain first, runs
    /// guessed rows: first those whose first guess is chained, whose chain
    /// of standing guesses a grant continues, then those of the head's
    /// guess alone, each in request order, so a request keeps its guessed
    /// rows from launch to launch while the cap allows, whatever order the
    /// requests are selected in.
    Built {
        form: BuiltForm,
        granted: BTreeMap<RequestId, usize>,
    },
    /// The claim of the pass queued behind the last, whose requests the
    /// launch holds, each at its queued depth: its rows are that pass's,
    /// held to its request's context and a chained launch's rows as it was
    /// queued (`ExecutorDomain::chained_rows`, this launch's budget), its
    /// chain's guesses then idle rows past them, so nothing here reshapes
    /// it.
    Claim(BTreeMap<RequestId, usize>),
}

impl FrontRows {
    /// The guessed rows of a decode launch of the requests it reserved,
    /// each with the kind of the first guess its round would run at the
    /// deepest grant (`passes`), within `rows`, its decode row budget,
    /// `fronts` granting each request's pass given the launch's requests
    /// and those that could front, in the order they take them
    /// (`ExecutorDomain::front_pass`): the claim of the queued pass, or a
    /// pass the host builds, of one form. Of the latter, guessed rows take
    /// rows of the budget beyond their request's one, so a grant is no
    /// deeper than the budget holds past a row a request, and grants stop,
    /// in that order, at the first the rows left do not hold. A launch
    /// reserves no more requests than its rows (`Service::select`), a row
    /// each.
    fn of<E>(
        passes: impl ExactSizeIterator<Item = (RequestId, Option<GuessKind>)>,
        rows: usize,
        fronts: impl FnOnce(&[RequestId], &[RequestId]) -> Result<FrontPass, E>,
    ) -> Result<Self, E> {
        let passes = passes.collect::<Vec<_>>();
        let requests = passes.iter().map(|&(id, _)| id).collect::<Vec<_>>();
        let mut fronted = passes
            .iter()
            .filter_map(|&(id, kind)| Some((kind? != GuessKind::Chained, id)))
            .collect::<Vec<_>>();
        // Chains of standing guesses continued first, then in request
        // order.
        fronted.sort_unstable();
        let fronted = fronted.into_iter().map(|(_, id)| id).collect::<Vec<_>>();
        let (form, depths) = match fronts(&requests, &fronted)? {
            FrontPass::Claim(claim) => return Ok(Self::Claim(claim)),
            FrontPass::Built { form, depths } => (form, depths),
        };
        // The budget past a row a request holds the grants.
        let mut left = rows
            .checked_sub(requests.len())
            .unwrap_or_else(|| panic!("invariant violated: a launch of {rows} rows reserved {} requests", requests.len()));
        let room = left;
        let mut granted = BTreeMap::new();
        for id in fronted {
            let Some(depth) = depths.get(&id).map(|&depth| depth.min(room)).filter(|&depth| depth > 0) else {
                continue;
            };
            if depth > left {
                break;
            }
            left -= depth;
            granted.insert(id, depth);
        }
        Ok(Self::Built { form, granted })
    }

    /// Request `id`'s grant of guessed rows, and the rows its round takes:
    /// its exact row and the guessed rows granted, of a pass the host
    /// builds in the launch's form (up to none, of none), or exactly its
    /// queued depth of the claim. Together they take no more than the rows
    /// the grants were held to ([`Self::of`]), so the depth priced is the
    /// depth built.
    fn grant(&self, id: RequestId) -> (FrontGrant, usize) {
        match self {
            Self::Built { form, granted } => {
                let depth = granted.get(&id).copied().unwrap_or(0);
                (form.grant(depth), 1 + depth)
            }
            Self::Claim(claim) => {
                let depth = *claim
                    .get(&id)
                    .unwrap_or_else(|| panic!("invariant violated: {id:?} builds a claim of the queued pass it is not in"));
                (FrontGrant::Exactly(depth), 1 + depth)
            }
        }
    }
}

/// Whether `generation` is resident, unfinished, and not yet past `position`.
fn prefilling_toward(generation: &Generation, position: usize) -> bool {
    generation.is_resident()
        && generation.finish_reason().is_none()
        && generation.resident_position() < position
}

/// Whether a request outside the pipeline will compute its prompt up to
/// `position`: it prefills toward it now, or it is admitted and unfinished
/// but not yet resident, and its first round prefills from where the cache
/// resumes it (`resumes`), below `position`. A request admitted while an
/// earlier one has not yet started otherwise prefills the shared prefix
/// beside it; one the cache resumes beyond `position` never computes it. A
/// request awaiting a prefix computes nothing until a peer caches it, so
/// waiters never wait on each other.
fn computes_toward(state: &State, position: usize, resumes: impl FnOnce(&Generation) -> usize) -> bool {
    match state {
        State::Between(generation, Stage::NonResident(_)) => {
            generation.finish_reason().is_none() && resumes(generation) < position
        }
        State::Between(_, Stage::AwaitingPrefix { .. }) => false,
        _ => prefilling_toward(state.generation(), position),
    }
}

/// Whether a finished request still owes the prefix cache its reconciliation
/// round: numerical state trails its accepted path.
fn owes_reconciliation(record: &Record, generation: &Generation) -> bool {
    record
        .prefix
        .as_ref()
        .is_some_and(|prefix| prefix.terminal_retention == TerminalRetention::Pending)
        && matches!(
            generation.finish_reason(),
            Some(FinishReason::Stop | FinishReason::Length | FinishReason::Context)
        )
        && generation.pending_reconciliation().is_some()
}

fn ended(generation: &Generation) -> bool {
    matches!(
        generation.finish_reason(),
        Some(FinishReason::Cancelled | FinishReason::Failed)
    )
}

/// Whether a finished request's terminal prefix retention is still owed,
/// completed, or abandoned after a non-fatal failure during post-finish
/// reconciliation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TerminalRetention {
    Pending,
    Retained,
    Abandoned,
}

/// Prefix-cache participation for a request admitted with
/// [`PrefixRetention::Retain`]. Absent when the request is transient, so
/// dead prefix-cache state cannot exist without participation.
struct PrefixParticipation {
    /// The cached prefix the request's residency forked, protected while
    /// work derived from it is submitted.
    source: Option<u64>,
    prefill_retained: bool,
    terminal_retention: TerminalRetention,
    /// Planned branch points, in prompt order: where this request diverges
    /// from a path the engine holds, and the cache points its admission
    /// declared. Prefill stops at each so the cache can retain a state every
    /// later request diverging there resumes from.
    branches: BTreeSet<usize>,
}

impl PrefixParticipation {
    fn new(branches: BTreeSet<usize>) -> Self {
        Self {
            source: None,
            prefill_retained: false,
            terminal_retention: TerminalRetention::Pending,
            branches,
        }
    }
}

/// The host's live output stream and its in-flight publication state.
/// Taking or dropping it atomically drops pending tokens and permits.
struct ActivePublication {
    sender: PublicationSender,
    batch_limit: usize,
    pending: Option<Vec<OutputToken>>,
    permits: VecDeque<PublicationPermit>,
}

impl ActivePublication {
    fn new(sender: PublicationSender, batch_limit: usize) -> Self {
        Self {
            sender,
            batch_limit,
            pending: None,
            permits: VecDeque::new(),
        }
    }

    fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    fn extend_permits(&mut self, permits: impl IntoIterator<Item = PublicationPermit>) {
        self.permits.extend(permits);
    }

    fn clear_pending(&mut self) {
        self.pending = None;
        self.permits.clear();
    }

    /// Drain accepted output into reserved permits. Returns whether the
    /// receiver has closed.
    fn drain(&mut self, generation: &mut Generation) -> bool {
        if !self.sender.receiver_open() {
            return true;
        }
        loop {
            let tokens = match self.pending.take() {
                Some(tokens) => tokens,
                None if generation.output_len() > 0 => match generation.take(self.batch_limit) {
                    Ok(tokens) => tokens,
                    Err(_) => break,
                },
                None => break,
            };
            let Some(permit) = self.permits.pop_front() else {
                // Output beyond the reserved bound waits for the next round's
                // reservation.
                self.pending = Some(tokens);
                break;
            };
            match self.sender.output_with_permit(tokens, permit) {
                Ok(()) => {}
                Err((PublishError::Full | PublishError::ExceedsCapacity, _)) => {
                    unreachable!("a publication permit owns an exact output slot")
                }
                Err((PublishError::ReceiverClosed, _)) => return true,
            }
        }
        if self.pending.is_none() && generation.output_len() == 0 {
            self.permits.clear();
        }
        false
    }
}

/// What stays with a request wherever its state is.
struct Record {
    waiting_since: u64,
    service_ns: u64,
    clock: RequestClock,
    error: Option<RequestError>,
    preemption_debt: u32,
    /// Prefix-cache participation; `None` when the request is transient.
    prefix: Option<PrefixParticipation>,
    /// The host's stream and its publication state; `None` once the host
    /// released the request or its terminal outcome was delivered.
    publication: Option<ActivePublication>,
    retirement: Retirement,
}

/// A request's wall-clock timing, kept as llama.cpp's server keeps a slot's:
/// its prompt from the first service it receives to its first accepted output
/// token, its generation from there to its latest accepted token. Everything in
/// between counts, whatever the engine spent it on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RequestClock {
    /// Admitted, never served.
    Waiting,
    /// Served since `started`, no output accepted yet.
    Prompt { started: Instant },
    /// Output accepted: the first token at `first`, the latest at `latest`.
    Generating { started: Instant, first: Instant, latest: Instant },
}

impl RequestClock {
    /// The request receives service at `now`: its prompt starts at its first.
    fn serve(&mut self, now: Instant) {
        if let Self::Waiting = self {
            *self = Self::Prompt { started: now };
        }
    }

    /// The request's generation accepted output at `now`.
    fn accepted(&mut self, now: Instant) {
        *self = match *self {
            Self::Waiting => panic!("output accepted for a request never served"),
            Self::Prompt { started } => Self::Generating { started, first: now, latest: now },
            Self::Generating { started, first, .. } => Self::Generating { started, first, latest: now },
        };
    }

    /// The timings of a request that completes at `now`.
    fn timings(self, now: Instant) -> RequestTimings {
        let nanos = |duration: Duration| u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        match self {
            Self::Waiting => RequestTimings::default(),
            Self::Prompt { started } => RequestTimings {
                prompt_ns: nanos(now - started),
                predicted_ns: 0,
            },
            Self::Generating { started, first, latest } => RequestTimings {
                prompt_ns: nanos(first - started),
                predicted_ns: nanos(latest - first),
            },
        }
    }
}

impl Record {
    /// Drop everything the host would still receive.
    fn release_publication(&mut self) {
        self.publication = None;
    }
}

/// Lowered operations a request submits before anything new. Never empty.
struct Work(Vec<Operation>);

impl Work {
    fn new(operations: Vec<Operation>) -> Option<Self> {
        (!operations.is_empty()).then_some(Self(operations))
    }
}

/// A residency or provisioning that waits on memory once the cache the
/// ledger took was freed: a request standing beside it holds what it
/// lacks, or the cache it took frees off this thread, or a real shortage
/// holds new work back (`MemoryResponse::admission_waits`). It retries at
/// the next change of ours past `at` (an eviction, a release, a request
/// ending, a reading that ends the shortage); one no change of ours can
/// let in fails at once with the typed capacity error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Wait {
    at: AvailabilityEpoch,
    on: WaitsOn,
}

/// What a wait is for: memory, or the end of another request's prompt
/// chain, whose queued chunks an open would orphan (a flight boundary,
/// which advances the epoch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WaitsOn {
    Memory,
    Chain,
}

enum Stage {
    /// Not resident; a live peer (`producer`) is computing the prefix through
    /// `through`, and this request will resume from it.
    AwaitingPrefix { producer: RequestId, through: usize },
    /// Not resident. Selection makes it resident.
    NonResident(Option<Wait>),
    /// Resident; selection starts a round. A finished request waits here for
    /// its output to drain, or for its reconciliation round.
    Ready,
    /// Resident; its host must drain output before the next round's
    /// publication permits fit.
    AwaitingCredit,
    /// Resident with work outside a target round to submit first: image
    /// encodes after residency, or method (drafter) operations.
    Pending(Work, Option<Wait>),
}

/// A request outside the pipeline.
enum State {
    /// No round started.
    Between(Generation, Stage),
    /// A started target round whose next operation couldn't be provisioned.
    /// It is submitted unchanged when selected.
    Parked(StartedRound, Work, Option<Wait>),
}

impl State {
    /// All composed work has completed. Only this state may be retained or
    /// retired successfully; a logical terminal decision can precede priming.
    fn settled(&self) -> Option<&Generation> {
        match self {
            Self::Between(
                generation,
                Stage::Ready
                | Stage::NonResident(_)
                | Stage::AwaitingCredit
                | Stage::AwaitingPrefix { .. },
            ) => Some(generation),
            _ => None,
        }
    }

    fn generation(&self) -> &Generation {
        match self {
            Self::Between(generation, _) => generation,
            Self::Parked(round, ..) => round.generation(),
        }
    }

    fn wait(&self) -> Option<Wait> {
        match self {
            Self::Between(_, Stage::NonResident(wait) | Stage::Pending(_, wait))
            | Self::Parked(_, _, wait) => *wait,
            Self::Between(_, _) => None,
        }
    }

    /// Terminate the request's generation, discarding any started round.
    fn terminate(self, cancel: bool) -> Generation {
        match self {
            Self::Between(generation, _) => terminated(generation, cancel),
            Self::Parked(round, ..) if cancel => round.cancel(),
            Self::Parked(round, ..) => round.fail(),
        }
    }
}

struct Request {
    record: Record,
    state: State,
}

/// A request whose work is in the current round.
enum Membership {
    /// Encodes or drafter work: no target round started.
    Generation(Generation),
    Started(StartedRound),
}

impl Membership {
    fn generation(&self) -> &Generation {
        match self {
            Self::Generation(generation) => generation,
            Self::Started(round) => round.generation(),
        }
    }

    fn terminate(self, cancel: bool) -> Generation {
        match self {
            Self::Generation(generation) => terminated(generation, cancel),
            Self::Started(round) if cancel => round.cancel(),
            Self::Started(round) => round.fail(),
        }
    }
}

/// A terminated request's finish: cancelled when withdrawn by its client, failed otherwise, so a
/// failure always publishes its classified error rather than an empty completion.
fn terminated(mut generation: Generation, cancel: bool) -> Generation {
    if cancel {
        generation.cancel();
    } else {
        generation.fail();
    }
    generation
}

struct Member {
    record: Record,
    state: Membership,
}

/// A request selection can take, moved out of the map while a round forms.
enum Candidate {
    NonResident(Generation),
    Ready(Generation),
    Pending(Generation, Work),
    Parked(StartedRound, Work),
}

impl Candidate {
    fn from_state(state: State) -> Self {
        match state {
            State::Between(generation, Stage::Ready) => Self::Ready(generation),
            State::Between(generation, Stage::NonResident(_)) => Self::NonResident(generation),
            State::Between(generation, Stage::Pending(work, _)) => Self::Pending(generation, work),
            State::Parked(round, work, _) => Self::Parked(round, work),
            State::Between(_, Stage::AwaitingPrefix { .. } | Stage::AwaitingCredit) => {
                unreachable!("unselectable state reached reservation")
            }
        }
    }

    /// The operations already composed for this candidate, which join a
    /// round as they are.
    fn composed(&self) -> Option<&[Operation]> {
        match self {
            Self::Pending(_, work) | Self::Parked(_, work) => Some(&work.0),
            Self::NonResident(_) | Self::Ready(_) => None,
        }
    }

    fn into_state(self) -> State {
        match self {
            Self::NonResident(generation) => State::Between(generation, Stage::NonResident(None)),
            Self::Ready(generation) => State::Between(generation, Stage::Ready),
            Self::Pending(generation, work) => State::Between(generation, Stage::Pending(work, None)),
            Self::Parked(round, work) => State::Parked(round, work, None),
        }
    }
}

/// How starting a selected request ended.
enum Start {
    /// Its work joins the round.
    Joined(Member, Vec<Operation>),
    /// It waits, or ended, outside the round.
    Returned(Request),
}

fn fail_generation(record: &mut Record, generation: &mut Generation, error: RequestError) {
    // A request that already reached a normal completion only ran a
    // post-finish reconciliation round for prefix retention. Failing that
    // round abandons terminal retention without turning the accepted
    // completion into a request failure.
    if matches!(
        generation.finish_reason(),
        Some(FinishReason::Stop | FinishReason::Length | FinishReason::Context)
    ) {
        if let Some(prefix) = &mut record.prefix {
            prefix.terminal_retention = TerminalRetention::Abandoned;
        }
        return;
    }
    // Accepted output still precedes the terminal failure on this
    // request's stream. Its reserved permits must survive until that
    // output has drained.
    if generation.finish_reason().is_none() {
        generation.fail();
    }
    record.error = Some(error);
}

/// `generation` failed to start its round on an engine invariant: it ends,
/// outside the round.
fn failed(mut record: Record, mut generation: Generation, detail: String) -> Request {
    fail_generation(&mut record, &mut generation, invariant("generation round start")(detail));
    Request {
        record,
        state: State::Between(generation, Stage::Ready),
    }
}

/// How reserving a selected request's start ended (`Service::reserve`).
enum Reservation {
    /// Work composed before the round: it joins as it is.
    Composed(Member, Vec<Operation>),
    /// A round, everything that could turn it back passed: it joins.
    Round(ReservedRound),
    /// It waits, or ended, outside the round.
    Returned(Request),
}

/// A request's next round reserved for its launch: room held for all a
/// round of its allowance publishes, so building it within no more rows
/// ([`Self::build`]) turns nothing back. A launch
/// prices only the rounds it reserved ([`FrontRows`]), which are the rounds
/// it builds.
struct ReservedRound {
    record: Record,
    generation: Generation,
    /// The most rows its round takes.
    allowance: usize,
}

impl ReservedRound {
    /// Reserve `generation`'s next round of at most `allowance` rows, with
    /// `selections` of the round's selection bound left, or turn it back:
    /// no row left it, its publication full (it awaits credit) or its
    /// receiver closed (it is cancelled). A round of fewer rows publishes no
    /// more, so the room held covers any round its launch builds.
    fn reserve(
        mut record: Record,
        generation: Generation,
        allowance: usize,
        selections: usize,
        epoch: &mut AvailabilityEpoch,
    ) -> Reservation {
        if owes_reconciliation(&record, &generation) {
            return Reservation::Round(Self { record, generation, allowance });
        }
        // A prefill chunk ends exactly at each planned branch point, and at
        // the prompt's retained boundary one row before its end.
        let resident = generation.resident_position();
        let allowance = next_stop(&record, &generation, resident)
            .map_or(allowance, |stop| allowance.min(stop - resident));
        // With the round's selection bound spent, a chunk that would reach
        // the end of the prompt (and select there) stops one row short and
        // finishes in a later round; with no row left before the end, the
        // request waits. Decode never gets here with the bound spent.
        let allowance = if selections == 0 {
            allowance.min(
                generation
                    .resume_bound()
                    .saturating_sub(1)
                    .saturating_sub(resident),
            )
        } else {
            allowance
        };
        if allowance == 0 {
            return Reservation::Returned(Request {
                record,
                state: State::Between(generation, Stage::Ready),
            });
        }
        let mut allowance = allowance;
        let mut bound = match generation.publication_bound(allowance) {
            Ok(bound) => bound,
            Err(detail) => return Reservation::Returned(failed(record, generation, detail)),
        };
        if bound != 0 {
            if let Some(publication) = &mut record.publication {
                let capacity = publication.sender.capacity();
                let max_bound = capacity.saturating_mul(publication.batch_limit);
                if bound > max_bound && max_bound > 0 {
                    allowance = allowance.min(max_bound);
                    bound = match generation.publication_bound(allowance) {
                        Ok(bound) => bound,
                        Err(detail) => {
                            return Reservation::Returned(failed(record, generation, detail));
                        }
                    };
                }
                if bound != 0 {
                    let slots = bound.div_ceil(publication.batch_limit);
                    match publication.sender.reserve_outputs(slots) {
                        Ok(permits) => publication.extend_permits(permits),
                        Err(PublishError::Full) => {
                            return Reservation::Returned(Request {
                                record,
                                state: State::Between(generation, Stage::AwaitingCredit),
                            });
                        }
                        Err(PublishError::ExceedsCapacity) => {
                            return Reservation::Returned(failed(
                                record,
                                generation,
                                format!(
                                    "round requires {slots} publication slots, exceeding queue capacity {capacity}"
                                ),
                            ));
                        }
                        Err(PublishError::ReceiverClosed) => {
                            let mut generation = generation;
                            generation.cancel();
                            generation.discard_output();
                            record.release_publication();
                            epoch.advance();
                            return Reservation::Returned(Request {
                                record,
                                state: State::Between(generation, Stage::Ready),
                            });
                        }
                    }
                }
            }
        }
        Reservation::Round(Self { record, generation, allowance })
    }

    /// The kind of the first guess its round runs if granted every guessed
    /// row it holds (`Generation::front_guess`), as the context allows.
    fn front_guess(&self) -> Option<GuessKind> {
        self.generation.front_guess()
    }

    fn into_request(self) -> Request {
        Request {
            record: self.record,
            state: State::Between(self.generation, Stage::Ready),
        }
    }

    /// Build request `id`'s round within `rows` of its rows, with the
    /// guessed rows its `grant` gives, in the form it gives;
    /// `publish_behind` takes whether the decode queued behind a target
    /// round publishes its feature. Nothing turns it back: only its
    /// generation failing an invariant leaves it out of the launch.
    fn build(self, id: RequestId, rows: usize, grant: FrontGrant, publish_behind: impl FnOnce(bool)) -> Start {
        let Self { record, generation, allowance } = self;
        let allowance = allowance.min(rows);
        if owes_reconciliation(&record, &generation) {
            return match generation.start_reconciliation(allowance) {
                Ok(round) => match lower_round(&round, id, round.generation().resident_position(), None) {
                    Ok(operation) => Start::Joined(
                        Member {
                            record,
                            state: Membership::Started(round),
                        },
                        vec![operation],
                    ),
                    Err(detail) => Start::Returned(failed(record, round.fail(), detail)),
                },
                Err((generation, detail)) => Start::Returned(failed(record, generation, detail)),
            };
        }
        match generation.start_round(id, allowance, grant) {
            Ok(RoundStart::Target(round)) => {
                // Where the next round prepares to draft, the decode queued
                // behind this one publishes its feature as that round.
                publish_behind(round.publishes_behind());
                match lower_round(&round, id, round.generation().resident_position(), None) {
                    Ok(operation) => Start::Joined(
                        Member {
                            record,
                            state: Membership::Started(round),
                        },
                        vec![operation],
                    ),
                    Err(detail) => Start::Returned(failed(record, round.fail(), detail)),
                }
            }
            Ok(RoundStart::Method(generation, operations)) => {
                if operations.iter().any(|operation| operation.request() != id) {
                    Start::Returned(failed(
                        record,
                        generation,
                        "method returned an operation for another request".into(),
                    ))
                } else {
                    Start::Joined(
                        Member {
                            record,
                            state: Membership::Generation(generation),
                        },
                        operations,
                    )
                }
            }
            Err((generation, detail)) => Start::Returned(failed(record, generation, detail)),
        }
    }
}

/// A round's launch as its requests join it: their members and work, the
/// rows that work takes, and the selection bound it leaves.
struct Launch {
    members: BTreeMap<RequestId, Member>,
    operations: Vec<Operation>,
    rows: usize,
    selections: usize,
}

impl Launch {
    fn new(selections: usize) -> Self {
        Self {
            members: BTreeMap::new(),
            operations: Vec::new(),
            rows: 0,
            selections,
        }
    }

    /// Request `id` started (`start`): its work joins the launch, or it goes
    /// back to `requests`. A member whose work passes the selection bound
    /// left joins all the same, so the owner terminates it with that error.
    fn join(
        &mut self,
        id: RequestId,
        start: Start,
        requests: &mut BTreeMap<RequestId, Request>,
    ) -> Result<(), RequestError> {
        let (member, mut work) = match start {
            Start::Joined(member, work) => (member, work),
            Start::Returned(request) => {
                requests.insert(id, request);
                return Ok(());
            }
        };
        self.rows += work
            .iter()
            .filter(|operation| matches!(operation, Operation::Forward { .. } | Operation::Head { .. }))
            .map(Operation::row_count)
            .sum::<usize>();
        let charge = selection_charge(&work);
        self.members.insert(id, member);
        self.operations.append(&mut work);
        self.selections = self.selections.checked_sub(charge).ok_or_else(|| {
            invariant("scheduler")("a started member exceeds the round's selection bound".into())
        })?;
        Ok(())
    }

    /// Build the rounds a launch that may pair reserved (`reserved`, in
    /// selection order), `fronts` its front rows: each at its exact row and
    /// the guessed rows granted it, or of a claim of the queued pass, at
    /// that pass's rows ([`FrontRows::grant`]). Every round joins, past a
    /// selection-bound violation too, the first of which it returns.
    fn build(
        &mut self,
        reserved: Vec<(RequestId, ReservedRound)>,
        fronts: &FrontRows,
        requests: &mut BTreeMap<RequestId, Request>,
        mut publish_behind: impl FnMut(RequestId, bool),
    ) -> Result<(), RequestError> {
        let mut joined = Ok(());
        for (id, round) in reserved {
            let (grant, rows) = fronts.grant(id);
            let start = round.build(id, rows, grant, |publishes| publish_behind(id, publishes));
            joined = joined.and(self.join(id, start, requests));
        }
        joined
    }
}

fn classify_domain_error(error: DomainError) -> RequestError {
    match error {
        DomainError::Capacity(error) => RequestError::Capacity(ServiceCapacityError {
            resource: CapacityResource::Execution(error.resource),
            required: error.required,
            available: error.available,
        }),
        DomainError::Input(error) => RequestError::Input(error),
        DomainError::State(error) => RequestError::State(error),
        DomainError::Device(error) | DomainError::Submit(SubmitError::Device(error)) => {
            RequestError::Device(error)
        }
        DomainError::Invariant(error) | DomainError::Submit(SubmitError::Invariant(error)) => {
            RequestError::Invariant(error)
        }
        DomainError::WeightRead(error) => RequestError::WeightRead(error),
        error @ DomainError::StateCeiling { .. } => RequestError::Invariant(InvariantError {
            context: "state ceiling",
            detail: error.to_string(),
        }),
    }
}

/// A request's refusal where nothing of ours is left to free for it: its
/// state short (as it opens, or grows ahead of a step), it alone needs
/// more than the load can ever give it
/// ([`RequestError::ContextExceedsMemory`]).
fn refused_alone(error: DomainError) -> RequestError {
    match error {
        DomainError::Capacity(magnitude_executor::CapacityError { resource: ResourceKind::RequestState, required, available, .. }) => {
            RequestError::ContextExceedsMemory { required, available }
        }
        error => classify_domain_error(error),
    }
}

/// Whether a domain error stops execution rather than one request.
fn is_fatal(error: &DomainError) -> bool {
    matches!(
        error,
        DomainError::Device(_)
            | DomainError::Invariant(_)
            | DomainError::State(_)
            | DomainError::Submit(SubmitError::Device(_) | SubmitError::Invariant(_))
    )
}

fn invariant(context: &'static str) -> impl FnOnce(String) -> RequestError {
    move |detail| RequestError::Invariant(InvariantError { context, detail })
}

/// An eviction that failed an engine invariant, as the domain's own
/// releases fail.
fn release_invariant(context: &'static str) -> impl FnOnce(String) -> DomainError {
    move |detail| DomainError::Invariant(InvariantError { context, detail })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    Runnable,
    AwaitingCompletion,
    OutputBlocked,
    Preempted,
    /// It never ran, and waits on memory: a reading that is not blind, a
    /// standing request's end, or the cache the ledger took to be freed.
    MemoryBlocked,
    Terminal(FinishReason),
}

/// A request refused before it acquires accepted numerical state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    ModelUnloaded {
        cause: ModelUnloadCause,
    },
    /// The request cannot be admitted: the typed failure it would otherwise
    /// have reported (invalid input or a failed engine invariant).
    Refused(RequestError),
}

impl AdmissionError {
    /// An engine invariant failed while admitting.
    fn invariant(context: &'static str) -> impl FnOnce(String) -> Self {
        move |detail| Self::Refused(RequestError::Invariant(InvariantError { context, detail }))
    }
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelUnloaded {
                cause: ModelUnloadCause::MemoryPressure,
            } => formatter.write_str("model unloaded due to memory pressure"),
            Self::Refused(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AdmissionError {}

impl From<DomainError> for AdmissionError {
    fn from(error: DomainError) -> Self {
        Self::Refused(classify_domain_error(error))
    }
}

/// What one [`Owner::run`] left behind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ran {
    /// Whether a group is on the device.
    pub in_flight: bool,
    /// Memory escalation decided to unload the model. Every request has been
    /// terminated with memory pressure; the caller drains the flight.
    pub unload: bool,
    /// The owner's next reading of memory that no arrival brings.
    pub reread: Reread,
}

/// How a running expert bank resize relates to the tier's previous resize:
/// a store's first resize follows the load's own reading (term `"load"`,
/// nothing moved before it, and no prior reading generation); each later
/// resize records the previous resize's reading generation, the terms that
/// moved between the two, and the one whose change set the target
/// (`SizingTerms::term`).
#[derive(Clone, Debug)]
enum ResizeOrigin {
    Load,
    Delta {
        before: Option<u64>,
        moved: Vec<SizingTerm>,
        term: Option<&'static str>,
    },
}

impl ResizeOrigin {
    fn from_previous(previous: Option<SizingTerms>, reading: &SizingTerms) -> Self {
        match previous {
            Some(before) => Self::Delta {
                before: before.reading,
                moved: reading.moved(&before),
                term: reading.term(&before),
            },
            None => Self::Load,
        }
    }

    fn before_generation(&self) -> Option<u64> {
        match self {
            Self::Load => None,
            Self::Delta { before, .. } => *before,
        }
    }

    fn moved(&self) -> &[SizingTerm] {
        match self {
            Self::Load => &[],
            Self::Delta { moved, .. } => moved,
        }
    }

    fn term(&self) -> Option<&'static str> {
        match self {
            Self::Load => Some("load"),
            Self::Delta { term, .. } => *term,
        }
    }
}

/// A running expert bank resize: what the process held in the tier's
/// store when it began (`ExecutorDomain::held_bytes_of`), and why it began:
/// the reading its store's target was derived from, and how it relates to the
/// store's previous resize ([`ResizeOrigin`]), which traces the resize to the
/// event that moved its sizing terms.
#[derive(Clone, Debug)]
struct BankBegun {
    held: u64,
    cause: BankBegunFor,
    reading: SizingTerms,
    origin: ResizeOrigin,
}

/// Why an expert bank resize began: a shrink to the bank's derived target,
/// or a growth into free memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BankBegunFor {
    Shrink,
    Growth,
}

impl BankBegunFor {
    /// The resize from `from` to `to` that finished, at `at_unix_ms`, as the
    /// census states it: the shrink or growth it began as, by at least one.
    fn finished(self, from: u64, to: u64, at_unix_ms: u64) -> Result<BankResize, String> {
        let by = |less: u64, more: u64| more.checked_sub(less).and_then(NonZeroU64::new);
        match self {
            Self::Shrink => by(to, from).map(|by| BankResize::Shrank { to, by, at_unix_ms }),
            Self::Growth => by(from, to).map(|by| BankResize::Grew { from, by, at_unix_ms }),
        }
        .ok_or_else(|| format!("a resize begun for {self:?} went from {from} to {to}"))
    }
}

struct ActiveGroup<F: ProgramFamily> {
    flight: DomainFlight<F>,
    operations: Vec<Operation>,
    prefix_uses: Vec<u64>,
}

struct Round {
    selection: Selection,
    started: u64,
    /// The requests whose work is in this round, moved out of the map.
    members: BTreeMap<RequestId, Member>,
    /// Groups not yet submitted, in order; reconciliation appends follow-ups.
    groups: VecDeque<OperationGroup>,
}

/// The owner's one pipeline: between flights, or one flight on the device,
/// a generation round's or a scoring flight, never both in one launch. A
/// flight holds the bindings until completion returns them, and its
/// outcomes reconcile only as the flight of its kind.
enum Pipeline<F: ProgramFamily> {
    /// Between flights: the right to change store bindings is here.
    Idle(StateBindings),
    /// One group of a generation round on the device.
    Round(Round, ActiveGroup<F>),
    /// One scoring flight on the device: a forward of each open walk.
    Scoring(ScoringFlight<F>),
}

/// A scoring flight ([`Service::scoring_group`]): its flight, always a
/// target launch, and its operations, one forward of each open walk in the
/// walks' order.
struct ScoringFlight<F: ProgramFamily> {
    flight: DomainFlight<F>,
    operations: Vec<Operation>,
}

/// How provisioning a group ended.
enum Provisioned {
    /// The group fits: submit it.
    Fits(OperationGroup),
    /// Capacity fits the leading operations: submit them first.
    Split(OperationGroup, OperationGroup),
    /// It waits on memory: its requests wait.
    Short(OperationGroup, Wait),
    /// Over budget once every release was tried and it holds one operation:
    /// its requests fail at once with the typed capacity error.
    Refused(OperationGroup, RequestError),
}

/// Everything the owner holds except its pipeline.
struct Service<F: ProgramFamily> {
    /// Boxed as it is made (`ExecutorDomain::with_family`).
    domain: Box<ExecutorDomain<F>>,
    scheduler: Scheduler,
    /// Requests outside the pipeline.
    requests: BTreeMap<RequestId, Request>,
    epoch: AvailabilityEpoch,
    now: u64,
    prefix_cache: PrefixCache<ResumeState>,
    /// What the owner holds of memory between flights ([`between_flights`]).
    memory: BetweenFlights,
    /// The model unloaded for memory since the run last asked
    /// ([`MemoryHost::unload`]).
    unloaded: bool,
    /// The timed reading the run's last step that read memory left
    /// ([`Self::ran`]).
    reread: Reread,
    wakes: Arc<dyn Wakes>,
    throughput: ThroughputAccumulator,
    cold_tier: Option<Rc<RefCell<ColdTier>>>,
    /// The load's score walks; none where it exports no logits to score
    /// with.
    scoring: Option<Walks>,
}

/// One bank tier's resize state: its running resize (if any) and the reading
/// its last resize began on, against which the next one's moved terms are read.
struct TierResize {
    /// The running expert bank resize this owner began: the stores resize on
    /// their own (`BankStep::Begun`).
    began: Option<BankBegun>,
    /// The reading this tier's last resize began on.
    sized: Option<SizingTerms>,
}

/// Whether a memory reading is owed at the next step and whether the worker is
/// waking from a [`Reread::Settled`] idle period.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadingState {
    /// Active: no reading owed, not waking from an idle period.
    Active,
    /// A reading is owed at the next step whatever brings it: the load's
    /// first, the worker's timed one ([`Owner::reread`]), or one a change
    /// beside it deferred (`MemoryReading::Beside`).
    Owed,
    /// The worker went idle with [`Reread::Settled`] (no flight and no timed
    /// reading running) at this `readings_generation`; no reading is owed yet.
    Settled(u64),
    /// Was [`Self::Settled`] at this `readings_generation` and now owes a
    /// fresh reading (e.g., after admission or a deferred wake reading): an
    /// unpressured wake whose stores were unread while asleep opens a fresh
    /// baseline.
    WakingFromIdle(u64),
}

impl ReadingState {
    fn is_owed(self) -> bool {
        matches!(self, Self::Owed | Self::WakingFromIdle(_))
    }

    /// Mark a reading owed at the next step, preserving any settled idle
    /// baseline generation.
    fn owe(&mut self) {
        *self = match *self {
            Self::Active | Self::Owed => Self::Owed,
            Self::Settled(generation) | Self::WakingFromIdle(generation) => {
                Self::WakingFromIdle(generation)
            }
        };
    }

    /// A new request was admitted: if the worker was idle-settled, it owes a
    /// fresh reading before selection/provisioning while keeping its idle
    /// baseline generation.
    fn on_admit(&mut self) {
        if let Self::Settled(generation) = *self {
            *self = Self::WakingFromIdle(generation);
        }
    }

    /// The worker went idle with [`Reread::Settled`] at `generation`.
    fn mark_settled(&mut self, generation: u64) {
        *self = match *self {
            Self::Active | Self::Settled(_) => Self::Settled(generation),
            Self::Owed | Self::WakingFromIdle(_) => Self::WakingFromIdle(generation),
        };
    }

    /// Begin a memory observation step: consume any owed flag, clear any idle
    /// baseline if `pressured`, and return `(owed, after_idle)`.
    fn begin_step(&mut self, pressured: bool, readings_generation: u64) -> (bool, bool) {
        let owed = self.is_owed();
        let after_idle = if pressured {
            *self = Self::Active;
            false
        } else {
            match *self {
                Self::Active | Self::Owed => {
                    *self = Self::Active;
                    false
                }
                Self::Settled(generation) | Self::WakingFromIdle(generation) => {
                    *self = Self::Settled(generation);
                    generation == readings_generation
                }
            }
        };
        (owed, after_idle)
    }

    /// A reading completed: clear any idle baseline.
    fn on_read(&mut self) {
        *self = Self::Active;
    }
}

/// What a loaded model's owner holds of its memory between flights
/// ([`between_flights`]): the expert bank's running resizes, the load's
/// newly stored measures, and whether new work waits on a real shortage.
pub struct BetweenFlights {
    /// Each store's expert bank resize state.
    tiers: [TierResize; BankTier::COUNT],
    /// Costs the executor stored since the load was priced, which price it
    /// from the next flight boundary on ([`Owner::measure_load`]).
    measures: Option<LoadMeasures>,
    /// The last reading's shortage holds the bank at its floor: no request
    /// becomes resident anew until a reading finds none (the ordered
    /// response's second step, `MemoryResponse::admission_waits`).
    admission_waits: bool,
    /// Whether a memory reading is owed and whether the worker is waking from
    /// an idle [`Reread::Settled`] period.
    reading: ReadingState,
}

impl BetweenFlights {
    /// A resize of `tier` storage began, as `begun`: the bank begins none of
    /// a store while one of it runs.
    fn bank_begun(&mut self, tier: BankTier, begun: BankBegun) -> Result<(), String> {
        match &mut self.tiers[tier.index()].began {
            Some(running) => Err(format!("a {tier:?} resize began beside one running since {:?}", running.cause)),
            slot => {
                *slot = Some(begun);
                Ok(())
            }
        }
    }

    /// A load's memory as it opens: nothing in motion, read at its first
    /// step.
    pub fn new() -> Self {
        Self {
            tiers: [const { TierResize { began: None, sized: None } }; BankTier::COUNT],
            measures: None,
            admission_waits: false,
            reading: ReadingState::Owed,
        }
    }
}

/// What [`between_flights`] runs on: the loaded model's domain and the
/// owner's memory, the service clock and its completion wakes, and the
/// model's unload (below the scheduler, an instrument has no requests): the
/// one owner of the between-flight memory step is that function, and a host
/// gives it only these.
pub trait MemoryHost {
    type Family: ProgramFamily;
    fn domain(&mut self) -> &mut ExecutorDomain<Self::Family>;
    fn memory(&mut self) -> &mut BetweenFlights;
    /// The service clock, in nanoseconds.
    fn now(&self) -> u64;
    fn completion(&self) -> magnitude_executor::CompletionWake;
    /// An availability change: what waits on one retries.
    fn advance_epoch(&mut self);
    /// Unload the model for memory: every request is terminated.
    fn unload(&mut self);
    /// Evict the retained prefixes worth least while the ledger owes their
    /// cache (`ExecutorDomain::prefix_owed`), settling the slabs after
    /// each: whether any was. A host without a prefix cache retains none.
    fn evict_owed_prefixes(&mut self) -> bool;
    /// The retained prefix worth least: the prefill tokens it saves and its
    /// hits since it was retained, which value the prefixes' cache
    /// (`ExecutorDomain::value_cache`); none without one.
    fn prefix_value(&self) -> Option<(u64, u64)>;
    /// The requests the host can evict now: the only ones a real
    /// shortage's relief names ([`Self::preempt`]).
    fn evictable(&self) -> BTreeSet<RequestId>;
    /// Evict `sequences`, the requests a real shortage's relief named, each
    /// of them evictable since the reading that named them.
    fn preempt(&mut self, sequences: &[RequestId]) -> Result<(), RequestError>;
}

pub struct Owner<F: ProgramFamily = NativeFamily> {
    service: Service<F>,
    pipeline: Pipeline<F>,
}

impl<F: ProgramFamily> Owner<F> {
    pub fn new(
        domain: Box<ExecutorDomain<F>>,
        bindings: StateBindings,
        limits: ServiceLimits,
        wakes: Arc<dyn Wakes>,
    ) -> Result<Self, String> {
        Self::with_prefix_cache_capacity(
            domain,
            bindings,
            limits,
            PrefixCacheCapacity::disabled(),
            wakes,
            None,
            None,
            0,
        )
    }

    pub fn with_prefix_cache_capacity(
        domain: Box<ExecutorDomain<F>>,
        bindings: StateBindings,
        limits: ServiceLimits,
        capacity: PrefixCacheCapacity,
        wakes: Arc<dyn Wakes>,
        cold_identity: Option<[u8; 32]>,
        state_cache: Option<PathBuf>,
        budget: u64,
    ) -> Result<Self, String> {
        // The OS's pressure events wake the owner, whose run reads memory.
        domain.watch_pressure(wakes.pressure());
        let scoring = match Shape::of(domain.limits()) {
            Some(shape) => Some(Walks::new(shape, Bounds::of(domain.definition())?)),
            None => None,
        };
        let identity = cold_identity;
        let cold_tier = match (identity, state_cache) {
            (Some(identity), Some(dir)) => {
                let hex = EntryKey(identity).hex();
                Some(Rc::new(RefCell::new(
                    ColdTier::open(dir.join(hex), budget).map_err(|e| e.to_string())?,
                )))
            }
            _ => None,
        };
        let mut service = Service {
            domain,
            scheduler: Scheduler::new(limits)?,
            requests: BTreeMap::new(),
            epoch: AvailabilityEpoch::default(),
            now: 0,
            prefix_cache: PrefixCache::new(capacity),
            memory: BetweenFlights::new(),
            unloaded: false,
            reread: Reread::Unread,
            wakes,
            throughput: ThroughputAccumulator::new(),
            cold_tier,
            scoring,
        };
        if let Some(tier) = service.cold_tier.clone() {
            let identity = identity.unwrap();
            service.prefix_cache.on_evict(move |info: &CachedPrefixKeyInfo, state: &ResumeState| {
                let method_str = match info.method {
                    MethodCheckpoint::Plain => "Plain",
                    MethodCheckpoint::Mtp(_) => "Mtp",
                    MethodCheckpoint::DFlash(_) => "DFlash",
                };
                let draft: Option<&DraftCheckpoint> = match info.method {
                    MethodCheckpoint::Mtp(d) | MethodCheckpoint::DFlash(d) => Some(d),
                    _ => None,
                };
                if !info.conditioning.is_empty() || draft.is_some_and(|d| !d.spillable()) {
                    return;
                }
                let started = std::time::Instant::now();
                let Ok(mut t) = tier.try_borrow_mut() else {
                    return;
                };
                if !t.can_spill() {
                    return;
                }
                let Ok(target) = state.target().export() else {
                    tracing::event!(tracing::Level::WARN, skip_unsupported = true, "cold-tier spill skipped");
                    return;
                };
                let Ok(head) = state.head().map(|h| h.export()).transpose() else {
                    tracing::event!(tracing::Level::WARN, skip_unsupported = true, "cold-tier spill skipped");
                    return;
                };
                let pending = draft.and_then(|d| {
                    let toks = d.pending_tokens().to_vec();
                    if toks.is_empty() {
                        None
                    } else {
                        d.pending_feature_bytes().map(|b| (toks, b.to_vec()))
                    }
                });
                let open = draft.and_then(|d| d.open_feature_bytes_spill(state.boundary()));
                let point = match info.point {
                    CachePoint::Branch => PointTag::Branch,
                    CachePoint::PromptBoundary => PointTag::PromptBoundary,
                    CachePoint::Terminal => PointTag::Terminal,
                };
                let key = entry_key(&identity, info.tokens, &[], method_str, info.position);
                let payload = encode_spill(
                    key,
                    info.tokens.to_vec(),
                    info.position,
                    SpilledCheckpoint {
                        method: match method_str {
                            "DFlash" => MethodTag::DFlash,
                            _ if draft.is_some() => MethodTag::Mtp,
                            _ => MethodTag::Plain,
                        },
                        point,
                        target,
                        head,
                        pending,
                        open,
                        skipped: draft.map_or(0, DraftCheckpoint::skipped),
                    },
                );
                let spill_bytes: usize = payload.regions.iter().map(|r| r.bytes.len()).sum();
                t.spill(payload, method_str);
                tracing::event!(tracing::Level::INFO, spill_bytes = spill_bytes as u64, spill_ms = started.elapsed().as_millis() as u64, "cold-tier spill");
            });
        }
        Ok(Self {
            service,
            pipeline: Pipeline::Idle(bindings),
        })
    }

    pub fn with_resource_plan(
        domain: Box<ExecutorDomain<F>>,
        bindings: StateBindings,
        limits: ServiceLimits,
        plan: &ResourcePlan,
        wakes: Arc<dyn Wakes>,
        cold_identity: Option<[u8; 32]>,
        state_cache: Option<PathBuf>,
        budget: u64,
    ) -> Result<Self, String> {
        Self::with_prefix_cache_capacity(
            domain,
            bindings,
            limits,
            PrefixCacheCapacity::from_resource_plan(plan),
            wakes,
            cold_identity,
            state_cache,
            budget,
        )
    }

    /// Logical admission: record the request and hand the host its stream.
    /// It opens no state and grows nothing; it becomes resident when a round
    /// selects it.
    pub fn admit(
        &mut self,
        generation: Generation,
        input: PreparedModelInput,
        retention: PrefixRetention,
        output_capacity: usize,
        now: u64,
    ) -> Result<(RequestId, PublicationReceiver), AdmissionError> {
        let members = match &mut self.pipeline {
            Pipeline::Round(round, _) => Some(&mut round.members),
            Pipeline::Idle(_) | Pipeline::Scoring(_) => None,
        };
        self.service
            .admit(generation, input, retention, output_capacity, now, members)
    }

    /// Queue a teacher-forced score of `request.tokens`; `request.done` is called exactly once: with the outcome,
    /// or with the request error that ends it (a load that exports no logits, a capacity refusal alone,
    /// termination, a domain failure). Its walk opens a sequence of the domain when a scoring flight takes it, in
    /// arrival order, with an identity of the owner's own that no request's status ever names.
    pub fn score(&mut self, request: ScoreRequest, now: u64) {
        match self.service.time(now) {
            Ok(()) => self.service.score(request),
            Err(error) => (request.done)(Err(error)),
        }
    }

    /// Stop a live request through its ordered stream. Work in flight still
    /// reconciles before the request retires.
    pub fn stop(&mut self, request: RequestId) {
        self.terminate_request(request, false);
    }

    /// Release a request the host abandoned: no terminal outcome reaches it.
    pub fn cancel(&mut self, request: RequestId) {
        self.terminate_request(request, true);
    }

    fn terminate_request(&mut self, id: RequestId, release: bool) {
        if let Pipeline::Round(round, _) = &mut self.pipeline {
            if let Some(Member { mut record, state }) = round.members.remove(&id) {
                // Work in flight reconciles as aborted; queued work is dropped.
                let mut generation = state.terminate(true);
                generation.discard_output();
                if release {
                    record.release_publication();
                } else if let Some(publication) = &mut record.publication {
                    publication.clear_pending();
                }
                round.members.insert(
                    id,
                    Member {
                        record,
                        state: Membership::Generation(generation),
                    },
                );
                return;
            }
        }
        self.service.terminate_request(id, release);
    }

    pub fn snapshot(&self, request: RequestId) -> Option<RequestSnapshot> {
        if let Pipeline::Round(round, _) = &self.pipeline {
            if let Some(member) = round.members.get(&request) {
                return Some(snapshot(
                    member.state.generation(),
                    match member.state.generation().finish_reason() {
                        Some(finish) if ended(member.state.generation()) => {
                            Status::Terminal(finish)
                        }
                        _ => Status::AwaitingCompletion,
                    },
                ));
            }
        }
        let request = self.service.requests.get(&request)?;
        Some(snapshot(
            request.state.generation(),
            self.service.status(request),
        ))
    }

    /// A fresh reading of every memory domain, then what each store's
    /// ledger holds allocated, by class.
    pub fn observe(&self) -> Result<WorkerReply, String> {
        Ok(WorkerReply::Observed {
            readings: self.service.domain.observe_memory(),
            stores: self.store_allocations(),
            serving: self.serving_census()?,
        })
    }

    /// A live accepted-output token snapshot over the owner's ten-second window.
    pub fn throughput(&self) -> ThroughputSnapshot {
        self.service
            .throughput
            .snapshot(Duration::from_nanos(self.service.now))
    }

    /// How the model serves now ([`ServingCensus`]).
    pub fn serving_census(&self) -> Result<ServingCensus, String> {
        let service = &self.service;
        let experts = service.domain.expert_residency().map_err(|error| error.to_string())?;
        Ok(ServingCensus {
            group_places: service.domain.group_places(),
            group_capacity: service.domain.chain_capacity(),
            experts,
        })
    }

    /// What each store's ledger holds allocated, by class.
    pub fn store_allocations(&self) -> Vec<StoreAllocation> {
        self.service.domain.store_allocations()
    }

    /// The state stores' bytes, every holder's (cached prefixes included)
    /// read from the stores' own records.
    pub fn state_bytes(&self) -> StateBytes {
        self.service.domain.state_bytes()
    }

    pub fn completed_service_ns(&self) -> u128 {
        self.service.scheduler.completed_service_ns()
    }

    pub fn cached_prefixes(&self) -> usize {
        self.service.prefix_cache.len()
    }

    /// Claim `bytes` of memory beside the model for the service's
    /// measurement job, in place of any claim before, which is released
    /// (`ExecutorDomain::claim_measurement_room`): a committed claim on the
    /// device's ledger, which cache yields to at once, so it is answered
    /// now, refused only where the model's committed memory leaves it short.
    /// A measurement waits for no request's end: short is short.
    pub fn claim_measurement_room(&mut self, bytes: u64) -> RoomGrant {
        let claimed = self.service.domain.claim_measurement_room(bytes);
        RoomGrant::decided(claimed.map_err(|short| RoomRefusal::Short { bytes: short.over }))
    }

    /// The measurement job holding the room was started as `process`
    /// (`ExecutorDomain::attach_measurement_process`).
    pub fn attach_measurement_process(&mut self, process: magnitude_executor::ProcessId) {
        self.service.domain.attach_measurement_process(process);
    }

    /// The measurement job ended: its claim is released.
    pub fn release_measurement_room(&mut self) {
        self.service.domain.release_measurement_room();
    }

    /// Price the load by `measures` from the next flight boundary on
    /// ([`between_flights`]), a later call's replacing an earlier one's not
    /// yet applied: no step in flight is recorded or priced by measures it
    /// did not run under.
    pub fn measure_load(&mut self, measures: LoadMeasures) {
        self.service.memory.measures = Some(measures);
    }

    /// The worker's timed reading of memory is due ([`Reread`]): the next
    /// run reads memory.
    pub fn reread(&mut self) {
        self.service.memory.reading.owe();
    }

    /// What the census counts moves only with: the generation of what the
    /// load's ledgers hold (`ExecutorDomain::holdings_generation`), and the
    /// expert bank's cover (`ExecutorDomain::expert_cover`), which moves
    /// through decode with no holding moving. Equal across two runs,
    /// nothing the census counts changed.
    pub fn census_key(&self) -> (u64, Option<magnitude_executor::ExpertCover>) {
        (self.service.domain.holdings_generation(), self.service.domain.expert_cover())
    }

    pub fn inspect_domain<R>(
        &self,
        inspect: impl FnOnce(&ExecutorDomain<F>) -> Result<R, String>,
    ) -> Result<R, String> {
        inspect(&self.service.domain)
    }

    /// Consume a reserved wake for this request's exact queue. Output credit
    /// lets a request waiting on it start its next round; receiver closure
    /// releases the request.
    pub fn publication_wake(&mut self, request: RequestId, wake: PublicationWake) {
        let cancelled = match wake.kind() {
            PublicationWakeKind::OutputCredit => {
                self.service.output_credit(request, wake, &mut self.pipeline);
                false
            }
            PublicationWakeKind::Cancelled => self
                .service
                .publication_sender(request, &mut self.pipeline)
                .is_some_and(|sender| sender.process_cancelled(wake)),
        };
        if cancelled {
            self.cancel(request);
        }
    }

    /// Terminate every live request with `error`, and answer every score
    /// with it: the lifecycle is draining.
    pub fn terminate_all(&mut self, error: RequestError) {
        let members = match &mut self.pipeline {
            Pipeline::Round(round, _) => Some(&mut round.members),
            Pipeline::Idle(_) | Pipeline::Scoring(_) => None,
        };
        self.service.terminalize_all(error, members);
    }

    /// Write the conversation's newest checkpoints to the cold tier as the
    /// owner ends: they are pinned in memory, so eviction never spills them.
    /// Returns once they are on disk ([`ColdTier::finish`]): the owner's stop
    /// is announced next, and its process may end before it drops.
    pub fn spill_on_unload(&mut self) {
        // ponytail: three entries (terminal, prompt boundary, branch) is what
        // `conversation_checkpoints` pins; raise if release deadline allows more.
        const UNLOAD_SPILL_LIMIT: usize = 3;
        self.service.prefix_cache.evict_all(UNLOAD_SPILL_LIMIT);
        if let Some(Ok(mut tier)) = self.service.cold_tier.as_ref().map(|tier| tier.try_borrow_mut()) {
            tier.finish();
        }
    }

    /// Run every transition that is possible now. An error is fatal: every
    /// request has been terminated with it and execution must stop.
    pub fn run(self, now: u64) -> Result<(Self, Ran), RequestError> {
        let Owner {
            mut service,
            pipeline,
        } = self;
        match service.drive(pipeline, now) {
            Ok((pipeline, ran)) => {
                if let Pipeline::Idle(_) = &pipeline {
                    service.check_no_unwakeable_wait();
                }
                Ok((Owner { service, pipeline }, ran))
            }
            Err((mut members, error)) => {
                // Members are dropped with the owner; terminating them first
                // delivers their outcome.
                service.terminalize_all(error.clone(), members.as_mut());
                Err(error)
            }
        }
    }
}

fn snapshot(generation: &Generation, status: Status) -> RequestSnapshot {
    let usage = generation.detailed_usage();
    RequestSnapshot {
        status,
        prompt_tokens: usage.prompt_tokens,
        cached_tokens: usage.cached_tokens,
        resident_position: generation.resident_position(),
        output_tokens: generation.output_len(),
    }
}

type Fatal = (Option<BTreeMap<RequestId, Member>>, RequestError);

impl<F: ProgramFamily> Service<F> {
    fn time(&mut self, now: u64) -> Result<(), RequestError> {
        if now < self.now {
            return Err(invariant("service clock")("service clock moved backwards".into()));
        }
        self.now = now;
        Ok(())
    }

    fn admit(
        &mut self,
        mut generation: Generation,
        input: PreparedModelInput,
        retention: PrefixRetention,
        output_capacity: usize,
        now: u64,
        members: Option<&mut BTreeMap<RequestId, Member>>,
    ) -> Result<(RequestId, PublicationReceiver), AdmissionError> {
        self.time(now).map_err(AdmissionError::Refused)?;
        self.domain.served_request(!input.vision().is_empty());
        if generation.is_resident() || generation.usage().completion_tokens != 0 {
            return Err(AdmissionError::invariant("admission")(
                "only a fresh generation can be admitted".into(),
            ));
        }
        let (prefix_cache, cache_points) = match retention {
            PrefixRetention::Transient => (false, Vec::new()),
            PrefixRetention::Retain { cache_points } => (true, cache_points),
        };
        generation
            .set_prefix_reuse(prefix_cache)
            .map_err(AdmissionError::invariant("admission"))?;
        if !input.tokens().is_empty()
            && (input.tokens() != generation.prompt()
                || input.layout() != generation.layout()
                || input.prompt_coordinates() != *generation.coordinates())
        {
            return Err(AdmissionError::invariant("admission")(
                "prepared input does not match generation prompt, layout, or coordinates".into(),
            ));
        }
        let path = PrefixPath::of(&generation);
        if cache_points
            .iter()
            .any(|&point| point >= generation.prompt().len() || !path.exact_boundary(point))
        {
            return Err(AdmissionError::invariant("admission")(
                "a cache point must be an exact boundary inside the prompt".into(),
            ));
        }
        let id = RequestId::admitted();
        let wakes = self.wakes.clone();
        let (sender, receiver) = PublicationQueue::bounded(output_capacity, move |wake| {
            wakes.publication(id, wake);
        })
        .map_err(|error| AdmissionError::invariant("publication")(error.to_owned()))?;
        self.domain
            .install_input(id, input)
            .map_err(|error| AdmissionError::Refused(RequestError::Input(error)))?;
        let (prefix, stage) = if prefix_cache {
            let plan = self.plan_branches(path, generation.resume_bound(), &cache_points, members);
            let stage = match plan.awaited {
                Some((producer, through)) => Stage::AwaitingPrefix { producer, through },
                None => Stage::NonResident(None),
            };
            (Some(PrefixParticipation::new(plan.branches)), stage)
        } else {
            (None, Stage::NonResident(None))
        };
        self.memory.reading.on_admit();
        self.requests.insert(
            id,
            Request {
                record: Record {
                    waiting_since: now,
                    service_ns: 0,
                    clock: RequestClock::Waiting,
                    error: None,
                    preemption_debt: 0,
                    prefix,
                    publication: Some(ActivePublication::new(sender, output_capacity)),
                    retirement: receiver.retirement(),
                },
                state: State::Between(generation, stage),
            },
        );
        Ok((id, receiver))
    }

    /// Queue `request`'s walk, or refuse it at once: a load that exports no
    /// logits scores none, and a score outside the load's bounds is invalid.
    fn score(&mut self, request: ScoreRequest) {
        let Some(walks) = &mut self.scoring else {
            return (request.done)(Err(RequestError::Input("this load exports no logits to score with".into())));
        };
        if let Some(refusal) = walks.bounds.refusal(&request.tokens) {
            return (request.done)(Err(refusal));
        }
        walks.waiting.push_back(request);
        // An idle owner reads memory afresh before new work opens.
        self.memory.reading.on_admit();
    }

    /// Branch points for a new request's `path`: each declared cache point
    /// no entry holds yet, and the deepest position where it diverges from a
    /// cached path or a live request's prompt. A live request that has not
    /// yet prefilled to its divergence from this one stops there too, so the
    /// shared prefix is cached by whichever request reaches it first and
    /// every later request resumes at it instead of recomputing.
    fn plan_branches(
        &mut self,
        path: PrefixPath,
        bound: usize,
        cache_points: &[usize],
        members: Option<&mut BTreeMap<RequestId, Member>>,
    ) -> BranchPlan {
        let resumed = self.prefix_cache.deepest(path, bound);
        let mut branches = BTreeSet::new();
        for &point in cache_points {
            if !self.prefix_cache.holds(path, point) {
                plan_declared(&mut branches, point, resumed, path.len());
            }
        }
        let mut deepest = self.prefix_cache.shared_boundary(path);
        // The deepest shared prefix a live peer is still prefilling toward
        // and will cache at its branch point, with that peer's identity.
        let mut in_flight: Option<(RequestId, usize)> = None;
        let live = self
            .requests
            .iter_mut()
            .filter_map(|(&id, request)| {
                let prefix = request.record.prefix.as_mut()?;
                Some((id, prefix, request.state.generation(), Some(&request.state)))
            })
            .chain(
                members
                    .into_iter()
                    .flat_map(|members| members.iter_mut())
                    .filter_map(|(&id, member)| {
                        let prefix = member.record.prefix.as_mut()?;
                        Some((id, prefix, member.state.generation(), None))
                    }),
            );
        for (peer_id, prefix, generation, outside) in live {
            let shared = path.shared_with_prompt_of(PrefixPath::of(generation));
            deepest = deepest.max(shared);
            plan_branch(
                &mut prefix.branches,
                shared,
                generation.resident_position(),
                generation.prompt().len(),
            );
            if let Some(stop) = peer_stop(&prefix.branches, shared) {
                // A round member is resident; a request outside the
                // pipeline may not be yet.
                let computes = match outside {
                    Some(state) => computes_toward(state, stop, |peer| {
                        self.prefix_cache
                            .deepest(PrefixPath::of(peer), peer.resume_bound())
                    }),
                    None => prefilling_toward(generation, stop),
                };
                if computes && in_flight.is_none_or(|(_, best)| stop > best) {
                    in_flight = Some((peer_id, stop));
                }
            }
        }
        if !worthwhile(deepest, resumed, path.len()) {
            return BranchPlan {
                branches,
                awaited: None,
            };
        }
        if let Some((producer, through)) = in_flight.and_then(|(producer, stop)| {
            awaited_prefix(stop, deepest, resumed, path.len()).map(|through| (producer, through))
        }) {
            // A peer computes this prefix now: wait for it to be cached.
            return BranchPlan {
                branches,
                awaited: Some((producer, through)),
            };
        }
        plan_branch(&mut branches, deepest, resumed, path.len());
        BranchPlan {
            branches,
            awaited: None,
        }
    }

    /// The cached entries live requests hold, among the service's requests
    /// and the in-flight round's `members`: the entry each resumed from and
    /// the deepest within its resident rows (see [`PrefixCache::held`]).
    /// Each costs one bank while its holder lives: the cache releases it
    /// last.
    fn held_entries(&self, members: Option<&BTreeMap<RequestId, Member>>) -> BTreeSet<u64> {
        self.requests
            .values()
            .map(|request| (&request.record, request.state.generation()))
            .chain(
                members
                    .into_iter()
                    .flat_map(BTreeMap::values)
                    .map(|member| (&member.record, member.state.generation())),
            )
            .filter(|(_, generation)| generation.is_resident() && generation.finish_reason().is_none())
            .flat_map(|(record, generation)| {
                self.prefix_cache.held(
                    record.prefix.as_ref().and_then(|prefix| prefix.source),
                    PrefixPath::of(generation),
                    generation.resident_position(),
                )
            })
            .collect()
    }

    /// Whether `peer` is a live, prefix-compatible producer still computing
    /// toward a planned branch point at `through` for `waiter`.
    fn is_prefix_producer(&self, waiter: &Generation, peer: &Request, through: usize) -> bool {
        peer.record
            .prefix
            .as_ref()
            .is_some_and(|prefix| prefix.branches.contains(&through))
            && PrefixPath::of(waiter)
                .shared_with_prompt_of(PrefixPath::of(peer.state.generation()))
                >= through
            && computes_toward(&peer.state, through, |generation| {
                self.prefix_cache
                    .deepest(PrefixPath::of(generation), generation.resume_bound())
            })
    }

    fn status(&self, request: &Request) -> Status {
        let generation = request.state.generation();
        if let Some(finish) = generation.finish_reason() {
            if !owes_reconciliation(&request.record, generation) {
                return Status::Terminal(finish);
            }
        }
        match request.state.wait() {
            Some(_) if self.waits_on_memory(request) => return Status::MemoryBlocked,
            // Behind another request's chain it waits on no memory: it runs
            // once the chain ends.
            Some(Wait { on: WaitsOn::Chain, .. }) => {}
            Some(_) => return Status::Preempted,
            None => {}
        }
        match &request.state {
            State::Between(_, Stage::AwaitingCredit) => Status::OutputBlocked,
            State::Between(_, Stage::NonResident(_)) if request.record.preemption_debt > 0 => {
                Status::Preempted
            }
            _ => Status::Runnable,
        }
    }

    fn publication_sender<'a>(
        &'a mut self,
        request: RequestId,
        pipeline: &'a mut Pipeline<F>,
    ) -> Option<&'a mut PublicationSender> {
        if let Pipeline::Round(round, _) = pipeline {
            if let Some(member) = round.members.get_mut(&request) {
                return member.record.publication.as_mut().map(|publication| &mut publication.sender);
            }
        }
        self.requests
            .get_mut(&request)
            .and_then(|request| request.record.publication.as_mut().map(|publication| &mut publication.sender))
    }

    /// A credit wake sends a request waiting on credit back to selection,
    /// whatever the ring holds now: its next reservation either fits or,
    /// refused, re-arms the queue's next wake.
    fn output_credit(&mut self, id: RequestId, wake: PublicationWake, pipeline: &mut Pipeline<F>) {
        if let Some(sender) = self.publication_sender(id, pipeline) {
            sender.process_output_credit(wake);
        }
        if let Some(request) = self.requests.get_mut(&id) {
            if let State::Between(_, stage @ Stage::AwaitingCredit) = &mut request.state {
                *stage = Stage::Ready;
            }
        }
    }

    /// Terminate a request outside the pipeline, and retire it once nothing
    /// remains for its host.
    fn terminate_request(&mut self, id: RequestId, release: bool) {
        let Some(Request { mut record, state }) = self.requests.remove(&id) else {
            return;
        };
        let mut generation = state.terminate(true);
        generation.discard_output();
        if release {
            record.release_publication();
        } else if let Some(publication) = &mut record.publication {
            publication.clear_pending();
        }
        self.requests.insert(
            id,
            Request {
                record,
                state: State::Between(generation, Stage::Ready),
            },
        );
        self.epoch.advance();
    }

    /// Terminate every live request with `error`, delivering each host its
    /// outcome now, and answer every open and waiting score with it: the one
    /// end of all work, whether the lifecycle drains, the model unloads or
    /// execution stops.
    fn terminalize_all(
        &mut self,
        error: RequestError,
        members: Option<&mut BTreeMap<RequestId, Member>>,
    ) {
        if let Some(walks) = &mut self.scoring {
            walks.fail_all(&error);
        }
        let terminate = |record: &mut Record, generation: &mut Generation| {
            if generation.finish_reason().is_none() {
                generation.fail();
                record.error.get_or_insert_with(|| error.clone());
            }
            // Only output already accepted by the queue precedes a terminal
            // outcome. Unpublished generation output is not promoted by a
            // second, unbounded failure delivery channel.
            if let Some(publication) = record.publication.take() {
                generation.discard_output();
                match generation.finish_reason() {
                    Some(FinishReason::Failed) | None => {
                        publication.sender.fail(record.error.clone().unwrap_or_else(|| error.clone()));
                    }
                    Some(finish) => publication.sender.complete(
                        finish,
                        generation.detailed_usage(),
                        generation.method_identity().to_owned(),
                        record.clock.timings(Instant::now()),
                    ),
                }
            }
        };
        let ids = self.requests.keys().copied().collect::<Vec<_>>();
        for id in ids {
            let Request { mut record, state } = self.requests.remove(&id).expect("listed request");
            let mut generation = match state {
                State::Between(generation, _) => generation,
                State::Parked(round, ..) => round.fail(),
            };
            terminate(&mut record, &mut generation);
            self.requests.insert(
                id,
                Request {
                    record,
                    state: State::Between(generation, Stage::Ready),
                },
            );
        }
        if let Some(members) = members {
            let ids = members.keys().copied().collect::<Vec<_>>();
            for id in ids {
                let Member { mut record, state } = members.remove(&id).expect("listed member");
                let mut generation = match state {
                    Membership::Generation(generation) => generation,
                    Membership::Started(round) => round.fail(),
                };
                terminate(&mut record, &mut generation);
                members.insert(
                    id,
                    Member {
                        record,
                        state: Membership::Generation(generation),
                    },
                );
            }
        }
    }

    fn drive(&mut self, mut pipeline: Pipeline<F>, now: u64) -> Result<(Pipeline<F>, Ran), Fatal> {
        self.time(now).map_err(|error| (None, error))?;
        let mut unload = false;
        // A flight submitted by this run is reconciled by a later one, even
        // when the device completed it during submission (CPU): its
        // completion event returns the worker here only after it has applied
        // the controls and events that arrived during the flight.
        let mut submitted = false;
        loop {
            if let Some(tier) = &self.cold_tier {
                if let Ok(mut guard) = tier.try_borrow_mut() {
                    guard.drain_results();
                }
            }
            pipeline = match pipeline {
                Pipeline::Round(round, mut active) => {
                    if submitted || !active.flight.completion().is_complete() {
                        // The round's members end with a fatal error, not
                        // dropped with the round.
                        if let Err(error) = self.step(None, false) {
                            return Err((Some(round.members), error));
                        }
                        unload |= std::mem::take(&mut self.unloaded);
                        return Ok((Pipeline::Round(round, active), self.ran(true, unload)));
                    }
                    // Reconciling reads memory at the flight's boundary and
                    // submits the round's next flight.
                    submitted = true;
                    self.reconcile(round, active)?
                }
                Pipeline::Scoring(mut scoring) => {
                    if submitted || !scoring.flight.completion().is_complete() {
                        self.step(None, false).map_err(|error| (None, error))?;
                        unload |= std::mem::take(&mut self.unloaded);
                        return Ok((Pipeline::Scoring(scoring), self.ran(true, unload)));
                    }
                    // Reconciling reads memory at the flight's boundary and
                    // submits nothing: the idle pipeline takes the next.
                    self.reconcile_scoring(scoring).map_err(|error| (None, error))?
                }
                Pipeline::Idle(mut bindings) => {
                    self.step(Some(&mut bindings), false).map_err(|error| (None, error))?;
                    // No flight holds the sequences of walks termination
                    // answered: they close now.
                    self.close_unclosed().map_err(|error| (None, error))?;
                    unload |= std::mem::take(&mut self.unloaded);
                    // Unloading terminated every request: nothing is left to
                    // select, and observing again would decide it again.
                    if unload {
                        self.publish_ready().map_err(|error| (None, error))?;
                        return Ok((Pipeline::Idle(bindings), self.ran(false, true)));
                    }
                    self.publish_ready().map_err(|error| (None, error))?;
                    match self.next_flight(&mut bindings)? {
                        Next::Scoring(group_) => {
                            submitted = true;
                            self.submit_scoring(group_, bindings).map_err(|error| (None, error))?
                        }
                        Next::Generation(Selected::Round(round)) => {
                            submitted = true;
                            self.advance(round, bindings)?
                        }
                        Next::Generation(Selected::Retry) => Pipeline::Idle(bindings),
                        Next::Generation(Selected::Empty) => {
                            // Waiting for work is no launch's service.
                            self.domain.host_idle();
                            let ran = self.ran(false, unload);
                            if ran.reread == Reread::Settled {
                                self.memory.reading.mark_settled(self.domain.readings_generation());
                            }
                            return Ok((Pipeline::Idle(bindings), ran));
                        }
                    }
                }
            };
        }
    }

    /// The memory step at an event ([`MemoryStep::observe`]), `boundary`
    /// where it is a flight's: the timed reading it leaves kept for the
    /// run's end ([`Self::ran`]); an unload is the host's
    /// ([`MemoryHost::unload`]).
    fn step(&mut self, bindings: Option<&mut StateBindings>, boundary: bool) -> Result<(), RequestError> {
        let observed = self.observe(bindings, boundary)?;
        if observed.reread != Reread::Unread {
            self.reread = observed.reread;
        }
        Ok(())
    }

    /// What the run came to. A flight in progress, or an unload, owes no
    /// timed reading (the flight's boundary reads next); otherwise the
    /// last step that read memory this run sets it, and a run that read
    /// none leaves it as it was.
    fn ran(&mut self, in_flight: bool, unload: bool) -> Ran {
        let reread = std::mem::replace(&mut self.reread, Reread::Unread);
        let reread = match in_flight || unload {
            true => Reread::Settled,
            false => reread,
        };
        Ran { in_flight, unload, reread }
    }
}

/// What one memory step came to: whether the model unloaded, and the timed
/// reading it leaves ([`Reread`]).
struct Observed {
    unload: bool,
    reread: Reread,
}

/// The owner's memory between flights, on what a host gives it
/// ([`MemoryHost`]): every host takes these, and none has its own.
trait MemoryStep: MemoryHost + Sized {
    /// The memory step at an event: `boundary` where it is a flight's
    /// boundary, `bindings` there where no flight holds them. Returns
    /// whether the model unloaded, after terminating every request, and
    /// the timed reading the step leaves. This is the owner's own work,
    /// which serves no request: no launch is charged the time it takes
    /// (`ExecutorDomain::owner_worked`).
    fn observe(&mut self, bindings: Option<&mut StateBindings>, boundary: bool) -> Result<Observed, RequestError> {
        let began = std::time::Instant::now();
        let observed = self.observe_memory(bindings, boundary);
        self.domain().owner_worked(began.elapsed());
        observed
    }

    /// The OS is read only at events (design §4.8), never at every run of
    /// the owner: a flight's boundary, the completion of a change of ours
    /// (a measured load priced, a bank resize or a slot release finished),
    /// an OS pressure event, and a reading owed (the load's first, the
    /// worker's timed one, one a change beside deferred). The reading
    /// (`ExecutorDomain::read_memory`) never reads our own change as another
    /// program's, and a real shortage is answered once per reading, in its
    /// order (spec §4.5), each step only where the one before did not meet
    /// it:
    /// 1. cache yields in rank order at once (taken by the reading); each
    ///    holder frees what was taken of it here, and the bank follows its
    ///    target, derived from the ledgers afresh;
    /// 2. new work waits while the bank sits at its floor;
    /// 3. the named sequences are preempted, the lowest value first;
    /// 4. the model unloads, only on a shortage that holds across readings
    ///    spanning the store's lag with no change of ours pending.
    ///
    /// Where no flight holds the bindings, what claims took of the cache is
    /// freed and the bank follows its target at every step: neither reads
    /// the OS.
    fn observe_memory(&mut self, mut bindings: Option<&mut StateBindings>, boundary: bool) -> Result<Observed, RequestError> {
        // A cut of the expert bank a chain's encoding began beside its
        // running steps (`ExecutorDomain::take_running_release`) is a shrink
        // this owner holds as begun: the advance that finishes it reports
        // it as any.
        if let Some(tier) = self.domain().take_running_release() {
            let held = self.domain().held_bytes_of(tier);
            let reading = self.domain().sizing_terms(tier);
            self.begin_resize(tier, BankBegunFor::Shrink, held, reading)?;
        }
        // Own changes that complete here: newly stored costs price the load
        // first, so every decision past it is made on them; the expert
        // bank's and the graph slots' releases advance. The cache and each
        // sequence's state are valued afresh.
        let mut completed = false;
        if let Some(bindings) = bindings.as_deref_mut() {
            // A refusal (the device would not take the tier copy's grid) is a device
            // fault the load ends on, the experts still priced as before.
            if let Some(measures) = self.memory().measures.take() {
                self.domain().measure_load(measures, bindings).map_err(classify_domain_error)?;
                completed = true;
            }
            completed |= self.advance_expert_bank()?;
            completed |= self.advance_slot_release();
            let prefix = self.prefix_value();
            self.domain().value_cache(prefix).map_err(classify_domain_error)?;
            self.domain().value_state().map_err(classify_domain_error)?;
        }
        let pressured = self.domain().take_pressure();
        let readings_generation = self.domain().readings_generation();
        let (owed, after_idle) = self.memory().reading.begin_step(pressured, readings_generation);
        let read = match boundary || completed || owed || pressured {
            false => None,
            true => match {
                let evictable = self.evictable();
                self.domain().read_memory(after_idle, &evictable).map_err(classify_domain_error)?
            } {
                MemoryReading::Read(read) => {
                    self.memory().reading.on_read();
                    Some(read)
                }
                // A change off this thread stood beside the sample: its
                // completion wakes the owner, whose step reads then.
                MemoryReading::Beside => {
                    self.memory().reading.owe();
                    None
                }
            },
        };
        if let Some(bindings) = bindings {
            let sizing = [BankTier::Bank, BankTier::Host].map(|tier| self.domain().sizing_terms(tier));
            self.follow_yields(bindings)?;
            // `follow_cache_targets` runs only here (in `between_flights`) for
            // external `HostRam` pressure and bank growth targets — never to
            // make room for claims, which reclaim synchronously inside
            // `Ledger::admit_or_reclaim`. Both paths reach the same bank shrink
            // release (`ExpertBank::release_evicted`).
            self.follow_cache_targets(bindings, sizing)?;
        }
        let Some(read) = read else {
            return Ok(Observed { unload: false, reread: Reread::Unread });
        };
        let response = read.response.unwrap_or_default();
        if response.unload {
            self.unload();
            return Ok(Observed { unload: true, reread: Reread::Settled });
        }
        // Each store's relief plan already shed its cache in `read_memory`;
        // a request two stores' plans name is evicted once.
        let preempt: BTreeSet<_> = response.preempt.into_iter().collect();
        let preempted = !preempt.is_empty();
        if preempted {
            let preempt = preempt.into_iter().collect::<Vec<_>>();
            tracing::info!(requests = ?preempt, "a real memory shortage preempts the requests worth least");
            self.preempt(&preempt)?;
        }
        // New work waits while a shortage holds the bank at its floor; the
        // reading that finds none lets it retry.
        let waited = std::mem::replace(&mut self.memory().admission_waits, response.admission_waits);
        if waited && !response.admission_waits {
            self.advance_epoch();
        }
        // The reading changed something where its response moved what we
        // hold (the ledgers' generation past the reading's own), or
        // preempted, or let new work in or held it back.
        let changed = preempted
            || waited != response.admission_waits
            || self.domain().memory_generation() != read.generation;
        const ACTIVE_MIN_REREAD: std::time::Duration = std::time::Duration::from_millis(50);
        const ACTIVE_MAX_REREAD: std::time::Duration = std::time::Duration::from_secs(1);
        const OPPORTUNITY_REREAD: std::time::Duration = std::time::Duration::from_secs(2);
        let reread = match self.domain().memory_standing().map_err(classify_domain_error)? {
            magnitude_executor::MemoryStanding::Active => Reread::Stands {
                cost: read.cost.max(ACTIVE_MIN_REREAD),
                changed,
                ceiling: self.domain().others_interval().map(|interval| interval.clamp(ACTIVE_MIN_REREAD, ACTIVE_MAX_REREAD)),
            },
            magnitude_executor::MemoryStanding::Opportunity => Reread::Stands {
                cost: OPPORTUNITY_REREAD,
                changed: false,
                ceiling: Some(OPPORTUNITY_REREAD),
            },
            magnitude_executor::MemoryStanding::Settled => Reread::Settled,
        };
        Ok(Observed { unload: false, reread })
    }

    /// A resize of `tier`'s storage begun for `cause`, `held` the bytes the
    /// process held as it began: held as begun with the reading its target
    /// was derived from and the terms that moved since the store's last.
    fn begin_resize(
        &mut self,
        tier: BankTier,
        cause: BankBegunFor,
        held: u64,
        reading: magnitude_executor::SizingTerms,
    ) -> Result<(), RequestError> {
        let origin = ResizeOrigin::from_previous(
            self.memory().tiers[tier.index()].sized.replace(reading.clone()),
            &reading,
        );
        let begun = BankBegun { held, cause, reading, origin };
        self.memory().bank_begun(tier, begun).map_err(invariant("expert bank"))
    }

    /// Follow the expert bank's derived target
    /// (`ExecutorDomain::follow_cache_targets`): called only from `observe`
    /// (`between_flights`) for external `HostRam` pressure and bank growth
    /// targets, never for claims (which reclaim synchronously through
    /// `Ledger::admit_or_reclaim`); both shrink paths reach the same bank
    /// shrink release (`ExpertBank::release_evicted`). A resize it begins is
    /// held as begun, a growth as one and a shrink as the target's. One
    /// deferred or running is followed again at the next boundary, which
    /// derives afresh.
    fn follow_cache_targets(
        &mut self,
        bindings: &mut StateBindings,
        sizing: [magnitude_executor::SizingTerms; 2],
    ) -> Result<bool, RequestError> {
        let [before_bank, before_host] =
            [BankTier::Bank, BankTier::Host].map(|tier| self.domain().held_bytes_of(tier));
        let wake = self.completion();
        let followed = self.domain().follow_cache_targets(bindings, wake).map_err(classify_domain_error)?;
        let (tier, cause) = match followed {
            Followed::Grew(BankStep::Begun(tier)) => (tier, BankBegunFor::Growth),
            Followed::Shrank(BankStep::Begun(tier)) => (tier, BankBegunFor::Shrink),
            Followed::Shrank(BankStep::Running | BankStep::Deferred) => return Ok(true),
            Followed::Holds | Followed::Grew(_) | Followed::Shrank(BankStep::AtLimit) => return Ok(false),
        };
        let [bank, host] = sizing;
        let (before, reading) = match tier {
            BankTier::Bank => (before_bank, bank),
            BankTier::Host => (before_host, host),
        };
        self.begin_resize(tier, cause, before, reading)?;
        Ok(cause == BankBegunFor::Shrink)
    }

    /// Free what the ledger took of the load's non-bank cache, as its holders
    /// do: the domain's own kinds at once (`ExecutorDomain::release_yielded`)
    /// and the retained prefixes worth least while their cache is owed
    /// ([`MemoryHost::evict_owed_prefixes`]). The expert bank reclaims
    /// synchronously inside `Ledger::admit_or_reclaim` before any claim is
    /// handed out, never asynchronously on yield; `follow_cache_targets` runs
    /// only in `observe` (`between_flights`) for external `HostRam` pressure
    /// and bank growth.
    fn follow_yields(&mut self, bindings: &mut StateBindings) -> Result<Yielded, RequestError> {
        let (before, started, wake) = (self.domain().held_bytes(), std::time::Instant::now(), self.completion());
        let evicted = self.evict_owed_prefixes();
        let yielded = combine_yielded(evicted, self.domain().release_yielded(bindings, wake).map_err(classify_domain_error)?);
        if let Yielded::Released { released, returned } = yielded {
            tracing::info!(
                micros = started.elapsed().as_micros() as u64,
                held_before = before,
                held_after = self.domain().held_bytes(),
                state_released = released,
                state_returned = returned,
                "memory released: the cache the ledger took"
            );
            self.advance_epoch();
        }
        Ok(yielded)
    }

    /// At a flight boundary, finish the idle graph slots' release off the
    /// submitting thread (`ExecutorDomain::advance_slot_release`), recorded
    /// with the slots and how long the release thread took. It freed
    /// memory, so what waited on memory may retry. Whether one finished.
    fn advance_slot_release(&mut self) -> bool {
        let Some(released) = self.domain().advance_slot_release() else {
            return false;
        };
        tracing::info!(
            slots = released.slots,
            released = released.bytes,
            release_micros = released.released.duration_since(released.began).as_micros() as u64,
            held_after = self.domain().held_bytes(),
            "graph slots released"
        );
        self.advance_epoch();
        true
    }

    /// At a flight boundary, advance the expert bank's resize
    /// (`ExecutorDomain::advance_expert_bank`). A finished one is recorded as
    /// a release is (a shrink as memory released, a growth as memory
    /// grown): how long it held this thread, the bank's slots after it,
    /// what the process held when it began and now, and the reading it
    /// began on with the terms of it that moved. A finished shrink freed
    /// memory: what waited on memory may retry. Whether one finished.
    fn advance_expert_bank(&mut self) -> Result<bool, RequestError> {
        let wake = self.completion();
        let Some((finished, held)) = self.domain().advance_expert_bank(wake).map_err(classify_domain_error)? else {
            return Ok(false);
        };
        let tier = self.domain().expert_holding_storage(finished.holding()).map_err(classify_domain_error)?;
        let after = self.domain().held_bytes_of(tier);
        let BankBegun { held: before, cause, reading, origin } = self.memory().tiers[tier.index()]
            .began
            .take()
            .ok_or_else(|| invariant("expert bank")(format!("a {tier:?} resize finished that never began")))?;
        let at_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| invariant("expert bank")(format!("the clock is before the Unix epoch: {error}")))?
            .as_millis() as u64;
        let resized = |from, to| cause.finished(from, to, at_unix_ms).map_err(invariant("expert bank"));
        // A finished resize has held this thread for a measured time, by
        // source: the call that began it, the boundaries' advances while it
        // ran, and the decisions that began nothing meanwhile.
        let micros = (held.total() * 1e6) as u64;
        let begun_micros = (held.begun * 1e6) as u64;
        let (advances, advance_micros) = (held.advances.count, (held.advances.seconds * 1e6) as u64);
        let (decisions, decision_micros) = (held.decisions.count, (held.decisions.seconds * 1e6) as u64);
        // Each step of the resize, in microseconds: those on this thread, and
        // how long before now the others happened, so a decode stall around
        // it is placed on the decode's timeline.
        let now = std::time::Instant::now();
        let before_now = |at: std::time::Instant| now.duration_since(at).as_micros() as u64;
        // The reading the resize began on: its store, its generation (as
        // its `memory read` record states it), how long before now it was
        // taken, its terms, the generation of the reading the store's last
        // resize began on (none for its first, whose term is `load`), the
        // terms that moved between the two, and the one whose change set
        // the target.
        let (before_generation, term, moved) = (origin.before_generation(), origin.term(), origin.moved());
        let read_before = before_now(reading.read_at);
        let (store, figures) = (reading.store, reading.figures);
        let reading = reading.reading;
        match finished {
            BankResized::Shrunk { holding, from, to, released, returned, times } => {
                self.domain().record_expert_resize(holding, resized(from, to)?);
                tracing::info!(
                    micros,
                    begun_micros,
                    advances,
                    advance_micros,
                    decisions,
                    decision_micros,
                    bank_slots = self.domain().expert_bank_slots(),
                    ?holding,
                    from,
                    to,
                    released,
                    returned,
                    held_before = before,
                    held_after = after,
                    store = ?store,
                    reading,
                    before_generation,
                    term,
                    read_before,
                    figures = ?figures,
                    moved = ?moved,
                    deciding_micros = times.decided.duration_since(held.started).as_micros() as u64,
                    queued_micros = times.queued.as_micros() as u64,
                    planned_micros = times.planned.as_micros() as u64,
                    plan_picked_micros = times.trips[0].picked.as_micros() as u64,
                    plan_served_micros = times.trips[0].served.as_micros() as u64,
                    commit_picked_micros = times.trips[1].picked.as_micros() as u64,
                    commit_served_micros = times.trips[1].served.as_micros() as u64,
                    decided_before = before_now(times.decided),
                    prepared_before = before_now(times.prepared),
                    installed_before = before_now(times.installed),
                    released_before = before_now(times.released),
                    "memory released"
                );
                self.advance_epoch();
            }
            // A growth's `prepared` is when its first slot joined and
            // `installed` its last (`ResizeTimes`).
            BankResized::Grown { holding, from, to, grown, refused, times } => {
                self.domain().record_expert_resize(holding, resized(from, to)?);
                tracing::info!(
                    micros,
                    begun_micros,
                    advances,
                    advance_micros,
                    decisions,
                    decision_micros,
                    bank_slots = self.domain().expert_bank_slots(),
                    ?holding,
                    from,
                    to,
                    grown,
                    refused = refused.as_ref().map(ToString::to_string),
                    held_before = before,
                    held_after = after,
                    store = ?store,
                    reading,
                    before_generation,
                    term,
                    read_before,
                    figures = ?figures,
                    moved = ?moved,
                    decided_before = before_now(times.decided),
                    prepared_before = before_now(times.prepared),
                    installed_before = before_now(times.installed),
                    released_before = before_now(times.released),
                    "memory grown"
                )
            }
            // Nothing changed: no resize of the holding is recorded.
            BankResized::Stopped { holding, at } => {
                tracing::info!(?holding, at, store = ?store, reading, term, read_before, figures = ?figures, moved = ?moved, "the expert bank's growth stopped before its first slot")
            }
            // Only a growth needs memory: a shrink's refusal broke the bank,
            // which reports it as an invariant violation instead.
            BankResized::GrowthRefused { holding, at, refusal } => {
                if cause != BankBegunFor::Growth {
                    return Err(invariant("expert bank")(format!("a shrink of {holding:?} reported a refusal: {refusal}")));
                }
                tracing::warn!(?holding, at, %refusal, store = ?store, reading, term, read_before, figures = ?figures, moved = ?moved, "the expert bank's storage refused a growth")
            }
        }
        Ok(true)
    }

    /// The one path a claim short of memory takes, from its `first` try,
    /// trying again by `attempt`: first free the cache the ledger took for
    /// claims already admitted ([`MemoryStep::follow_yields`]); where that
    /// frees nothing, ask the device's ledger for relief that covers what
    /// the claim lacks, taking cache, and where `preempting`, the requests
    /// that can be evicted now (`Ledger::plan_relief`), and carry it out
    /// whole: its cache yields and its requests are evicted
    /// ([`MemoryHost::preempt`]). Where no relief covers it, nothing yields. Each
    /// that freed anything tries again; one that
    /// frees memory only later (a round's end, slots releasing off this
    /// thread, the expert bank's shrink) leaves the claim pending, and the
    /// caller decides from there whether it waits or fails. Pending is
    /// only ever a change under way, whose end advances the epoch.
    fn relieve<T>(
        &mut self,
        bindings: &mut StateBindings,
        first: Result<T, DomainError>,
        preempting: bool,
        mut attempt: impl FnMut(&mut Self, &mut StateBindings) -> Result<Result<T, DomainError>, RequestError>,
    ) -> Result<Relieved<T>, RequestError> {
        let mut current = first;
        let mut exhausted: BTreeSet<magnitude_executor::memory::ledger::EntryId> = BTreeSet::new();
        loop {
            let pending = match &current {
                Err(DomainError::Capacity(capacity)) => match self.follow_yields(bindings)? {
                    Yielded::Released { .. } => None,
                    Yielded::Pending => Some(true),
                    Yielded::Nothing => {
                        let lacks = capacity.lacks();
                        let evictable = self.evictable();
                        let allow = match preempting {
                            true => Relief::CacheAndRequests(&evictable),
                            false => Relief::CacheOnly,
                        };
                        match self.domain().plan_relief_excluding(capacity.resource, lacks, allow, &exhausted) {
                            None => Some(false),
                            Some(plan) => {
                                let owed = self.domain().execute_relief(plan);
                                let newly_exhausted = owed.exhausted.iter().fold(false, |any, &id| exhausted.insert(id) || any);
                                if !owed.preempt.is_empty() {
                                    self.preempt(&owed.preempt)?;
                                }
                                // Retry when physical memory was freed synchronously
                                // (reclaimed cache or preempted requests) or by
                                // `follow_yields`. If only logical/deferred cache
                                // was drained from the ledger without freeing
                                // physical memory, or a reclaimer reported
                                // `AtFloor`, loop to plan relief from the
                                // remaining cache tiers (e.g. `CacheKind::Bank`)
                                // before giving up.
                                match self.follow_yields(bindings)? {
                                    Yielded::Released { .. } => None,
                                    Yielded::Pending => Some(true),
                                    Yielded::Nothing => match owed.reclaimed > 0 || !owed.preempt.is_empty() {
                                        true => None,
                                        false if owed.yielded > 0 || newly_exhausted => continue,
                                        false => Some(false),
                                    },
                                }
                            }
                        }
                    }
                },
                _ => Some(false),
            };
            match pending {
                Some(pending) => return Ok(Relieved { current, pending }),
                None => current = attempt(self, bindings)?,
            }
        }
    }
}

impl<H: MemoryHost> MemoryStep for H {}

/// The owner's memory step at an event, `bindings` there where no flight
/// holds them: memory read now, the expert bank's running resize advanced
/// and its derived target followed, and a real shortage answered in order,
/// the model unloaded only where it persists. Whether it unloaded. The one
/// step every host runs: the owner at each of its events, and an instrument
/// that drives the domain below the scheduler at each of its own, each of
/// which is a flight's boundary.
pub fn between_flights<H: MemoryHost>(host: &mut H, bindings: Option<&mut StateBindings>) -> Result<bool, RequestError> {
    host.observe(bindings, true).map(|observed| observed.unload)
}

/// The one relief of a claim short of memory, for every host: from its
/// `first` try, retried by `attempt` after each release
/// ([`MemoryStep::relieve`]); where `preempting`, the host's evictable
/// requests may be preempted for it.
pub fn relieve<H: MemoryHost, T>(
    host: &mut H,
    bindings: &mut StateBindings,
    first: Result<T, DomainError>,
    preempting: bool,
    attempt: impl FnMut(&mut H, &mut StateBindings) -> Result<Result<T, DomainError>, RequestError>,
) -> Result<Relieved<T>, RequestError> {
    host.relieve(bindings, first, preempting, attempt)
}

impl<F: ProgramFamily> Service<F> {
    /// Whether `request` waits on memory to become resident: it never was,
    /// and memory is being reclaimed or its last attempt was refused for
    /// memory. Preemption is the only eviction and it counts itself
    /// (`preemption_debt`), so a non-resident request without that debt never
    /// was resident; a preempted one is not refused, mid-prefill included.
    /// Such a request is the one refused once only the model's unload could
    /// make room for it.
    fn waits_on_memory(&self, request: &Request) -> bool {
        request.record.preemption_debt == 0
            && matches!(&request.state, State::Between(_, Stage::NonResident(Some(Wait { on: WaitsOn::Memory, .. }))))
    }

    /// End the waits that have ended in place, then report the requests
    /// selection can start while keeping every record owner-held in
    /// `self.requests`.
    fn selectable_requests(&mut self) -> BTreeSet<RequestId> {
        let epoch = self.epoch;
        let awaiting = self
            .requests
            .iter()
            .filter_map(|(&id, request)| match request.state {
                State::Between(_, Stage::AwaitingPrefix { producer, through }) => {
                    Some((id, producer, through))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        for (id, producer, through) in awaiting {
            let waiter = self.requests[&id].state.generation();
            let cached = self
                .prefix_cache
                .deepest(PrefixPath::of(waiter), waiter.resume_bound())
                >= through;
            if cached {
                if let State::Between(_, stage) = &mut self.requests.get_mut(&id).unwrap().state {
                    *stage = Stage::NonResident(None);
                }
                continue;
            }
            if self
                .requests
                .get(&producer)
                .is_some_and(|peer| self.is_prefix_producer(waiter, peer, through))
            {
                continue;
            }
            let replacement = self
                .requests
                .iter()
                .find(|(&peer_id, peer)| {
                    peer_id != id && self.is_prefix_producer(waiter, peer, through)
                })
                .map(|(&peer_id, _)| peer_id);
            if let State::Between(_, stage) = &mut self.requests.get_mut(&id).unwrap().state {
                *stage = match replacement {
                    Some(producer) => Stage::AwaitingPrefix { producer, through },
                    None => Stage::NonResident(None),
                };
            }
        }
        let mut selectable = BTreeSet::new();
        for (&id, request) in &mut self.requests {
            match &mut request.state {
                State::Between(_, Stage::NonResident(wait) | Stage::Pending(_, wait))
                | State::Parked(_, _, wait) => {
                    if wait.is_some_and(|wait| epoch.changed_since(wait.at)) {
                        *wait = None;
                    }
                }
                State::Between(_, Stage::Ready | Stage::AwaitingPrefix { .. } | Stage::AwaitingCredit) => {}
            }
            let ready = match &request.state {
                State::Between(generation, Stage::Ready) => {
                    generation.finish_reason().is_none()
                        || owes_reconciliation(&request.record, generation)
                }
                State::Between(_, Stage::NonResident(None) | Stage::Pending(_, None))
                | State::Parked(_, _, None) => true,
                State::Between(
                    _,
                    Stage::NonResident(Some(_))
                    | Stage::Pending(_, Some(_))
                    | Stage::AwaitingPrefix { .. }
                    | Stage::AwaitingCredit,
                )
                | State::Parked(_, _, Some(_)) => false,
            };
            if ready {
                selectable.insert(id);
            }
        }
        selectable
    }

    fn select(&mut self, bindings: &mut StateBindings) -> Result<Selected, Fatal> {
        let selectable = self.selectable_requests();
        let mut holder = self.domain.chain_holder();
        // A holder that is no longer scheduled ended mid-chain: its chain
        // concludes, its state the one its reconciled chunks reached.
        if holder.is_some_and(|holder| !selectable.contains(&holder)) {
            self.domain
                .conclude_chain()
                .map_err(|error| (None, classify_domain_error(error)))?;
            holder = None;
        }
        let operations = selectable
            .iter()
            .map(|&id| {
                let request = &self.requests[&id];
                let record = &request.record;
                let generation = request.state.generation();
                let reconciliation = owes_reconciliation(record, generation);
                let prefill = !reconciliation
                    && (!generation.is_resident()
                        || generation.resident_position() < generation.usage().prompt_tokens);
                ScheduledOperation {
                    identity: id,
                    phase: if prefill { Phase::Prefill } else { Phase::Decode },
                    active: record.service_ns > 0,
                    resident: generation.is_resident(),
                    waiting_since_ns: record.waiting_since,
                    service_ns: record.service_ns,
                    preemption_debt: record.preemption_debt,
                    // Its next chunk claims the chain's queued launch before
                    // any other round could orphan it.
                    holds_chain: holder == Some(id),
                }
            })
            .collect::<Vec<_>>();
        let selection = self
            .scheduler
            .select(&operations, self.now)
            .map_err(|error| (None, invariant("scheduler")(error)))?;
        // Idle, the stores' spare state and idle graph slots stay the
        // ledger's cache: they are freed only as it takes them
        // (`ExecutorDomain::release_yielded`), never apart from it.
        let Some(mut selection) = selection else {
            return Ok(Selected::Empty);
        };
        // A decode launch's forwards chain all or none ([`chained`]): it
        // takes the selected requests alike in this with the first, a
        // chained launch within its rows (`ExecutorDomain::chained_rows`),
        // the budget its queued pass was held to. Any other is left
        // out of the selection, uncharged, so its wait carries it ahead of
        // those the launch serves, and it leads a launch of its own kind.
        let chaining = matches!(selection.phase(), Phase::Decode)
            && selection
                .requests()
                .first()
                .is_some_and(|id| chained(&self.requests[id].state));
        let requests = selection
            .requests()
            .iter()
            .copied()
            .filter(|id| {
                !matches!(selection.phase(), Phase::Decode)
                    || chained(&self.requests[id].state) == chaining
            })
            .collect::<Vec<_>>();
        // Requests take rows from one phase-wide token budget in selection
        // order. Additional ready requests wait for the next round.
        let budget = match selection.phase() {
            Phase::Prefill => self.scheduler.limits().prefill_tokens,
            Phase::Decode if chaining => self.domain.chained_rows(),
            Phase::Decode => self.scheduler.limits().decode_tokens,
        };
        // The round's vocabulary work: selected target rows and drafting
        // requests together stay within the selection bound. Every such
        // charge also takes a row from the decode budget, so only a prefill
        // round finishing more prompts than the bound meets it.
        let mut launch = Launch::new(self.scheduler.limits().selection_bound());
        // Guessed rows, where every request decodes in front passes, in the
        // form and as many and as deep as the domain grants (`FrontRows`): a
        // claim of the pass queued behind the last, or one the host builds
        // in the form and to the depth the regime prices least. Such a
        // launch reserves every request before it is priced and builds them
        // after (`Launch::build`), so it prices exactly the requests it
        // builds.
        let mut reserved = Vec::new();
        let mut attempted = 0;
        if !chaining && matches!(selection.phase(), Phase::Decode) {
            let prior = self
                .domain
                .streamed_verify_prior(requests.len())
                .map_err(|error| (None, classify_domain_error(error)))?;
            for id in &requests {
                self.requests[id].state.generation().observe_streamed_prior(prior.clone());
            }
        }
        // Requests whose drafted blocks await verification share the launch
        // that verifies them: its rows go to the proposals most often right,
        // whichever request drafted them.
        let drafted = requests
            .iter()
            .filter_map(|id| Some((*id, self.requests[id].state.generation().draft_outlook()?)))
            .collect::<Vec<_>>();
        let outlooks = drafted
            .iter()
            .map(|(_, outlook)| outlook.clone())
            .collect::<Vec<_>>();
        let shares = magnitude_generation::share_rows(
            &outlooks,
            budget * drafted.len() / requests.len().max(1),
        )
        .map(|shares| {
            drafted
                .iter()
                .map(|(id, _)| *id)
                .zip(shares)
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();
        for &id in &requests {
            // A reserved round holds a row, the fewest it takes, until its
            // launch builds it.
            let remaining_rows = budget.saturating_sub(launch.rows + reserved.len());
            if remaining_rows == 0 {
                break;
            }
            let allowance = if chaining {
                // Built at its exact row and the guessed rows its launch
                // grants it (`FrontRows::grant`), within these.
                remaining_rows
            } else {
                // Share the remaining rows across requests still eligible in
                // this round. A lone request can use the whole budget, while
                // peers get compatible chunks instead of the first request
                // consuming it.
                shares
                    .get(&id)
                    .copied()
                    .unwrap_or_else(|| remaining_rows.div_ceil(requests.len() - attempted))
                    .min(remaining_rows)
            };
            attempted += 1;
            let Request { record, state } = self.requests.remove(&id).expect("selected request");
            let candidate = Candidate::from_state(state);
            // On a fatal error every selected request, started or not, goes
            // back where the owner terminates it with that error: one held
            // here would be dropped without its outcome.
            let reservation = match self.reserve(id, record, candidate, allowance, launch.selections, bindings) {
                Ok(reservation) => reservation,
                Err((_, error)) => {
                    self.return_reserved(reserved);
                    return Err((Some(launch.members), error));
                }
            };
            let start = match reservation {
                Reservation::Returned(request) => Start::Returned(request),
                Reservation::Composed(member, work) => {
                    assert!(!chaining, "a chaining launch selects only ready requests");
                    Start::Joined(member, work)
                }
                Reservation::Round(round) if chaining => {
                    reserved.push((id, round));
                    continue;
                }
                Reservation::Round(round) => {
                    round.build(id, allowance, FrontGrant::UpTo(0), |publishes| self.domain.publish_behind(id, publishes))
                }
            };
            if let Err(error) = launch.join(id, start, &mut self.requests) {
                return Err((Some(launch.members), error));
            }
        }
        if chaining {
            let passes = reserved.iter().map(|(id, round)| (*id, round.front_guess()));
            let fronts = FrontRows::of(passes, budget, |requests, fronted| self.domain.front_pass(requests, fronted));
            let fronts = match fronts {
                Ok(fronts) => fronts,
                Err(error) => {
                    self.return_reserved(reserved);
                    return Err((None, classify_domain_error(error)));
                }
            };
            let domain = &mut self.domain;
            let built = launch.build(reserved, &fronts, &mut self.requests, |id, publishes| {
                domain.publish_behind(id, publishes)
            });
            if let Err(error) = built {
                return Err((Some(launch.members), error));
            }
        }
        if let Err(error) = selection.retain(&requests[..attempted]) {
            return Err((Some(launch.members), invariant("scheduler")(error)));
        }
        if launch.operations.is_empty() {
            self.scheduler.completed(selection, 0);
            return Ok(Selected::Retry);
        }
        Ok(Selected::Round(Round {
            selection,
            started: self.now,
            members: launch.members,
            groups: group(&self.domain, launch.operations).into_iter().collect(),
        }))
    }

    /// Requests reserved for a launch that then failed go back as they were
    /// selected, ready.
    fn return_reserved(&mut self, reserved: Vec<(RequestId, ReservedRound)>) {
        for (id, round) in reserved {
            self.requests.insert(id, round.into_request());
        }
    }

    /// Reserve a selected request's start: everything that can turn it
    /// back, its work's selection charge, its residency and its round's
    /// ([`ReservedRound::reserve`]), so whatever it reserves joins the
    /// launch.
    fn reserve(
        &mut self,
        id: RequestId,
        mut record: Record,
        candidate: Candidate,
        allowance: usize,
        selections: usize,
        bindings: &mut StateBindings,
    ) -> Result<Reservation, Fatal> {
        // Work composed before this round joins as it is, or waits.
        if candidate
            .composed()
            .is_some_and(|work| selection_charge(work) > selections)
        {
            return Ok(Reservation::Returned(Request {
                record,
                state: candidate.into_state(),
            }));
        }
        record.clock.serve(Instant::now());
        let generation = match candidate {
            Candidate::Parked(round, work) => {
                return Ok(Reservation::Composed(
                    Member {
                        record,
                        state: Membership::Started(round),
                    },
                    work.0,
                ));
            }
            Candidate::Pending(generation, work) => {
                return Ok(Reservation::Composed(
                    Member {
                        record,
                        state: Membership::Generation(generation),
                    },
                    work.0,
                ));
            }
            Candidate::Ready(generation) => generation,
            Candidate::NonResident(generation) => {
                let residency = match self.make_resident(id, &mut record, generation, bindings) {
                    Ok(residency) => residency,
                    // No map holds `record` now: it takes the error itself.
                    Err((members, error)) => {
                        if let Some(publication) = record.publication.take() {
                            publication.sender.fail(error.clone());
                        }
                        return Err((members, error));
                    }
                };
                match residency {
                    Residency::Resident(generation, encodes) => match Work::new(encodes) {
                        Some(work) => {
                            return Ok(Reservation::Composed(
                                Member {
                                    record,
                                    state: Membership::Generation(generation),
                                },
                                work.0,
                            ));
                        }
                        None => generation,
                    },
                    Residency::Waiting(generation, wait) => {
                        return Ok(Reservation::Returned(Request {
                            record,
                            state: State::Between(generation, Stage::NonResident(Some(wait))),
                        }));
                    }
                    Residency::Failed(generation) => {
                        return Ok(Reservation::Returned(Request {
                            record,
                            state: State::Between(generation, Stage::Ready),
                        }));
                    }
                }
            }
        };
        Ok(ReservedRound::reserve(record, generation, allowance, selections, &mut self.epoch))
    }

    /// The single transition that makes a request resident: from the deepest
    /// cached prefix of its path below its resume bound, or from fresh state.
    fn make_resident(
        &mut self,
        id: RequestId,
        record: &mut Record,
        mut generation: Generation,
        bindings: &mut StateBindings,
    ) -> Result<Residency, Fatal> {
        let fatal = |error: RequestError| (None, error);
        // A real shortage holds new work back while the bank sits at its
        // floor (`MemoryResponse::admission_waits`).
        if self.memory.admission_waits {
            return Ok(Residency::Waiting(generation, Wait { at: self.epoch, on: WaitsOn::Memory }));
        }
        // It reserves the rows it is about to write as it opens: its prompt,
        // or the history it replays, and a launch's rows ahead
        // (`ExecutorDomain::open_state`), so no prompt row waits on memory.
        // Later rows are reserved ahead at its steps' boundaries
        // (`ExecutorDomain::provision`).
        let rows = generation.resume_bound() as u64;
        // Short of memory, it is relieved as any claim is ([`relieve`]),
        // taking cache alone, its prefix looked up afresh at each try, as an
        // eviction may drop it. While another request's chain is under way
        // the open changes no bindings: it waits for the chain's end, a
        // flight boundary.
        //
        // A disk entry the SSD cold tier holds deeper than the memory cache
        // is read once ([`Self::cold_restore`]); each try opens from it
        // first. Its restore failing other than for memory, or its resume
        // rejected, drops it, and the request opens as though there were no
        // entry, in the same try.
        let prefix = record.prefix.is_some();
        let mut cold = match prefix {
            true => self.cold_restore(&generation),
            false => None,
        };
        let mut restored = false;
        let mut open = |this: &mut Self, bindings: &mut StateBindings| {
                if let Some(restore) = &cold {
                    let Some(unchained) = this.domain.unchained(bindings) else {
                        return Ok(Ok(None));
                    };
                    match this.domain.restore_state(unchained, id, &restore.target, restore.head.as_ref(), rows) {
                        Ok(encodes) => {
                            if this.resume_cold(id, &mut generation, restore)? {
                                restored = true;
                                return Ok(Ok(Some((None, encodes))));
                            }
                        }
                        Err(error @ DomainError::Capacity(_)) => return Ok(Err(error)),
                        Err(error) => {
                            if let Some(Ok(mut tier)) = this.cold_tier.as_ref().map(|tier| tier.try_borrow_mut()) {
                                tier.remove(&restore.key);
                            }
                            tracing::warn!(position = restore.target.position, %error, "cold-tier restore failed; dropped");
                        }
                    }
                }
                cold = None;
                let Some(unchained) = this.domain.unchained(bindings) else {
                    return Ok(Ok(None));
                };
                let hit = match prefix {
                    true => this
                        .prefix_cache
                        .lookup(PrefixPath::of(&generation), generation.resume_bound())
                        .map_err(invariant("prefix cache"))?,
                    false => None,
                };
                let from = match hit {
                    Some(hit) => Some(this.prefix_cache.entry(hit).map_err(invariant("prefix cache"))?.state()),
                    None => None,
                };
                Ok(this.domain.open_state(unchained, id, from, rows).map(|encodes| Some((hit, encodes))))
            };
        let first = open(self, bindings).map_err(fatal)?;
        let relieved = self.relieve(bindings, first, false, open).map_err(fatal)?;
        let (hit, encodes) = match relieved.current {
            Ok(Some(opened)) => opened,
            Ok(None) => return Ok(Residency::Waiting(generation, Wait { at: self.epoch, on: WaitsOn::Chain })),
            // New work waits for memory a change of ours will free.
            Err(DomainError::Capacity(_))
                if relieved.pending || self.progressing_state_beside(&BTreeSet::from([id])) =>
            {
                return Ok(Residency::Waiting(generation, Wait { at: self.epoch, on: WaitsOn::Memory }));
            }
            Err(error) if is_fatal(&error) => return Err(fatal(classify_domain_error(error))),
            // Nothing of ours can make room: it fails at once.
            Err(error) => {
                fail_generation(record, &mut generation, refused_alone(error));
                return Ok(Residency::Failed(generation));
            }
        };
        // A restore resumed as it opened ([`Self::resume_cold`]).
        let resumed = match hit {
            _ if restored => Ok(()),
            Some(hit) => self
                .prefix_cache
                .entry(hit)
                .and_then(|cached| generation.resume_at(Some((hit.position(), cached.method())))),
            None => generation.resume_at(None),
        };
        if let Err(error) = resumed {
            return match self.domain.release_state(&[id]) {
                Ok(()) => {
                    fail_generation(record, &mut generation, invariant("residency")(error));
                    Ok(Residency::Failed(generation))
                }
                Err(rollback) => Err(fatal(invariant("residency")(format!(
                    "{error}; executor rollback failed ({rollback})"
                )))),
            };
        }
        // The table rows of the tokens the host knows it will feed, its path
        // from where its state resumes, are prefetched ahead of their steps.
        let position = generation.resident_position();
        let path = generation.prompt().iter().chain(generation.generated()).skip(position).copied().collect::<Vec<_>>();
        if let Err(error) = self.domain.prefetch_prompt(id, position, &path) {
            return Err(fatal(invariant("stored table prefetch")(error)));
        }
        if let Some(prefix) = &mut record.prefix {
            prefix.source = hit.map(|hit| hit.id());
        }
        // Its host answers it admitted only now: one refused before this
        // never was.
        if let Some(publication) = &record.publication {
            publication.sender.began();
        }
        Ok(Residency::Resident(generation, encodes))
    }

    /// The SSD cold tier's entry to open a request from, read and rebuilt:
    /// the deepest disk entry under its path, below its resume bound
    /// (`resume_at` rejects the row it samples from) and of its method, where
    /// it is deeper than what the memory cache would resume it from and its
    /// read beats the prefill it replaces (`prefer_disk`): a shallow memory
    /// hit (the shared system prompt) must not hide a much deeper entry on
    /// disk. An entry that fails to load, decode or rebuild is dropped. None
    /// without such an entry.
    fn cold_restore(&mut self, generation: &Generation) -> Option<ColdRestore> {
        let tier = self.cold_tier.clone()?;
        let path: Vec<_> = generation
            .prompt()
            .iter()
            .chain(generation.generated().iter())
            .copied()
            .collect();
        let method = match generation.method_choice() {
            magnitude_generation::MethodChoice::Plain => "Plain",
            magnitude_generation::MethodChoice::Mtp { .. } => "Mtp",
            magnitude_generation::MethodChoice::DFlash { .. } => "DFlash",
        };
        let bound = generation.resume_bound();
        let memory_depth = self.prefix_cache.depth(PrefixPath::of(generation), bound).unwrap_or(0);
        let disk = tier.try_borrow().ok()?.lookup_deepest(&path, bound, Some(method));
        let chosen = disk.filter(|&(_, depth, bytes)| prefer_disk(memory_depth, depth, bytes));
        tracing::info!(
            memory_depth,
            disk_depth = disk.map_or(0, |(_, depth, _)| depth),
            disk_bytes = disk.map_or(0, |(_, _, bytes)| bytes),
            resume_bound = bound,
            chosen = match (chosen, memory_depth) {
                (Some(_), _) => "disk",
                (None, 0) => "fresh",
                (None, _) => "memory",
            },
            "cold-tier decision"
        );
        let (key, _, bytes) = chosen?;
        let started = Instant::now();
        let payload = match tier.try_borrow_mut().ok()?.load(&key) {
            Ok(payload) => payload,
            Err(error) => {
                // `load` already evicted the key and deleted its file.
                tracing::warn!(%error, "cold-tier entry failed to load; dropped");
                return None;
            }
        };
        let drop_entry = |reason: &str, error: String| {
            if let Ok(mut guard) = tier.try_borrow_mut() {
                guard.remove(&key);
            }
            tracing::warn!(%error, "cold-tier entry failed to {reason}; dropped");
        };
        let spilled = match decode_spill(&payload) {
            Ok(spilled) => spilled,
            Err(error) => {
                drop_entry("decode", error);
                return None;
            }
        };
        let position = spilled.target.position;
        let method = match spilled.method {
            MethodTag::Plain => MethodCheckpoint::Plain,
            MethodTag::Mtp | MethodTag::DFlash => {
                // A draft checkpoint's `position` excludes its pending, open and
                // skipped rows (`target_rows` adds them back), unlike the target's
                // exported position.
                let carried = spilled.pending.as_ref().map_or(0, |(tokens, _)| tokens.len())
                    + usize::from(spilled.open.is_some())
                    + spilled.skipped;
                match DraftCheckpoint::from_spill(
                    position.saturating_sub(carried),
                    spilled.pending,
                    spilled.open,
                    spilled.skipped,
                ) {
                    Ok(draft) if spilled.method == MethodTag::DFlash => MethodCheckpoint::DFlash(draft),
                    Ok(draft) => MethodCheckpoint::Mtp(draft),
                    Err(error) => {
                        drop_entry("rebuild its draft checkpoint", error);
                        return None;
                    }
                }
            }
        };
        let point = match spilled.point {
            PointTag::Branch => CachePoint::Branch,
            PointTag::PromptBoundary => CachePoint::PromptBoundary,
            PointTag::Terminal => CachePoint::Terminal,
        };
        Some(ColdRestore {
            key,
            bytes,
            target: spilled.target,
            head: spilled.head,
            method,
            point,
            started,
        })
    }

    /// Resume a request opened from `restore`
    /// (`ExecutorDomain::restore_state`) at the restored position, and
    /// retain the restored prefix in the memory cache at the point it was
    /// spilled from; a miss there only leaves it on disk. Whether it
    /// resumed: rejected, the request's state is released, so it opens
    /// again as though there were no entry, and the entry is dropped. A
    /// release that fails is fatal.
    fn resume_cold(&mut self, id: RequestId, generation: &mut Generation, restore: &ColdRestore) -> Result<bool, RequestError> {
        let position = restore.target.position;
        if let Err(error) = generation.resume_at(Some((position, &restore.method))) {
            if let Some(Ok(mut tier)) = self.cold_tier.as_ref().map(|tier| tier.try_borrow_mut()) {
                tier.remove(&restore.key);
            }
            self.domain.release_state(&[id]).map_err(|rollback| {
                invariant("residency")(format!("{error}; executor rollback failed ({rollback})"))
            })?;
            tracing::warn!(position, %error, "cold-tier restore rejected on resume; dropped");
            return Ok(false);
        }
        generation.note_cold_restore();
        if let Ok(state) = self.domain.resume_state(id) {
            let _ = self
                .prefix_cache
                .retain(PrefixPath::of(generation), position, restore.point, state, restore.method.clone());
        }
        tracing::info!(
            restore_hit = true,
            position,
            bytes = restore.bytes,
            restore_ms = restore.started.elapsed().as_millis() as u64,
            "cold-tier restore"
        );
        Ok(true)
    }

    /// Submit the round's next groups until one is on the device or the
    /// round is exhausted.
    fn advance(&mut self, mut round: Round, mut bindings: StateBindings) -> Result<Pipeline<F>, Fatal> {
        loop {
            let Some(group_) = round.groups.pop_front() else {
                self.finish_round(round)
                    .map_err(|error| (None, error))?;
                return Ok(Pipeline::Idle(bindings));
            };
            // Work of a request that ended while the round ran is dropped.
            let live = group_
                .operations()
                .iter()
                .all(|operation| {
                    round
                        .members
                        .get(&operation.request())
                        .is_some_and(|member| !ended(member.state.generation()))
                });
            if !live {
                let live = group_
                    .into_operations()
                    .into_iter()
                    .filter(|operation| {
                        round
                            .members
                            .get(&operation.request())
                            .is_some_and(|member| !ended(member.state.generation()))
                    })
                    .collect::<Vec<_>>();
                for group_ in group(&self.domain, live).into_iter().rev() {
                    round.groups.push_front(group_);
                }
                continue;
            }
            let chunk_rows = self.scheduler.limits().prefill_tokens.max(1);
            let decode_owed = round.selection.contended().then(|| self.scheduler.limits().decode_owed());
            let alone = round.members.len() == 1
                && !self
                    .requests
                    .values()
                    .any(|other| competes(other, self.status(other)));
            // A chain opens only behind the round's last group: a later one would orphan it.
            let last = round.groups.is_empty();
            let planned = self.domain.plan_prompt(decode_owed, alone, last, |chunks| {
                planned_prefill(&round, group_.operations(), chunk_rows, chunks)
            });
            if let Err(error) = planned {
                return Err((Some(round.members), error));
            }
            let group_ = match self.provision(&mut bindings, &mut round, group_) {
                Ok(Provisioned::Fits(group_)) => group_,
                Ok(Provisioned::Split(leading, trailing)) => {
                    round.groups.push_front(trailing);
                    round.groups.push_front(leading);
                    continue;
                }
                Ok(Provisioned::Short(group_, wait)) => {
                    self.park(&mut round, group_, wait);
                    continue;
                }
                Ok(Provisioned::Refused(group_, error)) => {
                    self.refuse(&mut round, &group_, error);
                    continue;
                }
                Err(error) => return Err((Some(round.members), error)),
            };
            match submit_group(&mut self.domain, bindings, &group_, self.wakes.completion()) {
                Ok(flight) => {
                    let mut prefix_uses = Vec::new();
                    let requests = group_
                        .operations()
                        .iter()
                        .map(Operation::request)
                        .collect::<BTreeSet<_>>();
                    for request in requests {
                        let source = round
                            .members
                            .get(&request)
                            .and_then(|member| member.record.prefix.as_ref()?.source);
                        if let Some(source) = source {
                            match self.prefix_cache.begin_submitted_if_cached(source) {
                                Ok(true) => prefix_uses.push(source),
                                Ok(false) => {}
                                Err(error) => {
                                    return Err((Some(round.members), invariant("prefix cache")(error)));
                                }
                            }
                        }
                    }
                    return Ok(Pipeline::Round(
                        round,
                        ActiveGroup {
                            flight,
                            operations: group_.into_operations(),
                            prefix_uses,
                        },
                    ));
                }
                Err(SubmitFailure::Refused(error, returned)) => {
                    bindings = returned;
                    match error {
                        DomainError::Input(error) => self.refuse(&mut round, &group_, RequestError::Input(error)),
                        other => return Err((Some(round.members), classify_domain_error(other))),
                    }
                }
                Err(SubmitFailure::Failed(error)) => {
                    return Err((Some(round.members), classify_domain_error(error)));
                }
            }
        }
    }

    /// Provision a group's state and check its reservation. A requirement
    /// and a growth claim are one capacity question: recheck both after each
    /// release. Selection is provisional until every exact requirement is
    /// available: the sequences the device's ledger names for the group's
    /// rows are evicted (the ledger took every cheaper byte, cache worth
    /// least first, before it named them), the cache it took is freed where
    /// the device would not give the memory yet, then the group sheds its
    /// trailing operations, which loses no work. A request's state growing
    /// short takes the round's other members out of the round first, so
    /// they may be preempted for it rather than stand beside it. What is
    /// still short waits for a change of ours that frees memory (a request
    /// standing beside it ending, a named sequence leaving its round, the
    /// cache taken freeing off this thread); where none can come, its
    /// requests fail, a request's state alone past what the load can give
    /// it with [`RequestError::ContextExceedsMemory`].
    fn provision(
        &mut self,
        bindings: &mut StateBindings,
        round: &mut Round,
        group_: OperationGroup,
    ) -> Result<Provisioned, RequestError> {
        let short = |current: &Result<(), DomainError>| matches!(current, Err(DomainError::Capacity(_)));
        let mut group_ = group_;
        let Relieved { mut current, mut pending } = self.fitted(bindings, Some(&round.members), &group_, true)?;
        // A request's state growing short with nothing else to free: the
        // round's other members stand beside it only until they leave the
        // round. They leave it now, ready, so the relief may preempt them,
        // the lowest value first, and a growth never waits on requests that
        // may themselves be waiting on it.
        let growth = |current: &Result<(), DomainError>| {
            matches!(current, Err(DomainError::Capacity(capacity)) if capacity.resource == ResourceKind::RequestState)
        };
        let own = group_.operations().iter().map(Operation::request).collect::<BTreeSet<_>>();
        let others = round.members.keys().copied().filter(|id| !own.contains(id)).collect::<BTreeSet<_>>();
        if growth(&current) && !pending && !others.is_empty() {
            self.withdraw(round, &others);
            Relieved { current, pending } = self.relieve(bindings, current, true, |this, bindings| Ok(this.fits(bindings, Some(&round.members), &group_)))?;
        }
        if short(&current) {
            group_ = match group_.split_last() {
                Ok((leading, trailing)) => return Ok(Provisioned::Split(leading, trailing)),
                Err(whole) => whole,
            };
        }
        match current {
            Ok(()) => Ok(Provisioned::Fits(group_)),
            Err(DomainError::Capacity(_)) if pending || self.others_stand(&group_) => Ok(Provisioned::Short(group_, Wait { at: self.epoch, on: WaitsOn::Memory })),
            Err(error @ (DomainError::Capacity(_) | DomainError::Input(_))) => Ok(Provisioned::Refused(group_, refused_alone(error))),
            Err(error) => Err(classify_domain_error(error)),
        }
    }

    /// Whether the group fits ([`Self::fits`]), `members` its round's where
    /// it is a round's (a scoring flight's is none's), its shortage relieved
    /// as any claim's is ([`relieve`]), where `preempting` by preempting the
    /// requests that can be evicted now. Retained prefixes never make the
    /// stores grow past the workload state the plan charges: the group takes
    /// the rows and banks of the least recently used ones instead. Once none
    /// is left to release, `fits` sets no ceiling.
    fn fitted(
        &mut self,
        bindings: &mut StateBindings,
        members: Option<&BTreeMap<RequestId, Member>>,
        group_: &OperationGroup,
        preempting: bool,
    ) -> Result<Relieved<()>, RequestError> {
        let mut current = self.fits(bindings, members, group_);
        let held = self.held_entries(members);
        while matches!(current, Err(DomainError::StateCeiling { .. })) {
            self.prefix_cache.release_unspared(&held);
            current = self.fits(bindings, members, group_);
        }
        self.relieve(bindings, current, preempting, |this, bindings| Ok(this.fits(bindings, members, group_)))
    }

    /// Whether the group fits now: provision it, then check the reservation
    /// its requirement, derived from the layout it was just provisioned
    /// into, would take. While a retained prefix could be released instead,
    /// state growth stops at the workload state the plan charges.
    fn fits(
        &mut self,
        bindings: &mut StateBindings,
        members: Option<&BTreeMap<RequestId, Member>>,
        group_: &OperationGroup,
    ) -> Result<(), DomainError> {
        let held = self.held_entries(members);
        let ceiling = self
            .prefix_cache
            .holds_unspared(&held)
            .then(|| self.prefix_cache.workload_state_bytes());
        self.domain.provision(bindings, group_.operations(), ceiling)?;
        let requirement = requirements(&self.domain, group_)?;
        self.domain
            .can_reserve(&requirement)
            .map_err(DomainError::from)
    }

    /// Fail every request of a group the domain refused with `error`: what
    /// it held frees as each retires.
    fn refuse(&mut self, round: &mut Round, group_: &OperationGroup, error: RequestError) {
        let requests = group_.operations().iter().map(Operation::request).collect::<BTreeSet<_>>();
        for request in requests {
            Self::fail_member(round, request, error.clone());
        }
        self.epoch.advance();
    }

    /// Take `requests` out of `round`, preserving any work queued for them in
    /// its later groups (without a memory wait) so a peer that is not evicted
    /// resumes that work on the next round, while one that is evicted drops it
    /// with its state.
    fn withdraw(&mut self, round: &mut Round, requests: &BTreeSet<RequestId>) {
        let mut withdrawn = Vec::new();
        let mut kept = Vec::new();
        for queued in std::mem::take(&mut round.groups) {
            let (theirs, others): (Vec<_>, Vec<_>) = queued
                .into_operations()
                .into_iter()
                .partition(|operation| requests.contains(&operation.request()));
            withdrawn.extend(theirs);
            kept.extend(others);
        }
        round.groups = group(&self.domain, kept).into_iter().collect();
        for &request in requests {
            let Some(Member { record, state }) = round.members.remove(&request) else { continue };
            let theirs = withdrawn
                .iter()
                .filter(|operation| operation.request() == request)
                .cloned()
                .collect::<Vec<_>>();
            let state = match Work::new(theirs) {
                Some(work) => match state {
                    Membership::Started(started) => State::Parked(started, work, None),
                    Membership::Generation(generation) => {
                        State::Between(generation, Stage::Pending(work, None))
                    }
                },
                None => {
                    let generation = match state {
                        Membership::Started(started) => started.abandon(),
                        Membership::Generation(generation) => generation,
                    };
                    let stage = match generation.is_resident() {
                        true => Stage::Ready,
                        false => Stage::NonResident(None),
                    };
                    State::Between(generation, stage)
                }
            };
            self.requests.insert(request, Request { record, state });
        }
    }

    fn park(&mut self, round: &mut Round, group_: OperationGroup, wait: Wait) {
        let requests = group_
            .operations()
            .iter()
            .map(Operation::request)
            .collect::<BTreeSet<_>>();
        tracing::info!(
            requests = ?requests,
            rows = group_.operations().iter().map(Operation::row_count).sum::<usize>(),
            "a group waits on memory"
        );
        let mut operations = group_.into_operations();
        let groups = std::mem::take(&mut round.groups);
        let mut kept = Vec::new();
        for queued in groups {
            let (theirs, others): (Vec<_>, Vec<_>) = queued
                .into_operations()
                .into_iter()
                .partition(|operation| requests.contains(&operation.request()));
            operations.extend(theirs);
            kept.extend(others);
        }
        round.groups = group(&self.domain, kept).into_iter().collect();
        for request in requests {
            let Some(Member { record, state }) = round.members.remove(&request) else {
                continue;
            };
            let theirs = operations
                .iter()
                .filter(|operation| operation.request() == request)
                .cloned()
                .collect::<Vec<_>>();
            let work = Work::new(theirs).expect("a parked request has work in the group");
            let state = match state {
                Membership::Started(started) => State::Parked(started, work, Some(wait)),
                Membership::Generation(generation) => {
                    State::Between(generation, Stage::Pending(work, Some(wait)))
                }
            };
            self.requests.insert(request, Request { record, state });
        }
    }

    /// Reconcile the completed group and take back the bindings.
    fn reconcile(&mut self, mut round: Round, active: ActiveGroup<F>) -> Result<Pipeline<F>, Fatal> {
        let ActiveGroup {
            flight,
            operations,
            prefix_uses,
        } = active;
        for source in prefix_uses {
            if let Err(error) = self.prefix_cache.end_submitted(source) {
                return Err((Some(round.members), invariant("prefix cache")(error)));
            }
        }
        let (outcomes, continuation) = match finish(&mut self.domain, flight, closes_prompt(&round, &operations)) {
            Ok(finished) => finished,
            Err(error) => return Err((Some(round.members), classify_domain_error(error))),
        };
        let reconciled = match outcomes {
            FlightOutcomes::Target(pending) => self.reconcile_target(&mut round, pending, &operations),
            FlightOutcomes::Head(pending) => self.reconcile_method(&mut round, pending, &operations),
            FlightOutcomes::Vision(item) => self.reconcile_vision(&mut round, item, &operations),
            FlightOutcomes::Priming(_) => Ok(()),
        };
        if let Err(error) = reconciled {
            return Err((Some(round.members), error));
        }
        let mut bindings = match continuation {
            magnitude_executor::TargetContinuation::Ready(bindings) => bindings,
            magnitude_executor::TargetContinuation::Priming(priming) => {
                // Publish the accepted target token while the entry still owns
                // the binding right. Its own completion wakes the event pump;
                // controls remain responsive and no checkpoint can observe a
                // half-resolved pair of numerical lanes.
                self.publish_members(&mut round);
                let mut flight = DomainFlight::Priming(priming);
                flight.completion().notify(self.wakes.completion());
                return Ok(Pipeline::Round(round, ActiveGroup {
                    flight, operations: Vec::new(), prefix_uses: Vec::new(),
                }));
            }
        };
        self.epoch.advance();
        // The flight's boundary: memory is read once here.
        if let Err(error) = self.step(Some(&mut bindings), true) {
            return Err((Some(round.members), error));
        }
        if self.unloaded {
            // The unload terminated every request outside the round; the
            // round's members end with them, and the round with its flight.
            let error = RequestError::ModelUnloaded { cause: ModelUnloadCause::MemoryPressure };
            self.terminalize_all(error, Some(&mut round.members));
            self.finish_round(round).map_err(|error| (None, error))?;
            return Ok(Pipeline::Idle(bindings));
        }
        self.publish_members(&mut round);
        self.advance(round, bindings)
    }

    fn reconcile_target(
        &mut self,
        round: &mut Round,
        pending: Vec<PendingOperationOutcome>,
        operations: &[Operation],
    ) -> Result<(), RequestError> {
        let expected = operations
            .iter()
            .map(|operation| (operation.request(), operation))
            .collect::<BTreeMap<_, _>>();
        let actual = pending
            .iter()
            .map(PendingOperationOutcome::request)
            .collect::<BTreeSet<_>>();
        if pending.len() != operations.len()
            || actual.len() != pending.len()
            || actual != expected.keys().copied().collect()
        {
            return Err(invariant("service target flight")(
                "executor outcomes do not match submitted request identities".into(),
            ));
        }
        let mut followups = Vec::new();
        let sequences = pending.len();
        let rows = operations.iter().map(Operation::row_count).sum::<usize>();
        for item in pending {
            let request = item.request();
            if !matches!(expected[&request], Operation::Forward { .. }) {
                return Err(invariant("service target flight")(
                    "target flight contains a non-forward operation".into(),
                ));
            }
            let Some(Member { mut record, state }) = round.members.remove(&request) else {
                self.domain.abort(item).map_err(classify_domain_error)?;
                continue;
            };
            // The member returns to the round before any fatal error is
            // returned: the owner terminates the round's members with it, and
            // a member held here would be dropped without its outcome.
            let (state, fatal) = match state {
                Membership::Started(mut started) if !ended(started.generation()) => {
                    started.timed(TimedLaunch::Target, item.launch_duration(), sequences, rows);
                    let output = started.generation().generated().len();
                    match reconcile_forward(started, &mut self.domain, request, item) {
                        Ok((generation, effects)) => {
                            let accepted = generation.generated().len().saturating_sub(output);
                            if accepted > 0 {
                                record.clock.accepted(Instant::now());
                            }
                            self.throughput.append(Duration::from_nanos(self.now), accepted as u64);
                            followups.extend(effects.operations);
                            (Membership::Generation(generation), None)
                        }
                        // Interrupted: the generation runs the round again
                        // (or re-prefills), unfailed.
                        Err((generation, RoundError::Interrupted(_))) => (Membership::Generation(generation), None),
                        Err((mut generation, RoundError::Logical(detail))) => {
                            fail_generation(
                                &mut record,
                                &mut generation,
                                invariant("generation round")(detail),
                            );
                            (Membership::Generation(generation), None)
                        }
                        Err((generation, RoundError::Physical(error))) if is_fatal(&error) => {
                            (Membership::Generation(generation), Some(classify_domain_error(error)))
                        }
                        Err((mut generation, RoundError::Physical(error))) => {
                            fail_generation(
                                &mut record,
                                &mut generation,
                                classify_domain_error(error),
                            );
                            (Membership::Generation(generation), None)
                        }
                    }
                }
                state => {
                    let fatal = self.domain.abort(item).map_err(classify_domain_error).err();
                    (state, fatal)
                }
            };
            round.members.insert(request, Member { record, state });
            if let Some(error) = fatal {
                return Err(error);
            }
        }
        if !followups.is_empty() {
            round
                .groups
                .extend(group(&self.domain, followups));
        }
        Ok(())
    }

    fn reconcile_method(
        &mut self,
        round: &mut Round,
        pending: Vec<PendingOperationOutcome>,
        operations: &[Operation],
    ) -> Result<(), RequestError> {
        let expected = operations
            .iter()
            .map(|operation| (operation.request(), operation))
            .collect::<BTreeMap<_, _>>();
        let actual = pending
            .iter()
            .map(PendingOperationOutcome::request)
            .collect::<BTreeSet<_>>();
        if pending.len() != operations.len()
            || actual.len() != pending.len()
            || actual != expected.keys().copied().collect()
        {
            return Err(invariant("service method flight")(
                "method outcomes do not match submitted requests".into(),
            ));
        }
        let sequences = pending.len();
        let rows = operations.iter().map(Operation::row_count).sum::<usize>();
        for item in pending {
            let request = item.request();
            let operation = expected[&request];
            let correct = matches!(operation, Operation::Head { .. })
                && matches!(item.outcome(), Outcome::Head { .. });
            let Some(member) = round.members.get_mut(&request) else {
                self.domain.abort(item).map_err(classify_domain_error)?;
                continue;
            };
            let Membership::Generation(generation) = &mut member.state else {
                return Err(invariant("service method flight")(
                    "drafter work reached a request with a started target round".into(),
                ));
            };
            if ended(generation) {
                self.domain.abort(item).map_err(classify_domain_error)?;
                continue;
            }
            if !correct {
                self.domain.abort(item).map_err(classify_domain_error)?;
                fail_generation(
                    &mut member.record,
                    generation,
                    invariant("service method flight")(
                        "method outcome differs from submitted operation".into(),
                    ),
                );
                continue;
            }
            if let Operation::Head { proposals, .. } = operation {
                let launch = if proposals.is_empty() {
                    TimedLaunch::Entry
                } else {
                    TimedLaunch::Draft
                };
                generation.timed(launch, item.launch_duration(), sequences, rows);
            }
            let transition = match generation.prepare_method_transition(operation, item.outcome()) {
                Ok(transition) => transition,
                Err(detail) => {
                    self.domain.abort(item).map_err(classify_domain_error)?;
                    fail_generation(
                        &mut member.record,
                        generation,
                        invariant("generation method transition")(detail),
                    );
                    continue;
                }
            };
            match self.domain.reconcile(
                item,
                transition.decision(),
            ) {
                Ok(()) => generation.commit_method_transition(transition),
                Err(error) if is_fatal(&error) => return Err(classify_domain_error(error)),
                Err(error) => fail_generation(
                    &mut member.record,
                    generation,
                    classify_domain_error(error),
                ),
            }
        }
        Ok(())
    }

    fn reconcile_vision(
        &mut self,
        round: &mut Round,
        item: PendingOperationOutcome,
        operations: &[Operation],
    ) -> Result<(), RequestError> {
        let request = item.request();
        let valid = matches!(operations,
            [Operation::Encode { request: expected, .. }] if *expected == request)
            && matches!(item.outcome(), Outcome::Encode { .. });
        if !valid {
            return Err(invariant("service vision flight")(
                "vision outcome differs from submitted request".into(),
            ));
        }
        let Some(member) = round.members.get_mut(&request) else {
            return self.domain.abort(item).map_err(classify_domain_error);
        };
        if ended(member.state.generation()) {
            return self.domain.abort(item).map_err(classify_domain_error);
        }
        match self.domain.reconcile(item, PhysicalDecision::rows(0)) {
            Ok(()) => Ok(()),
            Err(error) if is_fatal(&error) => Err(classify_domain_error(error)),
            Err(error) => {
                Self::fail_member(round, request, classify_domain_error(error));
                Ok(())
            }
        }
    }

    /// Fail a round member with `error`; its started round, if any, ends.
    fn fail_member(round: &mut Round, request: RequestId, error: RequestError) {
        if let Some(Member { mut record, state }) = round.members.remove(&request) {
            let mut generation = state.terminate(false);
            fail_generation(&mut record, &mut generation, error);
            round.members.insert(
                request,
                Member {
                    record,
                    state: Membership::Generation(generation),
                },
            );
        }
    }

    /// The round is exhausted: charge its service and return its members.
    fn finish_round(&mut self, round: Round) -> Result<(), RequestError> {
        let Round {
            selection,
            started,
            members,
            ..
        } = round;
        let elapsed = self.now.saturating_sub(started);
        self.domain.served_round(selection.phase() == Phase::Decode);
        let count = selection.requests().len() as u64;
        let base = elapsed / count;
        let remainder = elapsed % count;
        let mut members = members;
        for (index, request) in selection.requests().iter().enumerate() {
            let share = base + u64::from((index as u64) < remainder);
            let record = match members.get_mut(request) {
                Some(member) => &mut member.record,
                None => match self.requests.get_mut(request) {
                    Some(request) => &mut request.record,
                    None => continue,
                },
            };
            record.service_ns = record.service_ns.saturating_add(share);
            record.waiting_since = self.now;
        }
        self.scheduler.completed(selection, elapsed);
        let ids = members.keys().copied().collect::<Vec<_>>();
        for (id, Member { mut record, state }) in members {
            let generation = match state {
                Membership::Generation(generation) => generation,
                // Every submitted target operation reconciled and every
                // unsubmitted one parked or ended, so a started round left
                // here lost its work: it fails.
                Membership::Started(started) => {
                    let mut generation = started.fail();
                    fail_generation(
                        &mut record,
                        &mut generation,
                        invariant("service round")(
                            "a started round ended without its operation".into(),
                        ),
                    );
                    generation
                }
            };
            let stage = if generation.is_resident() {
                Stage::Ready
            } else {
                Stage::NonResident(None)
            };
            self.requests.insert(
                id,
                Request {
                    record,
                    state: State::Between(generation, stage),
                },
            );
        }
        for id in ids {
            self.retain_branch_point(id)?;
            self.retain_prefill_boundary(id)?;
            self.retain_terminal(id)?;
        }
        self.publish_ready()
    }

    /// A request whose composed work is complete: its record and generation.
    /// Retention must wait for pending method work even after target replay
    /// reaches the accepted boundary.
    fn between(&self, id: RequestId) -> Option<(&Record, &Generation)> {
        let request = self.requests.get(&id)?;
        Some((&request.record, request.state.settled()?))
    }

    /// Retain the request's reconciled state at a planned branch point it
    /// has reached: the recurrent bank and method state there, sharing every
    /// history row before it with the request itself. Points it has passed
    /// (it resumed beyond them) or can no longer reach are dropped.
    fn retain_branch_point(&mut self, id: RequestId) -> Result<(), RequestError> {
        let Some((record, generation)) = self.between(id) else {
            return Ok(());
        };
        let Some(prefix) = record.prefix.as_ref() else {
            return Ok(());
        };
        if prefix.branches.is_empty() {
            return Ok(());
        }
        let resident = generation.resident_position();
        let live = generation.finish_reason().is_none() && generation.is_resident();
        let reached = live && prefix.branches.contains(&resident);
        let branches = &mut self.requests.get_mut(&id).unwrap().record.prefix.as_mut().unwrap().branches;
        if live {
            branches.retain(|&branch| branch > resident);
        } else {
            branches.clear();
        }
        if reached {
            self.cache_prefix(id, CachePoint::Branch)?;
        }
        Ok(())
    }

    /// Cache the request's reconciled state as the prefix of its path ending
    /// where that state does: a chain it holds concludes first, its state
    /// the one its reconciled chunks reached, and the prefix's position is
    /// read from the state then. The generation's resident position, which
    /// advanced as those chunks reconciled, is the same.
    fn cache_prefix(&mut self, id: RequestId, point: CachePoint) -> Result<(), RequestError> {
        if self.domain.chain_holder() == Some(id) {
            self.domain.conclude_chain().map_err(classify_domain_error)?;
        }
        let state = self
            .domain
            .resume_state(id)
            .map_err(invariant("prefix retention"))?;
        let position = state.position();
        let Some(Request {
            state: State::Between(generation, _),
            ..
        }) = self.requests.get(&id)
        else {
            return Ok(());
        };
        if generation.resident_position() != position {
            return Err(invariant("prefix retention")(
                "cached prefix state differs from its path position".into(),
            ));
        }
        let method = generation
            .method_checkpoint()
            .map_err(invariant("prefix retention"))?;
        let retained = self
            .prefix_cache
            .retain(PrefixPath::of(generation), position, point, state, method)
            .map_err(invariant("prefix retention"))?;
        if retained {
            let held = self.held_entries(None);
            let domain = &self.domain;
            // The state every holder occupies, the cached prefixes included,
            // read from the stores as each release drops its entry.
            self.prefix_cache.release_beyond_workload(&held, || domain.state_bytes().occupied());
        }
        Ok(())
    }

    /// Retain the request's prompt one row before its end (see
    /// [`prompt_boundary`]): the deepest point a request with the same prompt
    /// can resume from, since it must compute the row it samples from.
    fn retain_prefill_boundary(&mut self, id: RequestId) -> Result<(), RequestError> {
        let Some((record, generation)) = self.between(id) else {
            return Ok(());
        };
        let Some(boundary) = prompt_boundary(record, generation) else {
            return Ok(());
        };
        if ended(generation)
            || generation.resident_position() != boundary
            || !generation.is_resident()
        {
            return Ok(());
        }
        self.cache_prefix(id, CachePoint::PromptBoundary)?;
        self.requests.get_mut(&id).unwrap().record.prefix.as_mut().unwrap().prefill_retained = true;
        Ok(())
    }

    /// Cache a finished request's whole path, prompt and reply, or a stopped
    /// one's as far as its state reached: the next turn of a conversation
    /// extends it, and a client that stops a request and sends the
    /// conversation again resumes where it stopped.
    fn retain_terminal(&mut self, id: RequestId) -> Result<(), RequestError> {
        let Some((record, generation)) = self.between(id) else {
            return Ok(());
        };
        if !record
            .prefix
            .as_ref()
            .is_some_and(|prefix| prefix.terminal_retention == TerminalRetention::Pending)
            || !generation.is_resident()
        {
            return Ok(());
        }
        match generation.finish_reason() {
            // A completed request's whole path, once its state is reconciled
            // to it (`owes_reconciliation`).
            Some(FinishReason::Stop | FinishReason::Length | FinishReason::Context) => {
                if generation.resident_position() != generation.reconciled_position() {
                    if generation.pending_reconciliation().is_some() {
                        return Ok(());
                    }
                    return Err(invariant("prefix retention")(
                        "terminal numerical state is not reconciled at the accepted boundary".into(),
                    ));
                }
            }
            // A stopped request's work is the rows its state holds, cached
            // where it reached without computing more, where a request can
            // resume from it (`resumes_from_stop`).
            Some(FinishReason::Cancelled) => {
                let position = generation.resident_position();
                let resumable = generation
                    .method_checkpoint()
                    .is_ok_and(|method| resumes_from_stop(generation.layout(), position, &method));
                if !resumable {
                    return Ok(());
                }
            }
            _ => return Ok(()),
        }
        self.cache_prefix(id, CachePoint::Terminal)?;
        self.requests.get_mut(&id).unwrap().record.prefix.as_mut().unwrap().terminal_retention =
            TerminalRetention::Retained;
        Ok(())
    }

    /// Drain round members' accepted output into their reserved permits, so
    /// publication overlaps the next submission.
    fn publish_members(&mut self, round: &mut Round) {
        let mut closed = Vec::new();
        for (&id, member) in round.members.iter_mut() {
            if let Member {
                record,
                state: Membership::Generation(generation),
            } = member
            {
                if Self::drain(record, generation) {
                    closed.push(id);
                }
            }
        }
        // A closed receiver releases its request.
        for id in closed {
            if let Some(Member { mut record, state }) = round.members.remove(&id) {
                let mut generation = state.terminate(true);
                generation.discard_output();
                record.release_publication();
                round.members.insert(
                    id,
                    Member {
                        record,
                        state: Membership::Generation(generation),
                    },
                );
            }
        }
    }

    /// Drain accepted output into reserved permits. Returns whether the
    /// receiver has closed.
    fn drain(record: &mut Record, generation: &mut Generation) -> bool {
        record
            .publication
            .as_mut()
            .is_some_and(|publication| publication.drain(generation))
    }

    /// Publish every request outside the pipeline, and retire the ones with
    /// nothing left for their host.
    fn publish_ready(&mut self) -> Result<(), RequestError> {
        let ids = self.requests.keys().copied().collect::<Vec<_>>();
        for id in ids {
            let request = self.requests.get_mut(&id).expect("listed request");
            let State::Between(generation, _) = &mut request.state else {
                continue;
            };
            if Self::drain(&mut request.record, generation) {
                self.terminate_request(id, true);
            }
            let request = &self.requests[&id];
            let Some(generation) = request.state.settled() else {
                continue;
            };
            let record = &request.record;
            let finished = generation.finish_reason().is_some()
                && record.publication.as_ref().is_none_or(|publication| !publication.has_pending())
                && generation.output_len() == 0
                && !owes_reconciliation(record, generation);
            if !finished {
                continue;
            }
            self.retain_terminal(id)?;
            let request = self.requests.get_mut(&id).expect("listed request");
            let State::Between(generation, _) = &request.state else {
                continue;
            };
            let record = &mut request.record;
            if let Some(publication) = record.publication.take() {
                match generation.finish_reason().expect("terminal generation") {
                    FinishReason::Failed => publication.sender.fail(record.error.clone().unwrap_or_else(|| {
                        RequestError::Invariant(InvariantError {
                            context: "service request",
                            detail: "failed without a classified cause".into(),
                        })
                    })),
                    finish => publication.sender.complete(
                        finish,
                        generation.detailed_usage(),
                        generation.method_identity().to_owned(),
                        record.clock.timings(Instant::now()),
                    ),
                }
            }
            self.retire(id)?;
        }
        Ok(())
    }

    /// Close a terminal request's executor state and forget it.
    fn retire(&mut self, id: RequestId) -> Result<(), RequestError> {
        self.domain.close(id).map_err(invariant("request retirement"))?;
        let request = self.requests.remove(&id).expect("retiring a terminal request");
        request.record.retirement.retire();
        self.epoch.advance();
        Ok(())
    }

    /// Whether `request` can be evicted now: live and resident outside a
    /// round, its state no flight's: between rounds, its pending work
    /// unsubmitted, or its round parked on memory, which eviction discards
    /// for it to replay (`StartedRound::evicted`). A round's members are
    /// out of the map, and a finished request keeps its state until it
    /// retires. Without the parked, two requests each parked on the
    /// other's memory would wait on each other.
    fn can_evict(request: &Request) -> bool {
        let live = |generation: &Generation| generation.is_resident() && generation.finish_reason().is_none();
        match &request.state {
            State::Between(generation, Stage::Ready | Stage::AwaitingCredit | Stage::Pending(..)) => live(generation),
            // Parked on memory: its round's next operation is unsubmitted,
            // so its state is no flight's.
            State::Parked(started, ..) => live(started.generation()),
            State::Between(_, Stage::AwaitingPrefix { .. } | Stage::NonResident(_)) => false,
        }
    }

    /// The one eviction: `sequences`, each evictable now, have their state
    /// released and wait non-resident, a preemption owed.
    fn evict(&mut self, sequences: &[RequestId]) -> Result<(), DomainError> {
        self.domain.release_state(sequences).map_err(release_invariant("preemption"))?;
        self.evicted(sequences)
    }

    /// An idle owner never holds a request waiting on memory nothing can
    /// free: with no other progressing request holding state, no reading owed
    /// and no change under way (a bank resize begun, graph slots releasing),
    /// no epoch advance can come, so a wait on memory standing then is an
    /// invariant violation, never a silent park.
    fn check_no_unwakeable_wait(&self) {
        let under_way = self.memory.admission_waits
            || self.memory.reading.is_owed()
            || self.memory.tiers.iter().any(|tier| tier.began.is_some())
            || self.domain.releasing_slots();
        if under_way {
            return;
        }
        for (id, request) in &self.requests {
            if let Some(Wait { at, on: WaitsOn::Memory }) = request.state.wait() {
                assert!(
                    self.epoch.changed_since(at)
                        || self.progressing_state_beside(&BTreeSet::from([*id])),
                    "invariant violated: an idle owner holds {id:?} waiting on memory that nothing under way can free"
                );
            }
        }
        if let Some(walks) = &self.scoring {
            if let Some(at) = walks.waits {
                assert!(
                    self.epoch.changed_since(at) || self.domain.holds_state_beside(&walks.sequences()),
                    "invariant violated: an idle owner holds score walks waiting on memory that nothing under way can free"
                );
            }
        }
    }

    /// Whether a request other than `own` holds state on the device's ledger
    /// and is not parked on an unexpired memory wait: one in a flight, in
    /// the current round, or outside it and still able to run, drain, or
    /// retire to advance `self.epoch` and free memory.
    fn progressing_state_beside(&self, own: &BTreeSet<RequestId>) -> bool {
        let mut excluded = own.clone();
        for (&id, request) in &self.requests {
            if matches!(
                request.state.wait(),
                Some(Wait { at, on: WaitsOn::Memory }) if !self.epoch.changed_since(at)
            ) {
                excluded.insert(id);
            }
        }
        self.domain.holds_state_beside(&excluded)
    }

    /// Whether a request outside `group_` holds memory whose release would
    /// let it in: one holding state that is still progressing, between
    /// rounds, in this round or in a flight
    /// ([`Self::progressing_state_beside`]). A claim short beside one waits
    /// for it to end; with none, nothing of ours can free memory for it, and
    /// it fails.
    fn others_stand(&self, group_: &OperationGroup) -> bool {
        let own = group_.operations().iter().map(Operation::request).collect::<BTreeSet<_>>();
        self.progressing_state_beside(&own)
    }

    /// Mark `set`, whose state was just released, evicted: each waits
    /// non-resident, a preemption owed.
    fn evicted(&mut self, set: &[RequestId]) -> Result<(), DomainError> {
        for &id in set {
            let Request { mut record, state } = self.requests.remove(&id).expect("an evicted request");
            record.preemption_debt = record.preemption_debt.saturating_add(1);
            // A round parked on memory is discarded with its state: the
            // request replays from its committed history.
            let generation = match state {
                State::Between(mut generation, _) => {
                    generation.evicted().map_err(release_invariant("preemption"))?;
                    generation
                }
                State::Parked(started, ..) => started.evicted().map_err(release_invariant("preemption"))?,
            };
            self.requests.insert(
                id,
                Request {
                    record,
                    state: State::Between(generation, Stage::NonResident(None)),
                },
            );
        }
        self.epoch.advance();
        Ok(())
    }
}

/// Score walks: the owner's scoring flights, taken in turns with its
/// rounds, their sequences opened, provisioned and reconciled by the owner's
/// own paths, so memory has one owner.
impl<F: ProgramFamily> Service<F> {
    /// What the idle pipeline takes next: a generation round or a scoring
    /// flight, never both in one launch. Where both have work they take
    /// turns, a round (whatever flights it spans) and a scoring flight each;
    /// where generation has none, scoring takes the flight whatever the
    /// turn (a second look finds what selection's own changes let in).
    fn next_flight(&mut self, bindings: &mut StateBindings) -> Result<Next, Fatal> {
        if self.scoring.as_ref().is_some_and(|walks| walks.turn == Turn::Scoring) {
            if let Some(group_) = self.scoring_group(bindings).map_err(|error| (None, error))? {
                return Ok(Next::Scoring(group_));
            }
        }
        let selected = self.select(bindings)?;
        match (&selected, &mut self.scoring) {
            (Selected::Round(_), Some(walks)) => walks.turn = Turn::Scoring,
            (Selected::Empty, Some(_)) => {
                if let Some(group_) = self.scoring_group(bindings).map_err(|error| (None, error))? {
                    return Ok(Next::Scoring(group_));
                }
            }
            _ => {}
        }
        Ok(Next::Generation(selected))
    }

    /// The next scoring flight's group, provisioned: the waiting walks that
    /// may open opened, then a forward of every open walk. None where
    /// scoring takes no flight now: no walk is open, a prompt chain is under
    /// way (any other group's provisioning would conclude it), or the walks
    /// wait on memory. Short of memory, the newest walk is deferred while
    /// others remain, as a round sheds its trailing operations, and no
    /// request is preempted for them: scoring never takes generation's
    /// state for more of its walks. A walk alone may preempt the requests
    /// that can be evicted, as a round's provision does (beside a request
    /// parked on memory the walk holds, neither could otherwise go on);
    /// still short, it waits for a change of ours that frees memory or,
    /// where none can come, fails with the shortage, and the walks waiting
    /// behind it open in its place.
    fn scoring_group(&mut self, bindings: &mut StateBindings) -> Result<Option<OperationGroup>, RequestError> {
        let epoch = self.epoch;
        if self.domain.chain_holder().is_some() || !self.scoring.as_mut().is_some_and(|walks| walks.waited(epoch)) {
            return Ok(None);
        }
        loop {
            // A wait set here holds back the walks still to open, not the
            // flight of those open; a close ends it.
            let epoch = self.epoch;
            if self.walks_mut().waited(epoch) {
                self.open_walks(bindings)?;
            }
            let walks = self.walks_mut();
            if walks.active.is_empty() {
                return Ok(None);
            }
            // Relief closes no walk (walks are no requests), so a flight that
            // sheds one still does once relieved.
            let (sheds, forwards) = (walks.sheds(), walks.forwards());
            let Ok([group_]) = <[OperationGroup; 1]>::try_from(group(&self.domain, forwards)) else {
                return Err(invariant("scoring flight")("a scoring flight's forwards group apart".into()));
            };
            let Relieved { current, pending } = self.fitted(bindings, None, &group_, !sheds)?;
            match current {
                Ok(()) => {
                    self.walks_mut().turn = Turn::Generation;
                    return Ok(Some(group_));
                }
                Err(DomainError::Capacity(capacity)) => {
                    if self.defer_newest_walk()? {
                        continue;
                    }
                    if pending || self.others_stand(&group_) {
                        let epoch = self.epoch;
                        self.walks_mut().waits = Some(epoch);
                        return Ok(None);
                    }
                    // Nothing of ours can make room for the walk alone: it
                    // fails.
                    self.fail_walks(refused_alone(DomainError::Capacity(capacity)))?;
                }
                // The domain refuses the flight: its walks fail.
                Err(error @ DomainError::Input(_)) => self.fail_walks(refused_alone(error))?,
                Err(error) => return Err(classify_domain_error(error)),
            }
        }
    }

    /// Open waiting walks, in arrival order, while fewer than a flight's
    /// walks are open and memory lets new work in: no real shortage holds it
    /// back, and no generation request waits on memory. An open short of
    /// memory a change of ours will free (another's end or eviction, a
    /// release under way) waits for it, its score back at the front of the
    /// queue; one nothing of ours can make room for fails with the shortage.
    fn open_walks(&mut self, bindings: &mut StateBindings) -> Result<(), RequestError> {
        // Scoring yields memory to generation: while a request waits on
        // memory (to open, for its pending work or for its parked round), no
        // walk opens, so the room an end frees is the request's first and
        // scores arriving all the while never starve it. The walks open keep
        // flying, and their ends free the room it waits for.
        let generation_waits = self.requests.values().any(|request| matches!(request.state.wait(), Some(Wait { on: WaitsOn::Memory, .. })));
        loop {
            // A real shortage holds new work back while the bank sits at its
            // floor (`MemoryResponse::admission_waits`).
            if self.memory.admission_waits || generation_waits {
                return Ok(());
            }
            let walks = self.walks_mut();
            if walks.active.len() >= walks.shape.sequences || walks.waits.is_some() {
                return Ok(());
            }
            let Some(request) = walks.waiting.pop_front() else {
                return Ok(());
            };
            let sequence = RequestId::admitted();
            match self.open_walk(bindings, sequence, request.tokens.len()) {
                Ok(Opened::Open) => self.walks_mut().active.push(Walk::new(request, sequence)),
                Ok(Opened::Waits) => {
                    let epoch = self.epoch;
                    let walks = self.walks_mut();
                    walks.waiting.push_front(request);
                    walks.waits = Some(epoch);
                    return Ok(());
                }
                Ok(Opened::Refused(error)) => (request.done)(Err(error)),
                Err(error) => {
                    (request.done)(Err(error.clone()));
                    return Err(error);
                }
            }
        }
    }

    /// Open `sequence` for a walk of `rows` tokens, at position zero with no
    /// prompt, reserving the rows it writes as a request's open reserves its
    /// prompt (`ExecutorDomain::open_state`). Short of memory, it takes the
    /// cache alone, as a request's open does. A walk not opened leaves
    /// nothing of its sequence.
    fn open_walk(&mut self, bindings: &mut StateBindings, sequence: RequestId, rows: usize) -> Result<Opened, RequestError> {
        self.domain
            .install_input(sequence, PreparedModelInput::continuation_only())
            .map_err(invariant("score walk"))?;
        let rows = rows as u64;
        let open = |this: &mut Self, bindings: &mut StateBindings| {
            // Scoring takes no turn while a chain is under way.
            let unchained = this
                .domain
                .unchained(bindings)
                .ok_or_else(|| invariant("score walk")("a walk opens beside a prompt chain".into()))?;
            Ok(this.domain.open_state(unchained, sequence, None, rows))
        };
        let first = open(self, bindings)?;
        let relieved = self.relieve(bindings, first, false, open)?;
        let opened = match relieved.current {
            Ok(encodes) if encodes.is_empty() => return Ok(Opened::Open),
            Ok(_) => Err(invariant("score walk")("a walk opened with images to encode".into())),
            Err(DomainError::Capacity(_))
                if relieved.pending || self.domain.holds_state_beside(&BTreeSet::from([sequence])) =>
            {
                Ok(Opened::Waits)
            }
            Err(error) if is_fatal(&error) => Err(classify_domain_error(error)),
            Err(error) => Ok(Opened::Refused(refused_alone(error))),
        };
        self.domain.close(sequence).map_err(invariant("score walk"))?;
        opened
    }

    /// Submit a provisioned scoring flight. A refusal of its input fails its
    /// walks; any other failure stops execution.
    fn submit_scoring(&mut self, group_: OperationGroup, bindings: StateBindings) -> Result<Pipeline<F>, RequestError> {
        match submit_group(&mut self.domain, bindings, &group_, self.wakes.completion()) {
            Ok(flight) => Ok(Pipeline::Scoring(ScoringFlight { flight, operations: group_.into_operations() })),
            Err(SubmitFailure::Refused(DomainError::Input(error), bindings)) => {
                self.fail_walks(RequestError::Input(error))?;
                Ok(Pipeline::Idle(bindings))
            }
            Err(SubmitFailure::Refused(error, _) | SubmitFailure::Failed(error)) => Err(classify_domain_error(error)),
        }
    }

    /// Reconcile a completed scoring flight and take back the bindings:
    /// each walk's rows committed whole, the logits it reads read to the
    /// host now, before anything else is submitted, and reduced; a walk that
    /// scored every row closes its sequence and is answered. An outcome the
    /// expert bank interrupted is aborted, and its rows run again; one of a
    /// walk termination answered is aborted. Memory is read once, at the
    /// flight's boundary.
    fn reconcile_scoring(&mut self, scoring: ScoringFlight<F>) -> Result<Pipeline<F>, RequestError> {
        let flight_invariant = |detail: &str| invariant("scoring flight")(detail.into());
        let ScoringFlight { flight, operations } = scoring;
        let finished = finish(&mut self.domain, flight, false).map_err(classify_domain_error)?;
        let (FlightOutcomes::Target(pending), TargetContinuation::Ready(mut bindings)) = finished else {
            return Err(flight_invariant("a scoring flight finished other than as a target launch"));
        };
        let forwarded = operations
            .iter()
            .filter_map(|operation| match operation {
                Operation::Forward { request, position, tokens, .. } => Some((*request, *position..*position + tokens.len())),
                Operation::Head { .. } | Operation::Encode { .. } => None,
            })
            .collect::<BTreeMap<_, _>>();
        let actual = pending.iter().map(PendingOperationOutcome::request).collect::<BTreeSet<_>>();
        if forwarded.len() != operations.len() || pending.len() != operations.len() || !actual.iter().eq(forwarded.keys()) {
            return Err(flight_invariant("executor outcomes do not match the flight's walks"));
        }
        let vocabulary = self.walks_mut().bounds.vocabulary;
        let mut ended = BTreeMap::new();
        for item in pending {
            let sequence = item.request();
            let walk = self.scoring.as_mut().and_then(|walks| walks.active.iter_mut().find(|walk| walk.sequence() == sequence));
            let Some(walk) = walk.filter(|_| !matches!(item.outcome(), Outcome::Interrupted { .. })) else {
                // Termination answered its walk, whose sequence closes at the
                // idle boundary; or the expert bank interrupted it, its inputs
                // whole, and its walk forwards the same rows again.
                self.domain.abort(item).map_err(classify_domain_error)?;
                continue;
            };
            let outcome = item.outcome().clone();
            let accepted_rows = item.rows();
            let absorbed = match self.domain.reconcile(item, PhysicalDecision::rows(accepted_rows)) {
                Ok(()) => {
                    let Outcome::Forward { rows, .. } = outcome else {
                        return Err(flight_invariant("a walk's outcome is not a forward's"));
                    };
                    let rows = host_rows(walk, forwarded[&sequence].start, rows, vocabulary)?;
                    walk.absorb(forwarded[&sequence].clone(), rows).map_err(|detail| flight_invariant(&detail))?
                }
                Err(error) if is_fatal(&error) => return Err(classify_domain_error(error)),
                // Its sequence fails alone, its state resolved (aborted or
                // reinserted), so it closes.
                Err(error) => Absorbed::Ended(Err(classify_domain_error(error))),
            };
            if let Absorbed::Ended(result) = absorbed {
                ended.insert(sequence, result);
            }
        }
        let walks = self.walks_mut();
        let (mut closing, mut open) = (Vec::new(), Vec::new());
        for walk in std::mem::take(&mut walks.active) {
            match ended.remove(&walk.sequence()) {
                Some(result) => closing.push((walk, result)),
                None => open.push(walk),
            }
        }
        walks.active = open;
        self.end_walks(closing)?;
        // The flight's boundary: what it held is free, and memory is read
        // once here.
        self.epoch.advance();
        self.step(Some(&mut bindings), true)?;
        Ok(Pipeline::Idle(bindings))
    }

    /// Defer the newest open walk where a shortage sheds one
    /// ([`Walks::defer_newest`]), its sequence closed: no walk opens again
    /// until a change of ours past its close. Whether one was: a walk alone
    /// never is.
    fn defer_newest_walk(&mut self) -> Result<bool, RequestError> {
        let Some(sequence) = self.walks_mut().defer_newest() else {
            return Ok(false);
        };
        self.close_walk(sequence)?;
        let epoch = self.epoch;
        self.walks_mut().waits = Some(epoch);
        Ok(true)
    }

    /// End every open walk with `error`: a flight of them refused whole.
    fn fail_walks(&mut self, error: RequestError) -> Result<(), RequestError> {
        let failed = std::mem::take(&mut self.walks_mut().active);
        self.end_walks(failed.into_iter().map(|walk| (walk, Err(error.clone()))).collect())
    }

    /// Close each ended walk's sequence and answer it: every walk is
    /// answered before a failed close stops the owner.
    fn end_walks(&mut self, ended: Vec<(Walk, Result<ScoreOutcome, RequestError>)>) -> Result<(), RequestError> {
        let mut fault = None;
        for (walk, result) in ended {
            let result = match self.close_walk(walk.sequence()) {
                Ok(()) => result,
                Err(error) => Err(fault.get_or_insert(error).clone()),
            };
            walk.answer(result);
        }
        fault.map_or(Ok(()), Err)
    }

    /// Close the sequences of walks termination answered: at an idle
    /// boundary no flight holds them.
    fn close_unclosed(&mut self) -> Result<(), RequestError> {
        let Some(walks) = &mut self.scoring else {
            return Ok(());
        };
        std::mem::take(&mut walks.unclosed).into_iter().try_for_each(|sequence| self.close_walk(sequence))
    }

    /// Close a walk's sequence: what it held is free, and what waits on
    /// memory retries.
    fn close_walk(&mut self, sequence: RequestId) -> Result<(), RequestError> {
        self.domain.close(sequence).map_err(invariant("score walk"))?;
        self.epoch.advance();
        Ok(())
    }

    /// The load's walks: only a load that scores has scoring work.
    fn walks_mut(&mut self) -> &mut Walks {
        self.scoring.as_mut().expect("invariant violated: score walks of a load that scores none")
    }
}

/// The rows of `walk` a flight forwarded from `first` on, on the host: the
/// logits of each row the walk reads ([`Walk::reads`]). A read that fails,
/// or a row other than the vocabulary's width, stops execution.
fn host_rows(walk: &Walk, first: usize, rows: Vec<RowResult>, vocabulary: usize) -> Result<Vec<Option<Vec<f32>>>, RequestError> {
    rows.into_iter()
        .zip(first..)
        .map(|(row, position)| match row.logits {
            Some(logits) if walk.reads(position) => {
                let read = logits
                    .read_to_host()
                    .map_err(|error| RequestError::Device(DeviceError::Transfer(format!("reading a logits row: {error}"))))?;
                if read.len() != vocabulary {
                    return Err(invariant("scoring flight")(format!(
                        "a logits row has {} entries for a vocabulary of {vocabulary}",
                        read.len()
                    )));
                }
                Ok(Some(read))
            }
            _ => Ok(None),
        })
        .collect()
}

impl<F: ProgramFamily> MemoryHost for Service<F> {
    type Family = F;

    fn domain(&mut self) -> &mut ExecutorDomain<F> {
        &mut self.domain
    }

    fn memory(&mut self) -> &mut BetweenFlights {
        &mut self.memory
    }

    fn now(&self) -> u64 {
        self.now
    }

    fn completion(&self) -> magnitude_executor::CompletionWake {
        self.wakes.completion()
    }

    fn advance_epoch(&mut self) {
        self.epoch.advance();
    }

    fn unload(&mut self) {
        tracing::error!(requests = self.requests.len(), "unloading the model: a real memory shortage persists with nothing of ours pending");
        self.unloaded = true;
        self.terminalize_all(
            RequestError::ModelUnloaded {
                cause: ModelUnloadCause::MemoryPressure,
            },
            None,
        );
    }

    fn evict_owed_prefixes(&mut self) -> bool {
        let held = self.held_entries(None);
        let mut evicted = false;
        while self.domain.prefix_owed() > 0 && self.prefix_cache.evict_one(&held) {
            self.domain.settle_state();
            evicted = true;
        }
        evicted
    }

    fn prefix_value(&self) -> Option<(u64, u64)> {
        self.prefix_cache.least_value()
    }

    /// Those that can be evicted now ([`Self::can_evict`]); a chain's
    /// holder between its chunks is kept: its eviction would conclude the
    /// chain (`ExecutorDomain::release_state`), dropping the chunks it has
    /// already launched. A score walk's sequence is never one: walks are
    /// not requests, and only their own deferral closes them early.
    fn evictable(&self) -> BTreeSet<RequestId> {
        let holder = self.domain.chain_holder();
        self.requests.iter().filter(|(id, request)| Some(**id) != holder && Self::can_evict(request)).map(|(id, _)| *id).collect()
    }

    fn preempt(&mut self, sequences: &[RequestId]) -> Result<(), RequestError> {
        let evictable = self.evictable();
        assert!(sequences.iter().all(|id| evictable.contains(id)), "invariant violated: a relief names a request that cannot be evicted: {sequences:?}");
        self.evict(sequences).map_err(classify_domain_error)
    }
}

enum Selected {
    Round(Round),
    /// Selection started no work but changed some request's state.
    Retry,
    Empty,
}

/// What the idle pipeline takes next ([`Service::next_flight`]): what
/// generation's selection came to, or a scoring flight's group, provisioned.
enum Next {
    Generation(Selected),
    Scoring(OperationGroup),
}

/// What opening a walk's sequence came to ([`Service::open_walk`]).
enum Opened {
    Open,
    /// Short of memory a change of ours will free: it opens once one comes.
    Waits,
    /// Nothing of ours can make room for it, or the domain refuses it: it
    /// fails with this.
    Refused(RequestError),
}

/// A disk entry a request opens from ([`Service::cold_restore`]): the saved
/// state, read and rebuilt, with what retaining it in the memory cache
/// takes.
struct ColdRestore {
    key: EntryKey,
    /// The entry's bytes on disk.
    bytes: u64,
    target: ExportedState,
    head: Option<ExportedState>,
    method: MethodCheckpoint,
    point: CachePoint,
    /// When its read began.
    started: Instant,
}

enum Residency {
    Resident(Generation, Vec<Operation>),
    Waiting(Generation, Wait),
    Failed(Generation),
}

/// What relieving a claim short of memory came to ([`relieve`]): its last
/// try, and whether a change of ours under way will free memory for it.
pub struct Relieved<T> {
    pub current: Result<T, DomainError>,
    pub pending: bool,
}

/// Combine prefix eviction with the domain's `release_yielded` outcome:
/// `Yielded::Pending` is preserved unconditionally so an async pool or slot
/// release is never overwritten with `Yielded::Released`.
fn combine_yielded(evicted_prefixes: bool, yielded: Yielded) -> Yielded {
    match (evicted_prefixes, yielded) {
        (true, Yielded::Nothing) => Yielded::Released { released: 0, returned: 0 },
        (_, yielded) => yielded,
    }
}

#[cfg(test)]
mod branch_plan_tests {
    use super::{
        MIN_BRANCH_GAIN, MIN_PREFIX_HIT, awaited_prefix, peer_stop, plan_branch, plan_declared,
    };
    use std::collections::BTreeSet;

    /// Chat requests sharing a system prompt diverge a few rows into the
    /// last user message, fewer than the gain past the peer's declared cache
    /// point at the system prompt's end, so the divergence is never planned
    /// on the peer. The new request still waits for the peer's declared
    /// point instead of prefilling the whole prompt alongside it.
    #[test]
    fn a_request_waits_for_a_peer_stop_just_below_their_divergence() {
        let (declared, shared, prompt) = (14864, 14891, 14908);
        let mut peer = BTreeSet::from([declared, prompt - 6]);
        plan_branch(&mut peer, shared, 0, prompt);
        assert!(!peer.contains(&shared));
        assert_eq!(peer_stop(&peer, shared), Some(declared));
        assert_eq!(awaited_prefix(declared, shared, 0, prompt), Some(declared));
        // The divergence itself, when planned, is awaited as before.
        assert_eq!(awaited_prefix(shared, shared, 0, prompt), Some(shared));
        // Not a stop too far below the divergence, one not worth resuming
        // from (an entry already ends at or near it), or none at all.
        assert_eq!(awaited_prefix(shared - MIN_BRANCH_GAIN, shared, 0, prompt), None);
        assert_eq!(awaited_prefix(declared, shared, declared, prompt), None);
        assert_eq!(awaited_prefix(declared, shared, declared - 1, prompt), None);
        assert_eq!(awaited_prefix(0, shared, 0, prompt), None);
        assert_eq!(peer_stop(&BTreeSet::from([prompt - 6]), shared), None);
    }

    /// cc237 round 2: a conversation that shares the system prompt and the
    /// opening of the user message is cached, but its entry at the system
    /// prompt's end is gone, so nothing on the new requests' path ends below
    /// their divergence (they resume at 0). The cached path still shares
    /// rows up to just past the peer's stop. Those rows are no resume point,
    /// so the new request waits for the peer instead of prefilling the whole
    /// prompt beside it, as all eight requests did.
    #[test]
    fn a_cached_path_without_an_entry_in_the_prefix_never_replaces_the_peer() {
        let (declared, shared, prompt) = (14869, 14894, 14917);
        let cached_path = shared;
        let resumed = 0;
        let deepest = cached_path.max(shared);
        assert_eq!(awaited_prefix(declared, deepest, resumed, prompt), Some(declared));
    }

    /// A short turn of a conversation: it resumes where the previous turn
    /// declared the next one leaves the prompt, and declares the opening of
    /// its last message and where its own next turn leaves the prompt, all
    /// within the gain. Both are planned, so the next turn resumes beyond
    /// this one instead of where this one resumed.
    #[test]
    fn a_short_turn_plans_its_declared_points() {
        let (resumed, prompt) = (459, 490);
        let mut branches = BTreeSet::new();
        for point in [466, 483] {
            assert!(point < resumed + MIN_BRANCH_GAIN);
            plan_declared(&mut branches, point, resumed, prompt);
        }
        assert_eq!(branches, BTreeSet::from([466, 483]));
        // Never at or below the resume point, below a hit, or past the prompt.
        let mut branches = BTreeSet::new();
        for point in [resumed - 1, resumed, prompt] {
            plan_declared(&mut branches, point, resumed, prompt);
        }
        plan_declared(&mut branches, MIN_PREFIX_HIT - 1, 0, prompt);
        assert!(branches.is_empty());
    }

    #[test]
    fn a_branch_point_saves_enough_over_the_resume_point_and_lower_branches() {
        let prompt = 6000;
        let mut branches = BTreeSet::new();
        // Below a hit, at or past the prompt's end, or too near the resume
        // point: not worth a split chunk and a retained state.
        plan_branch(&mut branches, MIN_PREFIX_HIT - 1, 0, prompt);
        plan_branch(&mut branches, prompt, 0, prompt);
        plan_branch(&mut branches, 1000 + MIN_BRANCH_GAIN - 1, 1000, prompt);
        assert!(branches.is_empty());
        // A branch point, then a divergence just beyond it and one far
        // beyond it.
        plan_branch(&mut branches, 5800, 0, prompt);
        plan_branch(&mut branches, 5800 + MIN_BRANCH_GAIN - 1, 0, prompt);
        plan_branch(&mut branches, 5800 + MIN_BRANCH_GAIN, 0, prompt);
        // A shallower point is planned below the deeper ones.
        plan_branch(&mut branches, 3000, 0, prompt);
        assert_eq!(
            branches,
            BTreeSet::from([3000, 5800, 5800 + MIN_BRANCH_GAIN])
        );
    }
}

#[cfg(test)]
mod termination_tests {
    use super::Membership;
    use magnitude_generation::{
        EndOfGeneration, FinishReason, Generation, InputLayout, MethodChoice, Options, RoundStart,
        Sampling, Shaping, StartedRound, TokenId,
    };
    use std::collections::BTreeSet;

    pub(super) fn generation() -> Generation {
        generation_with_limit(8)
    }

    /// A test generation's options: at most `max_tokens`, by `method`.
    pub(super) fn options(max_tokens: usize, method: MethodChoice) -> Options {
        Options {
            max_tokens,
            output_capacity: 4,
            context_limit: 32,
            vocabulary: 100,
            stop_tokens: BTreeSet::from([TokenId(99)]),
            suppressed_tokens: BTreeSet::new(),
            sampling: Sampling::Greedy,
            shaping: Shaping {
                temperature: 0.0,
                ..Default::default()
            },
            seed: 42,
            forced_quantum: 4,
            method,
            end_of_generation: EndOfGeneration::Stop,
            reasoning_budget: None,
        }
    }

    pub(super) fn generation_with_limit(max_tokens: usize) -> Generation {
        let mut generation = Generation::new(
            vec![TokenId(1), TokenId(2)],
            InputLayout::new(2, vec![]).unwrap(),
            options(max_tokens, MethodChoice::Plain),
            None,
        )
        .unwrap();
        generation.resume_at(None).unwrap();
        generation
    }

    fn started() -> StartedRound {
        match generation().start_round(magnitude_generation::RequestId(1), 4, magnitude_executor::FrontGrant::UpTo(0)) {
            Ok(RoundStart::Target(round)) => round,
            _ => panic!("expected a target round"),
        }
    }

    #[test]
    fn a_failed_started_round_finishes_failed_not_cancelled() {
        let failed = Membership::Started(started()).terminate(false);
        assert_eq!(failed.finish_reason(), Some(FinishReason::Failed));
        let cancelled = Membership::Started(started()).terminate(true);
        assert_eq!(cancelled.finish_reason(), Some(FinishReason::Cancelled));
    }

    /// An interrupted round (`RoundError::Interrupted`) is abandoned: its
    /// generation stays unfinished and starts the same round again.
    #[test]
    fn an_abandoned_round_runs_again() {
        let round = started();
        let forward = round.round_forward().clone();
        let generation = round.abandon();
        assert_eq!(generation.finish_reason(), None, "an abandoned round finishes its generation");
        match generation.start_round(magnitude_generation::RequestId(1), 4, magnitude_executor::FrontGrant::UpTo(0)) {
            Ok(RoundStart::Target(again)) => assert_eq!(again.round_forward(), &forward, "the round starts again as it was"),
            _ => panic!("expected the target round again"),
        }
    }

    #[test]
    fn a_failed_waiting_request_finishes_failed() {
        let failed = Membership::Generation(generation()).terminate(false);
        assert_eq!(failed.finish_reason(), Some(FinishReason::Failed));
    }
}

#[cfg(test)]
mod timing_tests {
    use super::*;

    /// A lone prefill group queues its next chunk only when nothing outside
    /// the round competes for the device: another live request's work would
    /// orphan the queued chunk, discarding a whole chunk of device work.
    #[test]
    fn only_live_work_outside_the_round_competes_with_a_queued_chunk() {
        let record = || Record {
            prefix: Some(PrefixParticipation::new(BTreeSet::new())),
            ..super::front_rows_tests::record()
        };
        let request = |stage| Request {
            record: record(),
            state: State::Between(super::termination_tests::generation(), stage),
        };
        assert!(competes(&request(Stage::NonResident(None)), Status::Runnable));
        assert!(competes(&request(Stage::Ready), Status::Runnable));
        assert!(!competes(
            &request(Stage::AwaitingPrefix { producer: RequestId(99), through: 1 }),
            Status::Runnable
        ));
        assert!(!competes(
            &request(Stage::Ready),
            Status::Terminal(FinishReason::Length)
        ));
    }

    /// An admitted peer that has not yet become resident still computes the
    /// shared prefix in its first round, so a request admitted right after it
    /// waits for it instead of prefilling the prefix beside it. A peer the
    /// cache resumes at or beyond the prefix never computes it (a
    /// conversation's next turn resuming from its own deeper entry), so no
    /// request waits on it. A peer awaiting a prefix itself never counts, so
    /// two waiters cannot hold each other.
    #[test]
    fn an_admitted_peer_not_yet_resident_computes_its_prefix() {
        let generation = super::termination_tests::generation;
        let admitted = || State::Between(generation(), Stage::NonResident(None));
        assert!(computes_toward(&admitted(), 1, |_| 0));
        assert!(!computes_toward(&admitted(), 1, |_| 1));
        assert!(!computes_toward(&admitted(), 1, |_| 2));
        assert!(!computes_toward(
            &State::Between(generation(), Stage::AwaitingPrefix { producer: RequestId(99), through: 1 }),
            1,
            |_| 0
        ));
        let cancelled = terminated(generation(), true);
        assert!(!computes_toward(&State::Between(cancelled, Stage::NonResident(None)), 1, |_| 0));
    }

    /// The prompt runs from the request's first service to its first
    /// accepted token, the generation from there to its latest: later
    /// service (a round resumed, a recovery) restarts neither.
    #[test]
    fn the_clock_times_the_prompt_to_the_first_token_and_generation_to_the_latest() {
        let start = Instant::now();
        let at = |ms| start + Duration::from_millis(ms);
        let mut clock = RequestClock::Waiting;
        clock.serve(at(0));
        clock.serve(at(5));
        clock.accepted(at(100));
        clock.serve(at(150));
        clock.accepted(at(160));
        clock.accepted(at(400));
        assert_eq!(
            clock.timings(at(1000)),
            RequestTimings { prompt_ns: 100_000_000, predicted_ns: 300_000_000 }
        );
    }

    /// A request that ends before any output spent its whole service on
    /// its prompt; one never served spent nothing.
    #[test]
    fn a_request_without_output_times_only_its_prompt() {
        let start = Instant::now();
        let mut clock = RequestClock::Waiting;
        assert_eq!(clock.timings(start), RequestTimings::default());
        clock.serve(start);
        assert_eq!(
            clock.timings(start + Duration::from_millis(7)),
            RequestTimings { prompt_ns: 7_000_000, predicted_ns: 0 }
        );
    }

    #[test]
    #[should_panic(expected = "never served")]
    fn output_for_a_request_never_served_is_an_invariant_violation() {
        RequestClock::Waiting.accepted(Instant::now());
    }
}

#[cfg(test)]
mod pending_retention_tests {
    use super::*;
    use magnitude_executor::{DraftForm, FeatureReader, FeatureRows, FeatureSpan, HeadPhase};

    struct NoFeatures;
    impl FeatureReader for NoFeatures {
        fn read(&mut self, _: &FeatureSpan) -> Result<FeatureRows, String> {
            panic!("plain logical fixture needs no features")
        }
    }

    fn priming() -> Work {
        let operation = Operation::Head {
            request: RequestId(1),
            phase: HeadPhase::Priming { draft_from: 3 },
            tokens: vec![magnitude_generation::TokenId(3)],
            conditioning: FeatureRows::new(vec![0u8; 4].into(), 1).unwrap(),
            position: 2,
            proposals: Vec::new(),
            form: DraftForm::Chained,
        };
        operation.validate().unwrap();
        Work::new(vec![operation]).unwrap()
    }

    #[test]
    fn terminal_replay_with_parked_priming_cannot_retain_or_retire() {
        let generation = super::termination_tests::generation_with_limit(1);
        let Ok(RoundStart::Target(round)) = generation.start_round(RequestId(1), 4, FrontGrant::UpTo(0)) else {
            panic!("prompt round")
        };
        let transition = round.prepare_round_transition(RequestId(1), &[magnitude_generation::TokenId(3)], Vec::new(), None, &mut NoFeatures).unwrap();
        let (mut generation, _) = round.commit(transition);
        generation.discard_output();
        assert_eq!(generation.finish_reason(), Some(FinishReason::Length));
        let round = generation.start_reconciliation(4).unwrap_or_else(|(_, e)| panic!("{e}"));
        let work = lower_round(&round, RequestId(1), round.generation().resident_position(), None).unwrap();
        let parked = State::Parked(round, Work::new(vec![work]).unwrap(), Some(Wait { at: AvailabilityEpoch::default(), on: WaitsOn::Memory }));
        assert!(parked.settled().is_none());
        let State::Parked(round, _, _) = parked else { unreachable!() };
        let transition = round.prepare_round_transition(RequestId(1), &[], Vec::new(), None, &mut NoFeatures).unwrap();
        let (generation, _) = round.commit(transition);
        // These are exactly the old publication predicates: all are satisfied
        // before the priming continuation has become eligible to run.
        assert_eq!(generation.finish_reason(), Some(FinishReason::Length));
        assert_eq!(generation.output_len(), 0);
        assert!(generation.pending_reconciliation().is_none());
        assert_eq!(generation.resident_position(), generation.accepted_position());
        let pending = State::Between(generation, Stage::Pending(priming(), Some(Wait { at: AvailabilityEpoch::default(), on: WaitsOn::Memory })));
        assert!(pending.settled().is_none(), "pending priming must block retention and terminal publication");
        let State::Between(generation, _) = pending else { unreachable!() };
        // finish_round returns completed method work as Ready.
        let completed = State::Between(generation, Stage::Ready);
        assert_eq!(completed.settled().unwrap().finish_reason(), Some(FinishReason::Length));
    }

    #[test]
    fn abandoned_pending_work_can_still_finish_cancelled_or_failed() {
        for (cancel, expected) in [(true, FinishReason::Cancelled), (false, FinishReason::Failed)] {
            let pending = State::Between(
                super::termination_tests::generation(),
                Stage::Pending(priming(), Some(Wait { at: AvailabilityEpoch::default(), on: WaitsOn::Memory })),
            );
            assert!(pending.settled().is_none());
            // terminate_request discards composed work
            // and normalize the terminated generation to Ready.
            let terminal = State::Between(pending.terminate(cancel), Stage::Ready);
            assert_eq!(terminal.settled().unwrap().finish_reason(), Some(expected));
        }
    }

    #[test]
    fn finished_without_residency_does_not_require_model_work_to_publish() {
        let mut generation = super::termination_tests::generation_with_limit(0);
        generation.evicted().unwrap();
        assert!(!generation.is_resident());
        assert_eq!(generation.finish_reason(), Some(FinishReason::Length));
        let state = State::Between(generation, Stage::NonResident(Some(Wait { at: AvailabilityEpoch::default(), on: WaitsOn::Memory })));
        assert!(state.settled().is_some());
        let State::Between(generation, _) = state else { unreachable!() };
        let state = State::Between(generation, Stage::AwaitingPrefix { producer: RequestId(99), through: 1 });
        assert!(state.settled().is_some());
    }
}

/// The owner over the controlled domain (`magnitude_executor::fixtures`):
/// opens refused for memory, relief's evictions, and request identities
/// across loads.
#[cfg(test)]
mod claim_tests {
    use super::*;
    use crate::publication::Publication;
    use magnitude_executor::fixtures;
    use magnitude_generation::{EndOfGeneration, InputLayout, MethodChoice, Options, Sampling, Shaping, TokenId};

    pub(super) struct NoWakes;
    impl Wakes for NoWakes {
        fn completion(&self) -> magnitude_executor::CompletionWake {
            magnitude_executor::CompletionWake::new(|| {})
        }
        fn publication(&self, _: RequestId, _: PublicationWake) {}
        fn pressure(&self) -> Box<dyn Fn() + Send + Sync> {
            Box::new(|| {})
        }
    }

    /// Rounds of two rows, as the fixture's launches carry.
    const ROWS: usize = 2;

    pub(super) fn limits() -> ServiceLimits {
        ServiceLimits {
            prefill_tokens: ROWS,
            decode_tokens: ROWS,
            decode_share: 0.5,
            locality_seconds: 1.0,
            conversations: std::num::NonZeroUsize::MIN,
        }
    }

    /// A fresh generation of four prompt tokens within the fixture's
    /// eight-token context.
    fn fresh(vocabulary: usize) -> Generation {
        fresh_by(vocabulary, MethodChoice::Plain)
    }

    /// What generates by `method`: plain, or MTP by its own method.
    pub(super) fn method_of(method: MethodChoice) -> Arc<dyn magnitude_generation::Method> {
        match method {
            MethodChoice::Plain => Arc::new(magnitude_generation::Plain),
            MethodChoice::Mtp { proposals } => Arc::new(magnitude_generation::Mtp::new("fixture", proposals.into()).unwrap()),
            other => unreachable!("the owner's tests generate by no {other:?}"),
        }
    }

    /// [`fresh`], generating by `method` ([`method_of`]).
    fn fresh_by(vocabulary: usize, method: MethodChoice) -> Generation {
        let factory = method_of(method);
        Generation::new_with_method(
            vec![TokenId(1), TokenId(2), TokenId(3), TokenId(4)],
            InputLayout::new(4, vec![]).unwrap(),
            Options {
                max_tokens: 2,
                output_capacity: 4,
                context_limit: 8,
                vocabulary,
                stop_tokens: BTreeSet::new(),
                suppressed_tokens: BTreeSet::new(),
                sampling: Sampling::Greedy,
                shaping: Shaping { temperature: 0.0, ..Default::default() },
                seed: 0,
                forced_quantum: 0,
                method,
                end_of_generation: EndOfGeneration::Stop,
                reasoning_budget: None,
            },
            None,
            factory,
        )
        .unwrap()
    }

    /// Make `id` resident between rounds, as selection does: whether it is
    /// (`Some(true)`) or waits (`Some(false)`); none where it failed.
    fn residency(service: &mut Service<fixtures::TestFamily>, bindings: &mut StateBindings, id: RequestId) -> Option<bool> {
        let Request { mut record, state } = service.requests.remove(&id).expect("an admitted request");
        let State::Between(generation, Stage::NonResident(None)) = state else { panic!("{id:?} is not waiting to be resident") };
let Ok(residency) = service.make_resident(id, &mut record, generation, bindings) else { panic!("{id:?}'s open is fatal") };
        let (state, resident) = match residency {
            Residency::Resident(generation, _) => (State::Between(generation, Stage::Ready), Some(true)),
            Residency::Waiting(generation, _) => (State::Between(generation, Stage::NonResident(None)), Some(false)),
            Residency::Failed(generation) => (State::Between(generation, Stage::NonResident(None)), None),
        };
        service.requests.insert(id, Request { record, state });
        resident
    }

    /// Take every byte the device's ledger would give new work now (its
    /// free room and the cache worth nothing) as a measurement's room.
    pub(super) fn take_room<F: ProgramFamily>(service: &mut Service<F>) {
        leave_room(service, 0);
    }

    /// Leave `keep` bytes of what the device's ledger would give new work
    /// now, the rest taken as a measurement's room.
    pub(super) fn leave_room<F: ProgramFamily>(service: &mut Service<F>, keep: u64) {
        const PAST: u64 = u64::MAX / 4;
        service.domain.release_measurement_room();
        let room = match service.domain.claim_measurement_room(PAST) {
            Err(magnitude_executor::memory::ledger::Short { over, .. }) => PAST - over,
            other => panic!("a claim past the budget is refused over it, not {other:?}"),
        };
        assert!(room >= keep, "the ledger gives {room} bytes, not {keep}");
        service.domain.claim_measurement_room(room - keep).expect("the room the ledger gives is held");
    }

    /// Item 1 of the request-memory redesign: requests with long prompts and
    /// no `max_tokens` (the chat layer's rest of the context) open holding
    /// their prompts and a launch's rows ahead, not their whole contexts.
    /// Three open in room that holds three such openings, where reserving
    /// each one's whole context would hold no third.
    #[test]
    fn long_prompts_without_max_tokens_open_side_by_side() {
        const CONTEXT: usize = 4096;
        const PROMPT: usize = 1024;
        let mut definition = fixtures::tiny_definition();
        definition.decoder.context_limit = CONTEXT as u64;
        let vocabulary = definition.decoder.vocabulary as usize;
        let (domain, bindings) =
            fixtures::fixture_of(definition, fixtures::limits(2, 1, false), None, false, magnitude_executor::platform::DeviceRequest::Automatic);
        let opening = fixtures::reserved_bytes(&domain, fixtures::opening_rows(&domain, PROMPT as u64));
        let whole = fixtures::reserved_bytes(&domain, fixtures::opening_rows(&domain, CONTEXT as u64));
        assert!(2 * whole > 3 * opening, "two whole contexts take more than three openings: {whole} and {opening} bytes");
        let mut owner = Owner::new(domain, bindings, limits(), Arc::new(NoWakes)).unwrap();
        let unbounded = || {
            let options = Options {
                max_tokens: CONTEXT - PROMPT + 1,
                output_capacity: 4,
                context_limit: CONTEXT,
                vocabulary,
                stop_tokens: BTreeSet::new(),
                suppressed_tokens: BTreeSet::new(),
                sampling: Sampling::Greedy,
                shaping: Shaping { temperature: 0.0, ..Default::default() },
                seed: 0,
                forced_quantum: 0,
                method: MethodChoice::Plain,
                end_of_generation: EndOfGeneration::Stop,
                reasoning_budget: None,
            };
            Generation::new(vec![TokenId(1); PROMPT], InputLayout::new(PROMPT, vec![]).unwrap(), options, None).unwrap()
        };
        let ids = (0..3)
            .map(|_| owner.admit(unbounded(), PreparedModelInput::continuation_only(), PrefixRetention::Transient, 4, 0).unwrap().0)
            .collect::<Vec<_>>();
        let Owner { service, pipeline: Pipeline::Idle(bindings) } = &mut owner else { panic!("an idle owner") };
        leave_room(service, 3 * opening);
        for id in ids {
            assert_eq!(residency(service, bindings, id), Some(true), "{id:?} opens beside the others");
        }
    }

    /// Admit `count` fresh requests in turn.
    fn admit(owner: &mut Owner<fixtures::TestFamily>, count: usize) -> Vec<RequestId> {
        let vocabulary = fixtures::tiny_definition().decoder.vocabulary as usize;
        (0..count)
            .map(|_| owner.admit(fresh(vocabulary), PreparedModelInput::continuation_only(), PrefixRetention::Transient, 4, 0).unwrap().0)
            .collect()
    }

    /// The first request draws the load's request slot; with the room
    /// beside it taken, a second request's open is refused for memory and
    /// waits while the first holds the slot: it holds no state, and no
    /// prefix entry is touched.
    #[test]
    fn an_open_refused_while_the_slot_is_lent_waits_and_touches_no_prefix() {
        let (domain, bindings) = fixtures::fixture(None);
        let mut owner = Owner::new(domain, bindings, limits(), Arc::new(NoWakes)).unwrap();
        let [first, second] = admit(&mut owner, 2)[..] else { unreachable!() };
        let Pipeline::Idle(bindings) = &mut owner.pipeline else { panic!("an idle owner") };
        let service = &mut owner.service;
        assert_eq!(residency(service, bindings, first), Some(true), "the first draws the slot");
        take_room(service);
        let entries = service.prefix_cache.len();
        assert_eq!(residency(service, bindings, second), Some(false), "the second waits while the slot is lent");
        assert!(matches!(service.requests[&second].state, State::Between(_, Stage::NonResident(None))), "it holds no state");
        assert_eq!(service.prefix_cache.len(), entries, "no prefix entry is touched");
    }

    /// Relief carried out for a claim short of memory evicts exactly the
    /// requests its plan names, the latest admitted first and just enough,
    /// never the first.
    #[test]
    fn relief_evicts_exactly_the_requests_its_plan_names() {
        let (domain, bindings) = fixtures::fixture(None);
        let mut owner = Owner::new(domain, bindings, limits(), Arc::new(NoWakes)).unwrap();
        let ids = admit(&mut owner, 3);
        let Pipeline::Idle(bindings) = &mut owner.pipeline else { panic!("an idle owner") };
        let service = &mut owner.service;
        for id in &ids {
            assert_eq!(residency(service, bindings, *id), Some(true), "{id:?} opens");
        }
        take_room(service);
        let resource = magnitude_executor::ResourceKind::DeviceMemory;
        let evictable = MemoryHost::evictable(service);
        let lacks = std::iter::successors(Some(1u64), |bytes| bytes.checked_mul(2))
            .find(|bytes| service.domain.plan_relief(resource, *bytes, Relief::CacheAndRequests(&evictable)).is_some_and(|plan| !plan.preempt().is_empty()))
            .expect("a need some request covers");
        let named = service.domain.plan_relief(resource, lacks, Relief::CacheAndRequests(&evictable)).expect("the plan").preempt().to_vec();
        assert_eq!(named, [ids[2]], "the latest admitted, just enough");
        let short = Err(DomainError::Capacity(magnitude_executor::CapacityError::new(resource, lacks, 0)));
        let relieved = service.relieve(bindings, short, true, |_, _| Ok(Ok(()))).unwrap();
        assert!(relieved.current.is_ok(), "the claim fits once relieved");
        let evicted = |service: &Service<fixtures::TestFamily>, id: RequestId| {
            let request = &service.requests[&id];
            matches!(request.state, State::Between(_, Stage::NonResident(None))) && request.record.preemption_debt == 1
        };
        assert!(evicted(service, ids[2]), "the named request is evicted");
        for id in &ids[..2] {
            assert!(!evicted(service, *id) && service.requests[id].record.preemption_debt == 0, "{id:?} stands");
        }
    }

    /// When a store is already over budget (`Short::over > Short::required`),
    /// `CapacityError::from_short` reports `available == 0` without `u64`
    /// underflow and `MemoryStep::relieve` plans relief for the full `over`
    /// deficit — neither skipping relief (`lacks == 0`) nor partially evicting
    /// for only `required` bytes when `over` cannot be covered.
    #[test]
    fn relief_uses_full_over_budget_shortfall_without_underflow_or_partial_eviction() {
        let (domain, bindings) = fixtures::fixture(None);
        let mut owner = Owner::new(domain, bindings, limits(), Arc::new(NoWakes)).unwrap();
        let ids = admit(&mut owner, 3);
        let Pipeline::Idle(bindings) = &mut owner.pipeline else { panic!("an idle owner") };
        let service = &mut owner.service;
        for id in &ids {
            assert_eq!(residency(service, bindings, *id), Some(true), "{id:?} opens");
        }
        let resource = magnitude_executor::ResourceKind::RequestState;
        let evictable = MemoryHost::evictable(service);
        // Find the exact cache-only ceiling so `one_req_need = cache_only + 1`
        // requires evicting `ids[2]`, regardless of any spare pool cache on the
        // host GPU.
        let mut lo = 0u64;
        let mut hi = 1u64;
        while service.domain.plan_relief(resource, hi, Relief::CacheOnly).is_some() {
            lo = hi;
            hi = hi.checked_mul(2).expect("cache ceiling fits u64");
        }
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if service.domain.plan_relief(resource, mid, Relief::CacheOnly).is_some() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let one_req_need = lo + 1;
        let plan = service
            .domain
            .plan_relief(resource, one_req_need, Relief::CacheAndRequests(&evictable))
            .expect("evicting a request covers one byte past cache-only");
        assert_eq!(plan.preempt(), [ids[2]], "one_req_need evicts ids[2]");
        let store = plan.store();
        let tiny_required = one_req_need.min(40_960).saturating_sub(1).max(1);
        assert!(one_req_need > tiny_required, "test requires over > required");

        // Case A: `over` exceeds `required` and is coverable by evicting `ids[2]`.
        let coverable = magnitude_executor::memory::ledger::Short {
            store: store.clone(),
            required: tiny_required,
            over: one_req_need,
        };
        let cap_a = magnitude_executor::CapacityError::from_short(resource, &coverable);
        assert_eq!(cap_a.available, 0);
        assert_eq!(cap_a.lacks(), one_req_need);
        let relieved = service
            .relieve(bindings, Err(DomainError::Capacity(cap_a)), true, |_, _| Ok(Ok(())))
            .unwrap();
        assert!(relieved.current.is_ok(), "relief runs and succeeds when over is coverable");
        assert!(
            matches!(service.requests[&ids[2]].state, State::Between(_, Stage::NonResident(None)))
                && service.requests[&ids[2]].record.preemption_debt == 1,
            "ids[2] was evicted to cover the full over-budget shortfall"
        );

        // Case B: `over` exceeds everything evictable, even though `tiny_required`
        // alone would be covered by evicting `ids[1]`. Relief must NOT partially
        // evict `ids[1]` for nothing.
        let huge_over = 4_511_149_824u64;
        let uncoverable = magnitude_executor::memory::ledger::Short {
            store,
            required: tiny_required,
            over: huge_over,
        };
        let cap_b = magnitude_executor::CapacityError::from_short(resource, &uncoverable);
        assert_eq!(cap_b.available, 0);
        assert_eq!(cap_b.lacks(), huge_over);
        let result = service
            .relieve::<()>(bindings, Err(DomainError::Capacity(cap_b)), true, |_, _| {
                panic!("uncoverable over-budget shortfall must not retry")
            })
            .unwrap();
        assert!(!result.pending);
        match result.current {
            Err(DomainError::Capacity(err)) => {
                assert_eq!(err.required, tiny_required);
                assert_eq!(err.available, 0);
                assert_eq!(err.lacks(), huge_over);
            }
            other => panic!("expected CapacityError, got {other:?}"),
        }
        assert_eq!(
            service.requests[&ids[1]].record.preemption_debt, 0,
            "ids[1] must not be partially evicted when full over-budget shortfall cannot be covered"
        );
    }

    /// A drafting load whose drafter was released with nothing resident
    /// serves a drafting request with every byte of room taken: the
    /// drafter's reload is refused for memory while that request's group
    /// provisions, so the request fails on its own, and the owner, never
    /// stopping, serves the next request once the room is free again. Each
    /// request draws the load's request slot in turn, so none of them needs
    /// room to open.
    #[test]
    fn a_drafter_reload_refused_for_memory_fails_its_request_alone() {
        use std::task::{Context, Poll, Waker};
        let (directory, definition, domain, bindings) = fixtures::loaded::drafted("refused-drafter-reload", fixtures::loaded::MTP, 0);
        let vocabulary = definition.decoder.vocabulary as usize;
        let mut owner = Some(Owner::new(domain, bindings, limits(), Arc::new(NoWakes)).unwrap());
        // The service clock, a millisecond a run.
        let now = std::cell::Cell::new(0u64);
        // Run the owner until `receiver` ends: its terminal publication.
        let serve = |owner: &mut Option<Owner<magnitude_executor::NativeFamily>>, receiver: &mut PublicationReceiver| loop {
            let (ran, _) = owner.take().expect("the owner is between runs").run(now.get()).unwrap_or_else(|error| panic!("the owner stopped: {error}"));
            *owner = Some(ran);
            now.set(now.get() + 1_000_000);
            while let Poll::Ready(publication) = receiver.poll_next(&mut Context::from_waker(Waker::noop())) {
                match publication.expect("a request ends with a terminal publication") {
                    Publication::Output(_) => {}
                    terminal => return terminal,
                }
            }
        };
        let admit = |owner: &mut Option<Owner<magnitude_executor::NativeFamily>>, method| {
            owner.as_mut().unwrap().admit(fresh_by(vocabulary, method), PreparedModelInput::continuation_only(), PrefixRetention::Transient, 4, now.get()).unwrap().1
        };
        let mut plain = admit(&mut owner, MethodChoice::Plain);
        assert!(matches!(serve(&mut owner, &mut plain), Publication::Completed { .. }), "the first request is served");
        {
            let Owner { service, pipeline: Pipeline::Idle(bindings) } = owner.as_mut().unwrap() else { panic!("an idle owner") };
            // With nothing resident, the drafter is released, as the ledger takes it.
            service.domain.release_idle_optional_components(bindings, &[magnitude_executor::OptionalComponent::Drafter]).unwrap();
            take_room(service);
        }
        let mut drafting = admit(&mut owner, MethodChoice::Mtp { proposals: 2 });
        assert!(
            matches!(serve(&mut owner, &mut drafting), Publication::Failed(RequestError::Capacity(_))),
            "the drafting request fails for memory on its own"
        );
        owner.as_mut().unwrap().service.domain.release_measurement_room();
        let mut next = admit(&mut owner, MethodChoice::Plain);
        assert!(matches!(serve(&mut owner, &mut next), Publication::Completed { .. }), "the owner serves the next request");
        drop(owner);
        std::fs::remove_dir_all(&directory).expect("the fixture's directory is removed");
    }

    /// A request stopped part way through its prompt, while it holds a
    /// chain of prompt chunks, keeps the prompt work its rounds reconciled:
    /// its chain concludes there, and that state is cached as its path's
    /// prefix, as a completed request's whole path is, at the position the
    /// state holds; a request sharing the prompt resumes from it.
    #[test]
    fn a_request_stopped_mid_prompt_caches_the_prompt_it_reached() {
        use crate::prefix_cache::{PrefixCacheCapacity, PrefixPath};
        use std::task::{Context, Poll, Waker};
        const PROMPT: usize = 2 * MIN_PREFIX_HIT;
        let (directory, definition, domain, bindings) = fixtures::loaded::chaining("stopped-mid-prompt", 2 * PROMPT as u64, 3);
        let vocabulary = definition.decoder.vocabulary as usize;
        let prompt = || {
            let options = Options {
                max_tokens: 2,
                output_capacity: 4,
                context_limit: 2 * PROMPT,
                vocabulary,
                stop_tokens: BTreeSet::new(),
                suppressed_tokens: BTreeSet::new(),
                sampling: Sampling::Greedy,
                shaping: Shaping { temperature: 0.0, ..Default::default() },
                seed: 0,
                forced_quantum: 0,
                method: MethodChoice::Plain,
                end_of_generation: EndOfGeneration::Stop,
                reasoning_budget: None,
            };
            let tokens = (0..PROMPT as u32).map(|token| TokenId(1 + token % 7)).collect::<Vec<_>>();
            Generation::new(tokens, InputLayout::new(PROMPT, vec![]).unwrap(), options, None).unwrap()
        };
        let capacity = PrefixCacheCapacity { max_entries: 4, workload_state_bytes: u64::MAX };
        let mut owner = Some(Owner::with_prefix_cache_capacity(domain, bindings, limits(), capacity, Arc::new(NoWakes), None, None, 0).unwrap());
        let retain = PrefixRetention::Retain { cache_points: Vec::new() };
        let (id, mut receiver) = owner.as_mut().unwrap().admit(prompt(), PreparedModelInput::continuation_only(), retain, 4, 0).unwrap();
        let mut now = 0u64;
        let mut run = |owner: &mut Option<Owner<magnitude_executor::NativeFamily>>| {
            let (ran, _) = owner.take().expect("the owner is between runs").run(now).unwrap_or_else(|error| panic!("the owner stopped: {error}"));
            *owner = Some(ran);
            now += 1_000_000;
        };
        // Past the least prefix a lookup resumes from, between chunks of a
        // chain it holds, it is stopped.
        let stoppable = |owner: &Option<Owner<magnitude_executor::NativeFamily>>| {
            let owner = owner.as_ref().unwrap();
            owner.snapshot(id).unwrap().resident_position > MIN_PREFIX_HIT && owner.service.domain.chain_holder() == Some(id)
        };
        while !stoppable(&owner) {
            assert!(owner.as_ref().unwrap().snapshot(id).unwrap().resident_position < PROMPT, "its prompt chains before it ends");
            run(&mut owner);
        }
        let reached = owner.as_ref().unwrap().snapshot(id).unwrap().resident_position;
        assert!(reached < PROMPT, "stopped mid-prompt, at {reached}");
        let retirement = receiver.retirement();
        assert!(retirement.poll(Waker::noop()).is_pending());
        owner.as_mut().unwrap().stop(id);
        let finish = loop {
            run(&mut owner);
            match receiver.poll_next(&mut Context::from_waker(Waker::noop())) {
                Poll::Ready(Some(Publication::Completed { finish, .. })) => break finish,
                Poll::Ready(Some(Publication::Output(_))) | Poll::Pending => {}
                other => panic!("a stopped request completes, not {other:?}"),
            }
        };
        assert_eq!(finish, FinishReason::Cancelled);
        let served = owner.as_mut().unwrap();
        assert!(served.snapshot(id).is_none(), "the stopped request is retired");
        assert!(retirement.poll(Waker::noop()).is_ready());
        let sharing = prompt();
        let hit = served.service.prefix_cache.lookup(PrefixPath::of(&sharing), PROMPT).unwrap().expect("its prompt work is cached");
        assert_eq!(hit.position(), reached, "cached where it reached");
        let cached = served.service.prefix_cache.entry(hit).unwrap().state().position();
        assert_eq!(cached, hit.position(), "the cached state holds exactly the prefix it is cached as");
        drop(owner);
        std::fs::remove_dir_all(&directory).expect("the fixture's directory is removed");
    }

    /// A request stopped short of the least prefix a lookup resumes from
    /// leaves nothing cached; one stopped at it is cached there.
    #[test]
    fn a_stop_short_of_a_hit_is_not_retained() {
        let layout = InputLayout::new(2 * MIN_PREFIX_HIT, vec![]).unwrap();
        let plain = super::MethodCheckpoint::Plain;
        assert!(!super::resumes_from_stop(&layout, MIN_PREFIX_HIT - 1, &plain));
        assert!(super::resumes_from_stop(&layout, MIN_PREFIX_HIT, &plain));
    }

    /// A request stopped inside an indivisible media span leaves nothing
    /// cached, since no request resumes there; one stopped at the span's
    /// end or inside a causal span is cached where it stopped.
    #[test]
    fn a_stop_inside_an_indivisible_span_is_not_retained() {
        use magnitude_generation::{BoundaryRule, InputSpan};
        let span = |start, end, boundaries| InputSpan {
            start,
            end,
            identity: format!("media-{start}"),
            boundaries,
            language_history: false,
        };
        let (image, causal) = (MIN_PREFIX_HIT + 2..MIN_PREFIX_HIT + 10, MIN_PREFIX_HIT + 12..MIN_PREFIX_HIT + 20);
        let layout = InputLayout::new(
            2 * MIN_PREFIX_HIT,
            vec![span(image.start, image.end, BoundaryRule::Indivisible), span(causal.start, causal.end, BoundaryRule::Causal)],
        )
        .unwrap();
        let plain = super::MethodCheckpoint::Plain;
        assert!(!super::resumes_from_stop(&layout, image.start + 3, &plain), "inside the image");
        assert!(super::resumes_from_stop(&layout, image.end, &plain), "at the image's end");
        assert!(super::resumes_from_stop(&layout, causal.start + 3, &plain), "inside a causal span");
    }

    /// Two loads in one process, each with its own owner, admit in turn:
    /// no two of their requests share an identity, and the identities
    /// order their admissions across both, as the store's shared ledger
    /// keys and orders sequences by them.
    #[test]
    fn two_loads_never_share_a_request_identity() {
        let vocabulary = fixtures::tiny_definition().decoder.vocabulary as usize;
        let mut owners = [fixtures::fixture(None), fixtures::fixture(None)]
            .map(|(domain, bindings)| Owner::new(domain, bindings, limits(), Arc::new(NoWakes)).unwrap());
        let mut admitted = Vec::new();
        for owner in [0, 1, 0, 1] {
            let (id, receiver) = owners[owner]
                .admit(fresh(vocabulary), PreparedModelInput::continuation_only(), PrefixRetention::Transient, 4, 0)
                .unwrap();
            admitted.push((id, receiver));
        }
        let ids = admitted.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        assert!(ids.windows(2).all(|pair| pair[0] < pair[1]), "unique and in admission order across loads: {ids:?}");
    }

    /// Withdrawing a peer that still has a queued MTP head operation in the
    /// round preserves that operation as `Stage::Pending(_, None)` so it stays
    /// evictable if relief needs its slot and otherwise re-enters the next
    /// round unchanged.
    #[test]
    fn withdrawing_a_peer_with_pending_head_work_preserves_and_reselects_it() {
        let (domain, bindings) = fixtures::fixture(None);
        let mut owner = Owner::new(domain, bindings, limits(), Arc::new(NoWakes)).unwrap();
        let [peer, grower] = admit(&mut owner, 2)[..] else { unreachable!() };
        let Pipeline::Idle(bindings) = &mut owner.pipeline else { panic!("an idle owner") };
        let service = &mut owner.service;
        assert_eq!(residency(service, bindings, peer), Some(true));
        assert_eq!(residency(service, bindings, grower), Some(true));
        let head_op = Operation::Head {
            request: peer,
            phase: magnitude_executor::HeadPhase::Priming { draft_from: 5 },
            tokens: vec![TokenId(4)],
            conditioning: magnitude_executor::FeatureRows::new(vec![0u8; 4].into(), 1).unwrap(),
            position: 3,
            proposals: Vec::new(),
            form: magnitude_executor::DraftForm::Chained,
        };
        head_op.validate().unwrap();
        let Request { record: peer_record, state: State::Between(peer_gen, Stage::Ready) } =
            service.requests.remove(&peer).unwrap()
        else {
            panic!("resident peer");
        };
        let Request { record: grower_record, state: State::Between(grower_gen, Stage::Ready) } =
            service.requests.remove(&grower).unwrap()
        else {
            panic!("resident grower");
        };
        let Ok(RoundStart::Target(grower_round)) = grower_gen.start_round(grower, 2, FrontGrant::UpTo(0)) else {
            panic!("grower target round");
        };
        let grower_op = lower_round(&grower_round, grower, grower_round.generation().resident_position(), None).unwrap();
        let selection = service
            .scheduler
            .select(
                &[
                    ScheduledOperation {
                        identity: peer,
                        phase: Phase::Decode,
                        active: false,
                        resident: true,
                        waiting_since_ns: 0,
                        service_ns: 0,
                        preemption_debt: 0,
                        holds_chain: false,
                    },
                    ScheduledOperation {
                        identity: grower,
                        phase: Phase::Decode,
                        active: false,
                        resident: true,
                        waiting_since_ns: 0,
                        service_ns: 0,
                        preemption_debt: 0,
                        holds_chain: false,
                    },
                ],
                0,
            )
            .unwrap()
            .unwrap();
        let mut round = Round {
            selection,
            started: 0,
            members: BTreeMap::from([
                (peer, Member { record: peer_record, state: Membership::Generation(peer_gen) }),
                (grower, Member { record: grower_record, state: Membership::Started(grower_round) }),
            ]),
            groups: group(&service.domain, vec![grower_op, head_op.clone()]).into_iter().collect(),
        };
        service.withdraw(&mut round, &BTreeSet::from([peer]));
        assert!(!round.members.contains_key(&peer));
        assert!(round.groups.iter().flat_map(|g| g.operations()).all(|op| op.request() == grower));
        assert!(service.evictable().contains(&peer), "withdrawn peer with pending head work stays evictable");
        assert!(matches!(
            &service.requests[&peer].state,
            State::Between(_, Stage::Pending(work, None)) if work.0 == [head_op.clone()]
        ));
        service.scheduler.completed(round.selection, 0);
        let Ok(Selected::Round(next)) = service.select(bindings) else {
            panic!("withdrawn peer is re-selected");
        };
        let ops = next.groups.iter().flat_map(|g| g.operations()).cloned().collect::<Vec<_>>();
        assert_eq!(ops, vec![head_op], "withdrawn pending head op is preserved into the next round");
    }

    /// Failing a post-stop reconciliation round with `WeightRead` abandons
    /// terminal prefix retention without overwriting the accepted `Stop`
    /// completion: the owner publishes one `Completed { finish: Stop }`
    /// outcome, retires the resident slot, and does not retry reconciliation.
    #[test]
    fn failed_post_stop_reconciliation_completes_once_and_retires_slot() {
        use std::task::{Context, Poll, Waker};
        struct NoFeatures;
        impl magnitude_executor::FeatureReader for NoFeatures {
            fn read(&mut self, _: &magnitude_executor::FeatureSpan) -> Result<magnitude_executor::FeatureRows, String> {
                panic!("plain generation reads no features")
            }
        }
        let (domain, bindings) = fixtures::fixture(None);
        let vocabulary = fixtures::tiny_definition().decoder.vocabulary as usize;
        let stop = TokenId((vocabulary - 1) as u32);
        let capacity = PrefixCacheCapacity { max_entries: 4, workload_state_bytes: u64::MAX };
        let mut owner = Owner::with_prefix_cache_capacity(domain, bindings, limits(), capacity, Arc::new(NoWakes), None, None, 0).unwrap();
        let options = Options {
            max_tokens: 4,
            output_capacity: 4,
            context_limit: 8,
            vocabulary,
            stop_tokens: BTreeSet::from([stop]),
            suppressed_tokens: BTreeSet::new(),
            sampling: Sampling::Greedy,
            shaping: Shaping { temperature: 0.0, ..Default::default() },
            seed: 0,
            forced_quantum: 0,
            method: MethodChoice::Plain,
            end_of_generation: EndOfGeneration::Stop,
            reasoning_budget: None,
        };
        let generation = Generation::new(vec![TokenId(1), TokenId(2)], InputLayout::new(2, vec![]).unwrap(), options, None).unwrap();
        let (id, mut receiver) = owner
            .admit(generation, PreparedModelInput::continuation_only(), PrefixRetention::Retain { cache_points: Vec::new() }, 4, 0)
            .unwrap();
        let Pipeline::Idle(bindings) = &mut owner.pipeline else { panic!("an idle owner") };
        let service = &mut owner.service;
        assert_eq!(residency(service, bindings, id), Some(true));
        let Request { mut record, state: State::Between(generation, Stage::Ready) } =
            service.requests.remove(&id).unwrap()
        else {
            panic!("resident request");
        };
        record.clock.serve(Instant::now());
        let publication = record.publication.as_mut().unwrap();
        let permits = publication.sender.reserve_outputs(1).unwrap();
        publication.extend_permits(permits);
        let Ok(RoundStart::Target(round)) = generation.start_round(id, 4, FrontGrant::UpTo(0)) else {
            panic!("prompt target round");
        };
        let transition = round.prepare_round_transition(id, &[stop], Vec::new(), None, &mut NoFeatures).unwrap();
        let (generation, _) = round.commit(transition);
        assert_eq!(generation.finish_reason(), Some(FinishReason::Stop));
        assert!(owes_reconciliation(&record, &generation), "post-stop request owes reconciliation");
        service.requests.insert(id, Request { record, state: State::Between(generation, Stage::Ready) });
        service.publish_ready().unwrap();
        assert!(service.requests.contains_key(&id), "blocked on reconciliation before failure");
        let Ok(Selected::Round(mut round)) = service.select(bindings) else {
            panic!("reconciliation round is selected");
        };
        let group_ = round.groups.pop_front().expect("reconciliation group");
        service.refuse(
            &mut round,
            &group_,
            RequestError::WeightRead(magnitude_executor::WeightReadError {
                detail: "simulated weight read failure".into(),
            }),
        );
        service.finish_round(round).unwrap();
        assert!(!service.requests.contains_key(&id), "request retired after abandoned reconciliation");
        assert!(!service.domain.holds_state_beside(&BTreeSet::new()), "resident slot released");
        let mut publications = Vec::new();
        while let Poll::Ready(Some(publication)) = receiver.poll_next(&mut Context::from_waker(Waker::noop())) {
            publications.push(publication);
        }
        let terminals = publications
            .iter()
            .filter(|publication| !matches!(publication, Publication::Output(_)))
            .collect::<Vec<_>>();
        assert_eq!(terminals.len(), 1, "exactly one terminal publication: {publications:?}");
        assert!(
            matches!(terminals[0], Publication::Completed { finish: FinishReason::Stop, .. }),
            "completed with Stop, not Failed: {:?}",
            terminals[0]
        );
        assert!(matches!(
            receiver.poll_next(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(None)
        ));
        assert!(matches!(service.select(bindings), Ok(Selected::Empty)), "no reconciliation retry");
    }

    /// When a waiter's real prefix producer exits while an unrelated prompt
    /// happens to plan a branch at the same offset `through`, the waiter
    /// releases immediately instead of remaining stuck in `AwaitingPrefix`.
    #[test]
    fn waiter_releases_when_real_prefix_producer_exits_despite_unrelated_branch() {
        const THROUGH: usize = MIN_BRANCH_GAIN;
        const PROMPT: usize = THROUGH + 32;
        let mut definition = fixtures::tiny_definition();
        definition.decoder.context_limit = 512;
        let vocabulary = definition.decoder.vocabulary as usize;
        let (domain, bindings) = fixtures::fixture_of(
            definition,
            fixtures::limits(2, 1, false),
            None,
            false,
            magnitude_executor::platform::DeviceRequest::Automatic,
        );
        let capacity = PrefixCacheCapacity { max_entries: 4, workload_state_bytes: u64::MAX };
        let mut owner = Owner::with_prefix_cache_capacity(domain, bindings, limits(), capacity, Arc::new(NoWakes), None, None, 0).unwrap();
        let make_gen = |tokens: Vec<TokenId>| {
            let options = Options {
                max_tokens: 2,
                output_capacity: 4,
                context_limit: 512,
                vocabulary,
                stop_tokens: BTreeSet::new(),
                suppressed_tokens: BTreeSet::new(),
                sampling: Sampling::Greedy,
                shaping: Shaping { temperature: 0.0, ..Default::default() },
                seed: 0,
                forced_quantum: 0,
                method: MethodChoice::Plain,
                end_of_generation: EndOfGeneration::Stop,
                reasoning_budget: None,
            };
            Generation::new(tokens, InputLayout::new(PROMPT, vec![]).unwrap(), options, None).unwrap()
        };
        let mut producer_tokens = vec![TokenId(1); THROUGH];
        producer_tokens.resize(PROMPT, TokenId(2));
        let mut waiter_tokens = vec![TokenId(1); THROUGH];
        waiter_tokens.resize(PROMPT, TokenId(3));
        let unrelated_tokens = vec![TokenId(4); PROMPT];
        let (producer, _rx_producer) = owner
            .admit(make_gen(producer_tokens), PreparedModelInput::continuation_only(), PrefixRetention::Retain { cache_points: Vec::new() }, 4, 0)
            .unwrap();
        let (waiter, _rx_waiter) = owner
            .admit(make_gen(waiter_tokens), PreparedModelInput::continuation_only(), PrefixRetention::Retain { cache_points: Vec::new() }, 4, 0)
            .unwrap();
        assert!(matches!(
            owner.service.requests[&waiter].state,
            State::Between(_, Stage::AwaitingPrefix { producer: p, through }) if p == producer && through == THROUGH
        ));
        let (unrelated, _rx_unrelated) = owner
            .admit(make_gen(unrelated_tokens), PreparedModelInput::continuation_only(), PrefixRetention::Retain { cache_points: vec![THROUGH] }, 4, 0)
            .unwrap();
        assert!(owner.service.requests[&unrelated].record.prefix.as_ref().unwrap().branches.contains(&THROUGH));
        owner.cancel(producer);
        owner.service.publish_ready().unwrap();
        assert!(!owner.service.requests.contains_key(&producer));
        let selectable = owner.service.selectable_requests();
        assert!(selectable.contains(&waiter), "waiter releases immediately when its real producer exits");
        assert!(matches!(owner.service.requests[&waiter].state, State::Between(_, Stage::NonResident(None))));
        assert!(selectable.contains(&unrelated));
    }

    /// When a pre-reservation handoff inside `Service::select` fails, every
    /// selectable request remains owner-held in `self.requests` so
    /// `Owner::run` delivers the real error rather than dropping its
    /// `PublicationSender` (`RequestError::WorkerClosed`).
    #[test]
    fn select_failure_delivers_real_error_to_queued_requests() {
        use std::task::{Context, Poll, Waker};
        let (domain, bindings) = fixtures::fixture(None);
        let vocabulary = fixtures::tiny_definition().decoder.vocabulary as usize;
        let mut owner = Owner::new(domain, bindings, limits(), Arc::new(NoWakes)).unwrap();
        let (first, mut rx1) = owner
            .admit(fresh(vocabulary), PreparedModelInput::continuation_only(), PrefixRetention::Transient, 4, 0)
            .unwrap();
        let (_second, mut rx2) = owner
            .admit(fresh(vocabulary), PreparedModelInput::continuation_only(), PrefixRetention::Transient, 4, 0)
            .unwrap();
        // A waiting timestamp past `now` causes `scheduler.select` to fail
        // before the reservation loop.
        owner.service.requests.get_mut(&first).unwrap().record.waiting_since = 100;
        let err = owner.run(10).err().expect("select fails with scheduler invariant error");
        assert!(matches!(err, RequestError::Invariant(ref inv) if inv.context == "scheduler"));
        for (label, rx) in [("first", &mut rx1), ("second", &mut rx2)] {
            let Poll::Ready(Some(Publication::Failed(received))) =
                rx.poll_next(&mut Context::from_waker(Waker::noop()))
            else {
                panic!("{label} request did not receive terminal failure");
            };
            assert_eq!(received, err, "{label} request must receive the real error, not WorkerClosed");
        }
    }

    #[test]
    fn progressing_state_beside_excludes_unexpired_memory_waiters() {
        let (domain, bindings) = fixtures::fixture(None);
        let vocabulary = fixtures::tiny_definition().decoder.vocabulary as usize;
        let mut owner = Owner::new(domain, bindings, limits(), Arc::new(NoWakes)).unwrap();
        let (first, _rx1) = owner
            .admit(fresh(vocabulary), PreparedModelInput::continuation_only(), PrefixRetention::Transient, 4, 0)
            .unwrap();
        let (second, _rx2) = owner
            .admit(fresh(vocabulary), PreparedModelInput::continuation_only(), PrefixRetention::Transient, 4, 0)
            .unwrap();
        let Pipeline::Idle(bindings) = &mut owner.pipeline else { panic!("an idle owner") };
        let service = &mut owner.service;
        service.memory.reading = ReadingState::Active;
        assert_eq!(residency(service, bindings, first), Some(true));
        assert_eq!(residency(service, bindings, second), Some(true));
        let Request { record, state: State::Between(generation, Stage::Ready) } =
            service.requests.remove(&first).unwrap()
        else {
            panic!("resident first request");
        };
        let Ok(RoundStart::Target(round)) = generation.start_round(first, 4, FrontGrant::UpTo(0)) else {
            panic!("prompt target round");
        };
        let work = lower_round(&round, first, round.generation().resident_position(), None).unwrap();
        service.requests.insert(
            first,
            Request {
                record,
                state: State::Parked(
                    round,
                    Work::new(vec![work]).unwrap(),
                    Some(Wait { at: service.epoch, on: WaitsOn::Memory }),
                ),
            },
        );
        assert!(
            !service.progressing_state_beside(&BTreeSet::from([second])),
            "a peer parked on an unexpired memory wait cannot wake the last progressing request"
        );
        assert!(
            service.progressing_state_beside(&BTreeSet::from([first])),
            "a ready resident peer can still progress and free memory"
        );
        service.check_no_unwakeable_wait();
        service.epoch.advance();
        assert!(
            service.progressing_state_beside(&BTreeSet::from([second])),
            "once the epoch advances, the parked peer's wait has expired and it can retry"
        );
    }

    #[test]
    #[should_panic(expected = "waiting on memory that nothing under way can free")]
    fn check_no_unwakeable_wait_catches_mutual_memory_wait_deadlock() {
        let (domain, bindings) = fixtures::fixture(None);
        let vocabulary = fixtures::tiny_definition().decoder.vocabulary as usize;
        let mut owner = Owner::new(domain, bindings, limits(), Arc::new(NoWakes)).unwrap();
        let (first, _rx1) = owner
            .admit(fresh(vocabulary), PreparedModelInput::continuation_only(), PrefixRetention::Transient, 4, 0)
            .unwrap();
        let (second, _rx2) = owner
            .admit(fresh(vocabulary), PreparedModelInput::continuation_only(), PrefixRetention::Transient, 4, 0)
            .unwrap();
        let Pipeline::Idle(bindings) = &mut owner.pipeline else { panic!("an idle owner") };
        let service = &mut owner.service;
        service.memory.reading = ReadingState::Active;
        assert_eq!(residency(service, bindings, first), Some(true));
        assert_eq!(residency(service, bindings, second), Some(true));
        for id in [first, second] {
            let Request { record, state: State::Between(generation, Stage::Ready) } =
                service.requests.remove(&id).unwrap()
            else {
                panic!("resident request");
            };
            let Ok(RoundStart::Target(round)) = generation.start_round(id, 4, FrontGrant::UpTo(0)) else {
                panic!("prompt target round");
            };
            let work = lower_round(&round, id, round.generation().resident_position(), None).unwrap();
            service.requests.insert(
                id,
                Request {
                    record,
                    state: State::Parked(
                        round,
                        Work::new(vec![work]).unwrap(),
                        Some(Wait { at: service.epoch, on: WaitsOn::Memory }),
                    ),
                },
            );
        }
        service.check_no_unwakeable_wait();
    }
}

#[cfg(test)]
mod front_rows_tests {
    use super::*;
    use magnitude_executor::{FeatureReader, FeatureRef, FeatureRows, FeatureSpan, Guess, GuessKind, ResourceDomainId};
    use magnitude_generation::{RoundStart, StartedRound, TokenId};

    struct NoFeatures;
    impl FeatureReader for NoFeatures {
        fn read(&mut self, _: &FeatureSpan) -> Result<FeatureRows, String> {
            panic!("a front-decoding head reads no feature on the host")
        }
    }

    /// The features a chained step returns, which the head entered on the
    /// device behind it: none is read.
    fn features() -> FeatureRef {
        FeatureRef::logical(ResourceDomainId::new("front-rows").unwrap(), 2, 4).unwrap()
    }

    /// An MTP generation on a load that fronts of request `id` decoding
    /// within `context_limit` tokens: its prompt closed, its first token
    /// selected and no guess standing.
    fn decoding(id: RequestId, context_limit: usize) -> Generation {
        let options = magnitude_generation::Options {
            context_limit,
            output_capacity: 32,
            ..super::termination_tests::options(32, magnitude_generation::MethodChoice::Mtp { proposals: 2 })
        };
        let layout = magnitude_generation::InputLayout::new(2, vec![]).unwrap();
        let method = Arc::new(magnitude_generation::Mtp::new("fixture", 2).unwrap().fronting().unwrap());
        let mut generation = Generation::new_with_method(vec![TokenId(1), TokenId(2)], layout, options, None, method).unwrap();
        generation.resume_at(None).unwrap();
        commit(id, started(id, generation, 4, FrontGrant::UpTo(0)), &[TokenId(3)], Vec::new())
    }

    fn started(id: RequestId, generation: Generation, allowance: usize, grant: FrontGrant) -> StartedRound {
        match generation.start_round(id, allowance, grant) {
            Ok(RoundStart::Target(round)) => round,
            _ => panic!("expected a target round"),
        }
    }

    /// `round` committed with `samples`, its exact row's selection, and
    /// `chain`, the head launch behind it's.
    fn commit(id: RequestId, round: StartedRound, samples: &[TokenId], chain: Vec<Guess>) -> Generation {
        let transition = round.prepare_round_transition(id, samples, chain, Some(features()), &mut NoFeatures).unwrap();
        round.commit(transition).0
    }

    fn guess(token: u32, kind: GuessKind) -> Guess {
        Guess { token: TokenId(token), kind }
    }

    /// [`decoding`], then a first pass, of no guess, selecting 4, behind
    /// which the head guessed 5: its next pass pairs on that guess, where
    /// its rows fit the context.
    fn guessing(id: RequestId, context_limit: usize) -> Generation {
        let first = started(id, decoding(id, context_limit), 4, FrontGrant::UpTo(1));
        assert_eq!(first.round_forward().tokens, [TokenId(3)], "a first pass holds no guess");
        commit(id, first, &[TokenId(4)], vec![guess(5, GuessKind::Head)])
    }

    /// The domain's grant up to `depth` of a pass the host builds, with
    /// every launch paying: as many of the requests that could front, first
    /// first, as 8 rows hold beside every exact row.
    fn granted(requests: &[RequestId], fronted: &[RequestId], depth: usize) -> Result<FrontPass, ()> {
        let count = magnitude_executor::fronted_requests(magnitude_executor::CHAINED_ROWS, requests.len(), depth).min(fronted.len());
        Ok(FrontPass::Built { form: BuiltForm::Front, depths: fronted[..count].iter().map(|&id| (id, depth)).collect() })
    }

    /// The domain's grant up to depth 1 ([`granted`]).
    fn pairs(requests: &[RequestId], fronted: &[RequestId]) -> Result<FrontPass, ()> {
        granted(requests, fronted, 1)
    }

    /// A request's record, its prefix uncached and no stream attached.
    pub(super) fn record() -> Record {
        Record {
            waiting_since: 0, service_ns: 0, clock: RequestClock::Waiting, error: None, preemption_debt: 0,
            prefix: None,
            retirement: PublicationQueue::bounded(1, |_| {}).unwrap().1.retirement(),
            publication: None,
        }
    }

    /// Reserve `selected`, with `records`, as a launch of `rows` rows that
    /// may pair does, in selection order: each within the rows no round
    /// reserved before it holds. The rounds reserved; any other request is
    /// turned back.
    fn reserve_with(selected: Vec<(RequestId, Generation, Record)>, rows: usize) -> (Vec<(RequestId, ReservedRound)>, BTreeMap<RequestId, Request>) {
        let mut epoch = AvailabilityEpoch::default();
        let (mut reserved, mut returned) = (Vec::new(), BTreeMap::new());
        for (id, generation, record) in selected {
            match ReservedRound::reserve(record, generation, rows - reserved.len(), 1, &mut epoch) {
                Reservation::Round(round) => reserved.push((id, round)),
                Reservation::Returned(request) => {
                    returned.insert(id, request);
                }
                Reservation::Composed(..) => panic!("a ready request reserves a round"),
            }
        }
        (reserved, returned)
    }

    /// [`reserve_with`] `selected`, each with its own [`record`], every one
    /// reserved.
    fn reserve(selected: Vec<(RequestId, Generation)>, rows: usize) -> Vec<(RequestId, ReservedRound)> {
        let (reserved, returned) = reserve_with(selected.into_iter().map(|(id, generation)| (id, generation, record())).collect(), rows);
        assert!(returned.is_empty(), "every request reserves its round");
        reserved
    }

    /// [`FrontRows::of`] the `reserved` rounds within a budget of `rows`
    /// rows, `counted` taking what its pricing was asked.
    fn fronts_counted(reserved: &[(RequestId, ReservedRound)], rows: usize, counted: &mut Option<(usize, usize)>) -> FrontRows {
        FrontRows::of(reserved.iter().map(|(id, round)| (*id, round.front_guess())), rows, |requests, fronted| {
            *counted = Some((requests.len(), fronted.len()));
            pairs(requests, fronted)
        })
        .unwrap()
    }

    /// [`FrontRows::of`] the `reserved` rounds within a budget of `rows` rows.
    fn fronts_within(reserved: &[(RequestId, ReservedRound)], rows: usize) -> FrontRows {
        fronts_counted(reserved, rows, &mut None)
    }

    /// Whether `fronts` grants no request a guessed row.
    fn grants_none(fronts: &FrontRows) -> bool {
        matches!(fronts, FrontRows::Built { granted, .. } if granted.is_empty())
    }

    /// A launch's committed generations, and each request's rows and
    /// whether it ran a front row ([`launch`]).
    type Launched = (Vec<(RequestId, Generation)>, BTreeMap<RequestId, (usize, bool)>);

    /// One launch of the `reserved` rounds, `fronts` its front rows, built
    /// as the scheduler builds it ([`Launch::build`]), each chaining
    /// its forward, and committed with its guess right: a pass that ran a
    /// front row keeps a chained guess, and one that ran none gets the
    /// head's. The requests that ran a front row, and the rows each took.
    fn launch(reserved: Vec<(RequestId, ReservedRound)>, fronts: &FrontRows) -> Launched {
        let mut built = Launch::new(usize::MAX);
        let mut returned = BTreeMap::new();
        built.build(reserved, fronts, &mut returned, |_, _| {}).unwrap();
        assert!(returned.is_empty(), "every reserved round is built");
        let after = |token: TokenId| TokenId(token.0 + 1);
        let mut ran = BTreeMap::new();
        let mut outcomes = BTreeMap::new();
        for operation in &built.operations {
            let Operation::Forward { request, tokens, front, .. } = operation else {
                panic!("a decode round is a forward");
            };
            assert!(operation.chain().is_some(), "every forward of the launch chains");
            let front = !front.is_empty();
            // A pair's exact row selects its guess, and the head's chain
            // continues it; one row selects the token after its own.
            let outcome = match (tokens.as_slice(), front) {
                (&[_, guess], true) => (guess, vec![Guess { token: after(guess), kind: GuessKind::Chained }]),
                (&[pending], _) => (after(pending), vec![Guess { token: after(after(pending)), kind: GuessKind::Head }]),
                other => panic!("a decode round of {other:?}"),
            };
            ran.insert(*request, (operation.row_count(), front));
            outcomes.insert(*request, outcome);
        }
        assert_eq!(built.rows, ran.values().map(|&(rows, _)| rows).sum::<usize>());
        let next = built
            .members
            .into_iter()
            .map(|(id, member)| {
                let Membership::Started(round) = member.state else {
                    panic!("a decode round starts");
                };
                let (sample, chain) = outcomes.remove(&id).unwrap();
                (id, commit(id, round, &[sample], chain))
            })
            .collect();
        (next, ran)
    }

    /// `requests` requests from 1, decoding with guesses standing.
    fn guessing_all(requests: u64) -> Vec<(RequestId, Generation)> {
        (1..=requests).map(|id| (RequestId(id), guessing(RequestId(id), 32))).collect()
    }

    /// N requests decoding with guesses standing in a launch pairing
    /// min(N, 8 - N) of them: those pair, two rows each, and the rest decode
    /// one row, each held to it although allowed more, so the launch holds
    /// the rows its pairing counted.
    #[test]
    fn requests_pair_only_as_their_launch_counted() {
        for requests in [2u64, 4, 5, 7] {
            let reserved = reserve(guessing_all(requests), 64);
            let fronts = fronts_within(&reserved, 64);
            let (_, ran) = launch(reserved, &fronts);
            let left = magnitude_executor::fronted_requests(magnitude_executor::CHAINED_ROWS, requests as usize, 1);
            let paired = ran.values().filter(|&&(rows, front)| (rows, front) == (2, true)).count();
            let rows = ran.values().map(|&(rows, _)| rows).sum::<usize>();
            assert_eq!((paired, rows), (left, requests as usize + left), "{requests} requests: {ran:?}");
        }
    }

    /// N = 5 to 7 requests, selected in a rotating order over several
    /// launches from decode's start: the same min(N, 8 - N) requests hold
    /// front rows in every launch, the lowest first as none had a guess,
    /// and from the second launch on each continues its chain of guesses
    /// before the others' head guesses; none of them loses its front row to
    /// another request selected before it.
    #[test]
    fn requests_keep_their_front_rows_across_launches() {
        for requests in 5u64..=7 {
            let left = magnitude_executor::fronted_requests(magnitude_executor::CHAINED_ROWS, requests as usize, 1);
            let mut generations = (1..=requests).map(|id| (RequestId(id), decoding(RequestId(id), 32))).collect::<BTreeMap<_, _>>();
            for turn in 0..4 {
                let mut order = generations.keys().copied().collect::<Vec<_>>();
                let rotation = turn % order.len();
                order.rotate_left(rotation);
                let selected = order.iter().map(|id| (*id, generations.remove(id).unwrap())).collect::<Vec<_>>();
                let reserved = reserve(selected, 64);
                let fronts = fronts_within(&reserved, 64);
                assert_eq!(grants_none(&fronts), turn == 0, "only the first launch holds no guess");
                let (next, ran) = launch(reserved, &fronts);
                generations.extend(next);
                if turn == 0 {
                    continue;
                }
                let fronted = ran.iter().filter(|(_, &(_, front))| front).map(|(id, _)| id.0).collect::<Vec<_>>();
                assert_eq!(fronted, (1..=left as u64).collect::<Vec<_>>(), "{requests} requests, launch {turn}: {ran:?}");
                assert!(fronted.iter().all(|&id| ran[&RequestId(id)].0 == 2), "{requests} requests, launch {turn}: every front row pairs: {ran:?}");
            }
        }
    }

    /// A request whose two-row pass would not fit its context, so it would
    /// decode in one row, is counted as no pair: its launch is priced for
    /// the others alone, and it is granted no front row.
    #[test]
    fn a_request_falling_back_to_one_row_counts_as_no_pair() {
        let short = guessing(RequestId(3), 4);
        assert_eq!(short.front_guess(), None, "its pair's two rows pass its context");
        let selected = vec![(RequestId(1), guessing(RequestId(1), 32)), (RequestId(2), guessing(RequestId(2), 32)), (RequestId(3), short)];
        let reserved = reserve(selected, 64);
        let mut counted = None;
        let fronts = fronts_counted(&reserved, 64, &mut counted);
        assert_eq!(counted, Some((3, 2)), "three requests, two of which could pair");
        let (_, ran) = launch(reserved, &fronts);
        assert_eq!(ran[&RequestId(3)], (1, false));
        assert_eq!((ran[&RequestId(1)], ran[&RequestId(2)]), ((2, true), (2, true)));
    }

    /// A request its launch turns back, its publication full, is neither
    /// priced nor granted a front row: the launch is priced for the requests
    /// it reserved, and builds each pair it priced. Priced before the
    /// reservation turned it back, the launch would have counted three
    /// requests and granted it a pair it never builds.
    #[test]
    fn a_request_turned_back_is_neither_priced_nor_granted_a_front_row() {
        let (sender, _receiver) = PublicationQueue::bounded(1, |_| {}).unwrap();
        let _held = sender.reserve_outputs(1).unwrap();
        let full = Record { publication: Some(ActivePublication::new(sender, 4)), ..record() };
        let selected = vec![
            (RequestId(1), guessing(RequestId(1), 32), record()),
            (RequestId(2), guessing(RequestId(2), 32), full),
            (RequestId(3), guessing(RequestId(3), 32), record()),
        ];
        let (reserved, returned) = reserve_with(selected, 64);
        assert!(matches!(returned[&RequestId(2)].state, State::Between(_, Stage::AwaitingCredit)), "its publication is full");
        assert_eq!(returned.len(), 1);
        let mut counted = None;
        let fronts = fronts_counted(&reserved, 64, &mut counted);
        assert_eq!(counted, Some((2, 2)), "the two requests reserved, both of which could pair");
        let (_, ran) = launch(reserved, &fronts);
        assert_eq!(ran, BTreeMap::from([(RequestId(1), (2, true)), (RequestId(3), (2, true))]));
    }

    /// A request whose receiver is closed is turned back by its reservation
    /// too, so it is neither priced nor granted a front row.
    #[test]
    fn a_request_with_a_closed_receiver_is_neither_priced_nor_granted_a_front_row() {
        let (sender, receiver) = PublicationQueue::bounded(1, |_| {}).unwrap();
        drop(receiver);
        let closed = Record { publication: Some(ActivePublication::new(sender, 4)), ..record() };
        let selected = vec![
            (RequestId(1), guessing(RequestId(1), 32), record()),
            (RequestId(2), guessing(RequestId(2), 32), closed),
            (RequestId(3), guessing(RequestId(3), 32), record()),
        ];
        let (reserved, _) = reserve_with(selected, 64);
        let mut counted = None;
        let fronts = fronts_counted(&reserved, 64, &mut counted);
        assert_eq!(counted, Some((2, 2)), "the two requests reserved, both of which could pair");
        let (_, ran) = launch(reserved, &fronts);
        assert_eq!(ran, BTreeMap::from([(RequestId(1), (2, true)), (RequestId(3), (2, true))]));
    }

    /// A decode row budget short of a row a request and a front row a pair
    /// pairs only as many as it holds past a row a request, and each of
    /// them pairs: no pair is granted that falls back to one row. A budget
    /// of a row a request pairs none.
    #[test]
    fn a_short_decode_budget_pairs_only_what_it_holds() {
        assert!(grants_none(&fronts_within(&reserve(guessing_all(4), 4), 4)), "a row a request leaves no front row room");
        let reserved = reserve(guessing_all(4), 5);
        let fronts = fronts_within(&reserved, 5);
        let (_, ran) = launch(reserved, &fronts);
        let fronted = ran.iter().filter(|(_, &(_, front))| front).map(|(id, &taken)| (id.0, taken)).collect::<Vec<_>>();
        assert_eq!(fronted, [(1, (2, true))], "{ran:?}");
        assert_eq!(ran.values().map(|&(rows, _)| rows).sum::<usize>(), 5);
    }

    /// A request whose chain of guesses runs keeps its front row before a
    /// lower request that holds none: with room for one pair, the higher
    /// request holding a guess is granted it, and pairs.
    #[test]
    fn a_standing_guess_is_granted_before_a_lower_request() {
        let reserved = reserve(vec![(RequestId(1), decoding(RequestId(1), 32)), (RequestId(2), guessing(RequestId(2), 32))], 3);
        let fronts = fronts_within(&reserved, 3);
        let (_, ran) = launch(reserved, &fronts);
        assert_eq!((ran[&RequestId(1)], ran[&RequestId(2)]), ((1, false), (2, true)), "{ran:?}");
    }

    /// A request granted the regime's depth runs as many front rows as its
    /// chain holds within the budget, and a budget short of the depth
    /// grants as deep as it holds: one sequence of a chain of two runs
    /// three rows up to a depth of 7, and is granted 6 of a budget of 7.
    #[test]
    fn a_granted_depth_runs_the_chains_rows_within_the_budget() {
        let id = RequestId(1);
        let round = started(id, decoding(id, 32), 4, FrontGrant::UpTo(0));
        let generation = commit(id, round, &[TokenId(4)], vec![guess(5, GuessKind::Chained), guess(6, GuessKind::Head)]);
        let deep = |requests: &[RequestId], fronted: &[RequestId]| granted(requests, fronted, 7);
        let reserved = reserve(vec![(id, generation)], 8);
        let fronts = FrontRows::of(reserved.iter().map(|(id, round)| (*id, round.front_guess())), 8, deep).unwrap();
        assert_eq!(fronts.grant(id), (FrontGrant::UpTo(7), 8));
        let mut built = Launch::new(usize::MAX);
        built.build(reserved, &fronts, &mut BTreeMap::new(), |_, _| {}).unwrap();
        let rows = built.operations.iter().map(|operation| (operation.row_count(), operation.front_rows())).collect::<Vec<_>>();
        assert_eq!(rows, [(3, 2)], "its chain's two front rows");
        assert!(grants_none(&FrontRows::of([(id, None::<GuessKind>)].into_iter(), 7, deep).unwrap()), "no chain, no grant");
        let short = reserve(vec![(id, guessing(id, 32))], 8);
        let short = FrontRows::of(short.iter().map(|(id, round)| (*id, round.front_guess())), 7, deep).unwrap();
        assert_eq!(short.grant(id), (FrontGrant::UpTo(6), 7), "a budget short of the depth");
    }

    /// A launch of the requests of the pass queued behind the last one
    /// builds that pass's depths exactly, each request its own
    /// (`ExecutorDomain::front_pass`): one queued shallower than its chain runs
    /// no deeper, one queued deeper runs idle rows past its chain, one
    /// queued with no front row runs one row, whatever its chain, and one
    /// holding no guess runs its queued depth all idle, so the launch claims
    /// the queued pass rather than orphan it.
    #[test]
    fn a_launch_of_the_queued_pass_builds_each_requests_queued_depth() {
        let chained = |id: RequestId, chain: &[u32]| {
            let round = started(id, decoding(id, 32), 4, FrontGrant::UpTo(0));
            commit(id, round, &[TokenId(4)], chain.iter().map(|&token| guess(token, GuessKind::Chained)).collect())
        };
        let (single, shallower, deeper, unguessed) = (RequestId(1), RequestId(2), RequestId(3), RequestId(4));
        let reserved = reserve(
            vec![(single, chained(single, &[5, 6, 7])), (shallower, chained(shallower, &[5, 6, 7])), (deeper, chained(deeper, &[5])), (unguessed, chained(unguessed, &[]))],
            64,
        );
        let queued = |requests: &[RequestId], fronted: &[RequestId]| {
            assert_eq!((requests, fronted), (&[single, shallower, deeper, unguessed][..], &[single, shallower, deeper][..]), "every request of a guess could front");
            Ok::<_, ()>(FrontPass::Claim(BTreeMap::from([(single, 0), (shallower, 1), (deeper, 2), (unguessed, 1)])))
        };
        let fronts = FrontRows::of(reserved.iter().map(|(id, round)| (*id, round.front_guess())), 64, queued).unwrap();
        assert_eq!(fronts.grant(single), (FrontGrant::Exactly(0), 1));
        assert_eq!((fronts.grant(shallower), fronts.grant(deeper)), ((FrontGrant::Exactly(1), 2), (FrontGrant::Exactly(2), 3)));
        assert_eq!(fronts.grant(unguessed), (FrontGrant::Exactly(1), 2), "a request of no guess takes its queued depth");
        let mut built = Launch::new(usize::MAX);
        built.build(reserved, &fronts, &mut BTreeMap::new(), |_, _| {}).unwrap();
        let rows = built.operations.iter().map(|operation| (operation.request(), operation.row_count(), operation.front_rows())).collect::<Vec<_>>();
        assert_eq!(rows, [(single, 1, 0), (shallower, 2, 1), (deeper, 3, 2), (unguessed, 2, 1)], "each request's queued depth");
    }

    /// A claim of the queued pass runs that pass's rows, however unevenly
    /// they fall among its requests: queued at depths 3, 0 and 0 within the
    /// launch's 6 rows, the first runs its four rows and the others one
    /// each.
    #[test]
    fn a_claim_runs_the_queued_rows() {
        let ids = [RequestId(1), RequestId(2), RequestId(3)];
        let reserved = reserve(ids.iter().map(|&id| (id, guessing(id, 32))).collect(), 6);
        let queued = |_: &[RequestId], _: &[RequestId]| {
            Ok::<_, ()>(FrontPass::Claim(BTreeMap::from([(ids[0], 3), (ids[1], 0), (ids[2], 0)])))
        };
        let fronts = FrontRows::of(reserved.iter().map(|(id, round)| (*id, round.front_guess())), 6, queued).unwrap();
        let mut built = Launch::new(usize::MAX);
        built.build(reserved, &fronts, &mut BTreeMap::new(), |_, _| {}).unwrap();
        let rows = built.operations.iter().map(|operation| (operation.request(), operation.row_count(), operation.front_rows())).collect::<Vec<_>>();
        assert_eq!(rows, [(ids[0], 4, 3), (ids[1], 1, 0), (ids[2], 1, 0)], "the queued pass's rows");
    }

    /// Reads every feature row as zeros: a verify pass's accepted pairs
    /// cross to the host.
    struct Zeros;
    impl FeatureReader for Zeros {
        fn read(&mut self, span: &FeatureSpan) -> Result<FeatureRows, String> {
            FeatureRows::new(vec![0u8; 4 * span.count].into(), span.count).map_err(|error| error.to_string())
        }
    }

    /// A launch the regime prices in verify rows builds every request's
    /// pass in that form, so none of its forwards chains: a request granted
    /// rows verifies its chain's guesses, and one granted none verifies its
    /// exact row alone. A request whose verify pass left its accepted pairs
    /// trailing the head runs guessed rows on the head's guesses, its round
    /// first the head transaction entering them and drafting as deep as its
    /// grant.
    #[test]
    fn a_verify_launch_builds_every_pass_unchained() {
        let verifying = |requests: &[RequestId], fronted: &[RequestId]| {
            let count = magnitude_executor::fronted_requests(magnitude_executor::CHAINED_ROWS, requests.len(), 2).min(fronted.len());
            Ok::<_, ()>(FrontPass::Built { form: BuiltForm::Verify, depths: fronted[..count].iter().map(|&id| (id, 2)).collect() })
        };
        let chained = |id: RequestId, chain: &[u32]| {
            let round = started(id, decoding(id, 32), 4, FrontGrant::UpTo(0));
            commit(id, round, &[TokenId(4)], chain.iter().map(|&token| guess(token, GuessKind::Chained)).collect())
        };
        let (deep, unguessed) = (RequestId(1), RequestId(2));
        let reserved = reserve(vec![(deep, chained(deep, &[5, 6])), (unguessed, chained(unguessed, &[]))], 64);
        let fronts = FrontRows::of(reserved.iter().map(|(id, round)| (*id, round.front_guess())), 64, verifying).unwrap();
        assert_eq!((fronts.grant(deep), fronts.grant(unguessed)), ((FrontGrant::Verify(2), 3), (FrontGrant::Verify(0), 1)));
        let mut built = Launch::new(usize::MAX);
        built.build(reserved, &fronts, &mut BTreeMap::new(), |_, _| {}).unwrap();
        let rows = built
            .operations
            .iter()
            .map(|operation| (operation.request(), operation.row_count(), operation.acceptable_rows(), operation.chain().is_some()))
            .collect::<Vec<_>>();
        assert_eq!(rows, [(deep, 3, 2, false), (unguessed, 1, 0, false)], "no forward of the launch chains");
        // Its first guess stood and its second fell: the pairs of 5 and 9
        // trail the head.
        let Membership::Started(round) = built.members.remove(&deep).unwrap().state else {
            panic!("a decode round starts");
        };
        let transition = round.prepare_round_transition(deep, &[TokenId(5), TokenId(9), TokenId(11)], Vec::new(), Some(features()), &mut Zeros).unwrap();
        let generation = round.commit(transition).0;
        assert_eq!(generation.generated(), [TokenId(3), TokenId(4), TokenId(5), TokenId(9)]);
        assert_eq!(generation.front_guess(), Some(GuessKind::Head), "it drafts its chain first");
        let reserved = reserve(vec![(deep, generation)], 64);
        let fronts = FrontRows::of(reserved.iter().map(|(id, round)| (*id, round.front_guess())), 64, verifying).unwrap();
        let mut built = Launch::new(usize::MAX);
        built.build(reserved, &fronts, &mut BTreeMap::new(), |_, _| {}).unwrap();
        let [Operation::Head { tokens, proposals, .. }] = built.operations.as_slice() else {
            panic!("the head transaction: {:?}", built.operations);
        };
        assert_eq!((tokens.clone(), proposals.len()), (vec![TokenId(5), TokenId(9)], 2));
    }

    /// [`decoding`], then a first pass committed with `chain` standing.
    fn with_chain(id: RequestId, chain: &[u32]) -> Generation {
        let round = started(id, decoding(id, 32), 4, FrontGrant::UpTo(0));
        commit(id, round, &[TokenId(4)], chain.iter().map(|&token| guess(token, GuessKind::Chained)).collect())
    }

    /// The domain's grant of a pass the host builds in verify rows, every
    /// request that could front at depth 2.
    fn verifying_all(_: &[RequestId], fronted: &[RequestId]) -> Result<FrontPass, ()> {
        Ok(FrontPass::Built { form: BuiltForm::Verify, depths: fronted.iter().map(|&id| (id, 2)).collect() })
    }

    /// A launch the regime prices in verify rows builds every pass in that
    /// form even where its rows hold no guessed row: two requests of chains
    /// within two rows each verify their exact row alone, and neither
    /// forward chains a head behind it, as a front pass would.
    #[test]
    fn a_verify_launch_granting_no_guessed_row_builds_verify_passes() {
        let ids = [RequestId(1), RequestId(2)];
        let reserved = reserve(ids.iter().map(|&id| (id, with_chain(id, &[5, 6]))).collect(), 2);
        let fronts = FrontRows::of(reserved.iter().map(|(id, round)| (*id, round.front_guess())), 2, verifying_all).unwrap();
        assert!(grants_none(&fronts), "a row a request leaves no guessed row room");
        assert_eq!((fronts.grant(ids[0]), fronts.grant(ids[1])), ((FrontGrant::Verify(0), 1), (FrontGrant::Verify(0), 1)));
        let mut built = Launch::new(usize::MAX);
        built.build(reserved, &fronts, &mut BTreeMap::new(), |_, _| {}).unwrap();
        let rows = built
            .operations
            .iter()
            .map(|operation| (operation.request(), operation.row_count(), operation.chain().is_some()))
            .collect::<Vec<_>>();
        assert_eq!(rows, [(ids[0], 1, false), (ids[1], 1, false)], "verify passes, none chained");
    }

    /// Every round a launch reserved is built, whatever rows the rounds
    /// built before it took: a request whose verify pass left two pairs
    /// trailing its head enters both, a row past the one its grant counted,
    /// and the request after it within the launch's two rows still decodes
    /// its exact row rather than fail for want of a row.
    #[test]
    fn every_reserved_round_is_built_past_an_earlier_rounds_rows() {
        let (trailing, other) = (RequestId(1), RequestId(2));
        let reserved = reserve(vec![(trailing, with_chain(trailing, &[5, 6]))], 64);
        let fronts = FrontRows::of(reserved.iter().map(|(id, round)| (*id, round.front_guess())), 64, verifying_all).unwrap();
        let mut built = Launch::new(usize::MAX);
        built.build(reserved, &fronts, &mut BTreeMap::new(), |_, _| {}).unwrap();
        let Membership::Started(round) = built.members.remove(&trailing).unwrap().state else {
            panic!("a decode round starts");
        };
        // Its first guess stood and its second fell: the pairs of 5 and 9
        // trail the head.
        let transition = round.prepare_round_transition(trailing, &[TokenId(5), TokenId(9), TokenId(11)], Vec::new(), Some(features()), &mut Zeros).unwrap();
        let generation = round.commit(transition).0;
        let reserved = reserve(vec![(trailing, generation), (other, with_chain(other, &[]))], 2);
        let fronts = FrontRows::of(reserved.iter().map(|(id, round)| (*id, round.front_guess())), 2, verifying_all).unwrap();
        let mut built = Launch::new(usize::MAX);
        let mut returned = BTreeMap::new();
        built.build(reserved, &fronts, &mut returned, |_, _| {}).unwrap();
        assert!(returned.is_empty(), "no round is left out: {:?}", returned.keys().collect::<Vec<_>>());
        let rows = built.operations.iter().map(|operation| (operation.request(), operation.row_count())).collect::<Vec<_>>();
        assert_eq!(rows, [(trailing, 2), (other, 1)], "the head enters both pairs, and the other request its exact row");
    }

    /// A launch's forwards chain all or none: only a request decoding in
    /// front passes whose next round is a decode chains. A plain request's
    /// decode, a drafting request's, or a fronting request's prompt chunk
    /// does not, so its rows are never held beside chained ones.
    #[test]
    fn only_a_front_pass_chains() {
        assert!(chained(&State::Between(guessing(RequestId(1), 32), Stage::Ready)), "a front-decoding request's decode");
        let layout = || magnitude_generation::InputLayout::new(2, vec![]).unwrap();
        let mut plain = Generation::new(vec![TokenId(1), TokenId(2)], layout(), super::termination_tests::options(8, magnitude_generation::MethodChoice::Plain), None).unwrap();
        plain.resume_at(None).unwrap();
        let plain = commit(RequestId(2), started(RequestId(2), plain, 4, FrontGrant::UpTo(0)), &[TokenId(3)], Vec::new());
        assert!(!chained(&State::Between(plain, Stage::Ready)), "a plain request's decode");
        let options = || super::termination_tests::options(8, magnitude_generation::MethodChoice::Mtp { proposals: 2 });
        let drafting = Generation::new_with_method(vec![TokenId(1), TokenId(2)], layout(), options(), None, Arc::new(magnitude_generation::Mtp::new("fixture", 2).unwrap())).unwrap();
        assert!(!drafting.chains(), "a drafting request");
        let fronting = Arc::new(magnitude_generation::Mtp::new("fixture", 2).unwrap().fronting().unwrap());
        let mut prompting = Generation::new_with_method(vec![TokenId(1), TokenId(2)], layout(), options(), None, fronting).unwrap();
        prompting.resume_at(None).unwrap();
        assert!(!chained(&State::Between(prompting, Stage::Ready)), "a front-decoding request's prompt chunk");
    }

    #[test]
    fn follow_yields_preserves_pending_when_prefixes_evicted() {
        assert_eq!(super::combine_yielded(true, Yielded::Pending), Yielded::Pending);
        assert_eq!(super::combine_yielded(false, Yielded::Pending), Yielded::Pending);
        assert_eq!(
            super::combine_yielded(true, Yielded::Nothing),
            Yielded::Released { released: 0, returned: 0 }
        );
        assert_eq!(super::combine_yielded(false, Yielded::Nothing), Yielded::Nothing);
        assert_eq!(
            super::combine_yielded(true, Yielded::Released { released: 128, returned: 64 }),
            Yielded::Released { released: 128, returned: 64 }
        );
    }

    #[test]
    fn reserve_clamps_allowance_to_publication_queue_capacity() {
        let (sender, _receiver) = PublicationQueue::bounded(1, |_| {}).unwrap();
        let record = Record {
            publication: Some(ActivePublication::new(sender, 1)),
            ..record()
        };
        let generation = guessing(RequestId(1), 32);
        assert!(
            generation.publication_bound(4).unwrap() > 1,
            "unclamped allowance would require more slots than queue capacity"
        );
        let mut epoch = AvailabilityEpoch::default();
        match ReservedRound::reserve(record, generation, 4, 1, &mut epoch) {
            Reservation::Round(reserved) => {
                assert_eq!(reserved.allowance, 1);
                assert_eq!(reserved.record.publication.as_ref().unwrap().permits.len(), 1);
            }
            Reservation::Returned(_) => panic!("must clamp allowance instead of stalling in AwaitingCredit"),
            Reservation::Composed(..) => panic!("unexpected composed work"),
        }
    }
}

#[cfg(test)]
mod scoring_tests {
    use super::claim_tests::{NoWakes, leave_room, limits, method_of, take_room};
    use super::*;
    use crate::publication::Publication;
    use crate::scoring::{SCORING_SEQUENCES, SCORING_WIDTH};
    use magnitude_executor::{PlannedMethod, fixtures};
    use magnitude_generation::{EndOfGeneration, MethodChoice, Options, Sampling, Shaping, TokenId};
    use std::sync::{Mutex, mpsc};
    use std::task::{Context, Poll, Waker};

    /// The fixture's context, where a test names none: room for every score
    /// and generation of those tests.
    const CONTEXT: u64 = 64;
    /// The logits rows the exporting fixture exports a launch: its launch's.
    const EXPORTED: usize = 4;
    /// The tokens each generation here writes; its stream holds them all,
    /// as no output credit reaches the owner.
    const GENERATED: usize = 6;

    /// An owner over a CPU load, run as the worker runs it, a millisecond a
    /// run; the load's directory is removed once the owner is dropped.
    struct Served {
        owner: Option<Owner>,
        now: u64,
        directory: PathBuf,
        vocabulary: usize,
        /// The load's context, its generations' limit.
        context: usize,
        /// The method its generations take: the load's.
        method: MethodChoice,
    }

    impl Served {
        /// An owner over the planning fixture's decoder exporting up to
        /// `exported` rows of logits a launch.
        fn new(test: &str, exported: usize) -> Self {
            Self::chaining(test, 1, exported)
        }

        /// [`Self::new`], its prompt groups chaining up to `chain_capacity`
        /// chunks.
        fn chaining(test: &str, chain_capacity: usize, exported: usize) -> Self {
            Self::within(test, CONTEXT, chain_capacity, exported)
        }

        /// [`Self::chaining`] at a context of `context` rows.
        fn within(test: &str, context: u64, chain_capacity: usize, exported: usize) -> Self {
            Self::over(fixtures::loaded::exporting(test, context, chain_capacity, exported), MethodChoice::Plain)
        }

        /// An owner over the planning fixture's decoder with an MTP head,
        /// served by `method` ([`fixtures::loaded::drafted`]) and exporting
        /// up to `exported` rows of logits a launch; its generations draft
        /// as the load does.
        fn drafted(test: &str, method: PlannedMethod, exported: usize) -> Self {
            let generates = match method {
                PlannedMethod::Plain => MethodChoice::Plain,
                PlannedMethod::Mtp { greedy_proposals, sampled_proposals, .. } => {
                    assert_eq!(greedy_proposals, sampled_proposals, "the fixture's generations draft one width");
                    MethodChoice::Mtp { proposals: greedy_proposals }
                }
                PlannedMethod::DFlash { .. } => unreachable!("the drafted fixture drafts by its head"),
            };
            Self::over(fixtures::loaded::drafted(test, method, exported), generates)
        }

        /// An owner over a loaded fixture, its generations taking `method`.
        fn over(
            (directory, definition, domain, bindings): (PathBuf, Rc<magnitude_family_contracts::ModelDefinition>, Box<ExecutorDomain>, StateBindings),
            method: MethodChoice,
        ) -> Self {
            let owner = Owner::new(domain, bindings, limits(), Arc::new(NoWakes)).unwrap();
            let (vocabulary, context) = (definition.decoder.vocabulary as usize, definition.decoder.context_limit as usize);
            Self { owner: Some(owner), now: 0, directory, vocabulary, context, method }
        }

        fn owner(&mut self) -> &mut Owner {
            self.owner.as_mut().expect("the owner is between runs")
        }

        fn walks(&mut self) -> &Walks {
            self.owner().service.scoring.as_ref().expect("the load scores")
        }

        fn run(&mut self) -> Ran {
            let owner = self.owner.take().expect("the owner is between runs");
            let (owner, ran) = owner.run(self.now).unwrap_or_else(|error| panic!("the owner stopped: {error}"));
            self.owner = Some(owner);
            self.now += 1_000_000;
            ran
        }

        /// Run the owner until `done` holds.
        fn until(&mut self, mut done: impl FnMut(&mut Self) -> bool) {
            for _ in 0..10_000 {
                if done(self) {
                    return;
                }
                self.run();
            }
            panic!("the owner never settled");
        }

        /// `count` tokens within the vocabulary, varied by `salt`.
        fn tokens(&self, count: usize, salt: usize) -> Vec<u32> {
            (0..count).map(|index| ((index * 37 + salt * 11 + 5) % self.vocabulary) as u32).collect()
        }

        fn score(&mut self, tokens: &[u32], top: bool, next: bool) -> Answer {
            let (sender, receiver) = mpsc::channel();
            let request = ScoreRequest {
                tokens: tokens.iter().copied().map(TokenId).collect(),
                top,
                next,
                done: Box::new(move |outcome| sender.send(outcome).unwrap()),
            };
            let now = self.now;
            self.owner().score(request, now);
            Answer { receiver, answer: None }
        }

        /// The score `answer` comes to, the owner run until it is answered.
        fn answered(&mut self, mut answer: Answer) -> ScoreOutcome {
            self.until(|_| answer.poll());
            answer.outcome()
        }

        /// Admit a generation of `GENERATED` tokens after a prompt of
        /// `prompt` tokens, greedy, or sampled at temperature 1 within the
        /// top 0.9, by the load's method.
        fn admit(&mut self, prompt: usize, sampled: bool) -> Stream {
            let (sampling, shaping) = match sampled {
                false => (Sampling::Greedy, Shaping { temperature: 0.0, ..Default::default() }),
                true => (Sampling::Categorical, Shaping { temperature: 1.0, top_p: 0.9, ..Default::default() }),
            };
            let prompt = self.tokens(prompt, 1).into_iter().map(TokenId).collect::<Vec<_>>();
            let options = Options {
                max_tokens: GENERATED,
                output_capacity: GENERATED,
                context_limit: self.context,
                vocabulary: self.vocabulary,
                stop_tokens: BTreeSet::new(),
                suppressed_tokens: BTreeSet::new(),
                sampling,
                shaping,
                seed: 7,
                forced_quantum: 0,
                method: self.method,
                end_of_generation: EndOfGeneration::Stop,
                reasoning_budget: None,
            };
            let layout = InputLayout::new(prompt.len(), vec![]).unwrap();
            let generation = Generation::new_with_method(prompt.clone(), layout, options, None, method_of(self.method)).unwrap();
            let now = self.now;
            let (id, receiver) = self
                .owner()
                .admit(generation, PreparedModelInput::continuation_only(), PrefixRetention::Transient, GENERATED, now)
                .unwrap();
            Stream { id, prompt, receiver, tokens: Vec::new(), end: None, drafted: 0 }
        }
    }

    impl Drop for Served {
        fn drop(&mut self) {
            self.owner.take();
            std::fs::remove_dir_all(&self.directory).ok();
        }
    }

    /// A score's callback, as its host receives it.
    struct Answer {
        receiver: mpsc::Receiver<Result<ScoreOutcome, RequestError>>,
        answer: Option<Result<ScoreOutcome, RequestError>>,
    }

    impl Answer {
        /// Whether it is answered.
        fn poll(&mut self) -> bool {
            if self.answer.is_none() {
                self.answer = self.receiver.try_recv().ok();
            }
            self.answer.is_some()
        }

        /// Its answer, the one its callback gave: no second follows.
        fn result(mut self) -> Result<ScoreOutcome, RequestError> {
            assert!(self.poll(), "the score is answered");
            assert!(matches!(self.receiver.try_recv(), Err(mpsc::TryRecvError::Disconnected)), "a score is answered once");
            self.answer.take().unwrap()
        }

        fn outcome(self) -> ScoreOutcome {
            self.result().unwrap_or_else(|error| panic!("the score failed: {error}"))
        }
    }

    /// A generation's request, its prompt, the tokens its stream published,
    /// its end and the tokens it drafted, once completed.
    struct Stream {
        id: RequestId,
        prompt: Vec<TokenId>,
        receiver: PublicationReceiver,
        tokens: Vec<TokenId>,
        end: Option<Result<FinishReason, RequestError>>,
        drafted: usize,
    }

    impl Stream {
        /// Take what the stream published; whether it ended.
        fn poll(&mut self) -> bool {
            while self.end.is_none() {
                match self.receiver.poll_next(&mut Context::from_waker(Waker::noop())) {
                    Poll::Ready(Some(Publication::Output(output))) => self.tokens.extend(output.iter().map(|output| output.token)),
                    Poll::Ready(Some(Publication::Completed { finish, usage, .. })) => {
                        self.end = Some(Ok(finish));
                        self.drafted = usage.draft_n;
                    }
                    Poll::Ready(Some(Publication::Failed(error))) => self.end = Some(Err(error)),
                    Poll::Ready(None) => panic!("a stream ends with a terminal publication"),
                    Poll::Pending => break,
                }
            }
            self.end.is_some()
        }
    }

    /// A score of n tokens on an idle owner gives n - 1 positions, each a
    /// log-probability and a rank, the top token where it differs and the
    /// next token where asked; the same tokens score the same again. Its
    /// walk is no request's: no status names it, and no relief evicts it.
    #[test]
    fn a_score_on_an_idle_owner_scores_every_token_after_the_first() {
        let mut served = Served::new("score-on-an-idle-owner", EXPORTED);
        let tokens = served.tokens(7, 0);
        let first = served.score(&tokens, true, true);
        assert!(served.run().in_flight, "the score's first flight is submitted");
        let walk = served.walks().active.first().map(Walk::sequence).expect("the score's walk is open");
        assert!(served.owner().snapshot(walk).is_none(), "no status names a walk");
        assert!(!MemoryHost::evictable(&served.owner().service).contains(&walk), "no relief evicts a walk");
        let first = served.answered(first);
        assert_eq!(first.positions.len(), tokens.len() - 1);
        for position in &first.positions {
            assert!(position.logprob <= 0.0 && position.rank >= 1, "{position:?}");
            if let Some(top) = position.top {
                assert!(top.logprob >= position.logprob && position.rank > 1, "{position:?}");
            }
        }
        assert!(first.next.is_some(), "the next token was asked for");
        let again = served.score(&tokens, true, true);
        assert_eq!(served.answered(again), first, "the same tokens score the same");
        assert!(served.walks().active.is_empty() && served.walks().unclosed.is_empty());
        let closed = served.owner().inspect_domain(|domain| Ok(!domain.holds_state_beside(&BTreeSet::new()))).unwrap();
        assert!(closed, "an answered walk's sequence is closed");
    }

    /// A score submitted while a greedy request decodes, and one beside a
    /// sampled request, complete in launches of their own, a round and a
    /// scoring flight in turn: each request's tokens are those it generates
    /// alone. A score of the greedy request's prompt and output ranks each
    /// token it generated first, and names the last as the next.
    #[test]
    fn scores_beside_greedy_and_sampled_generation_leave_their_tokens_unchanged() {
        let mut served = Served::new("scores-beside-generation", EXPORTED);
        for sampled in [false, true] {
            let mut alone = served.admit(5, sampled);
            served.until(|_| alone.poll());
            assert_eq!(alone.end, Some(Ok(FinishReason::Length)));
            assert_eq!(alone.tokens.len(), GENERATED);
            let mut beside = served.admit(5, sampled);
            served.until(|_| {
                beside.poll();
                !beside.tokens.is_empty()
            });
            assert!(beside.end.is_none(), "the request decodes as the score arrives");
            let tokens = served.tokens(9, 2);
            let mut score = served.score(&tokens, false, true);
            // Whether each flight while both have work is a scoring one.
            let mut flights = Vec::new();
            served.until(|served| {
                let (ended, answered) = (beside.poll(), score.poll());
                if !ended && !answered {
                    flights.push(matches!(served.owner().pipeline, Pipeline::Scoring(_)));
                }
                ended && answered
            });
            assert!(flights.contains(&true) && flights.windows(2).all(|pair| pair[0] != pair[1]), "rounds and scoring flights take turns: {flights:?}");
            assert_eq!(beside.end, Some(Ok(FinishReason::Length)));
            assert_eq!(beside.tokens, alone.tokens, "sampled {sampled}: generation beside a score is unchanged");
            assert_eq!(score.outcome().positions.len(), tokens.len() - 1);
            if sampled {
                continue;
            }
            let written = alone.prompt.iter().chain(&alone.tokens[..GENERATED - 1]).map(|token| token.0).collect::<Vec<_>>();
            let scored = served.score(&written, false, true);
            let scored = served.answered(scored);
            let ranks = scored.positions[alone.prompt.len() - 1..].iter().map(|position| position.rank).collect::<Vec<_>>();
            assert_eq!(ranks, vec![1; GENERATED - 1], "each greedy token is its row's most likely");
            assert_eq!(scored.next, alone.tokens.last().copied(), "the next token is the greedy one");
        }
    }

    /// A drafting load (an MTP head over the decoder, as the service's
    /// `Auto` serves a model with one) scores as a plain load of the same
    /// model does: alone, and beside MTP generation, greedy and sampled,
    /// whose tokens are those it generates alone. Each walk's sequence
    /// closes with its head's state.
    #[test]
    fn scores_on_a_drafting_load_are_a_plain_loads() {
        let mut plain = Served::drafted("scores-on-the-model-plain", PlannedMethod::Plain, EXPORTED);
        let mut drafting = Served::drafted("scores-on-a-drafting-load", fixtures::loaded::MTP, EXPORTED);
        let width = drafting.walks().shape.width;
        // Two walks, one over several flights: flights of both, then of one.
        let tokens = [drafting.tokens(2 * width + 3, 6), drafting.tokens(5, 7)];
        let scored = |served: &mut Served| tokens.clone().map(|tokens| served.score(&tokens, true, true)).map(|answer| served.answered(answer));
        let occupied = |served: &mut Served| {
            let bytes = served.owner().state_bytes();
            (bytes.target.occupied, bytes.head.map(|head| head.occupied))
        };
        let reference = scored(&mut plain);
        let idle = occupied(&mut drafting);
        assert!(idle.1.is_some(), "the drafting load keeps its head's history");
        assert_eq!(scored(&mut drafting), reference, "scores alone on the drafting load");
        assert_eq!(occupied(&mut drafting), idle, "the walks' sequences closed with their heads' state");
        for sampled in [false, true] {
            let mut alone = drafting.admit(5, sampled);
            drafting.until(|_| alone.poll());
            assert_eq!(alone.end, Some(Ok(FinishReason::Length)));
            assert_eq!(alone.tokens.len(), GENERATED);
            let mut beside = drafting.admit(5, sampled);
            drafting.until(|_| {
                beside.poll();
                !beside.tokens.is_empty()
            });
            assert!(beside.end.is_none(), "the request drafts as the scores arrive");
            let mut answers = tokens.clone().map(|tokens| drafting.score(&tokens, true, true));
            let mut scored_beside = false;
            drafting.until(|served| {
                let ended = beside.poll();
                scored_beside |= !ended && matches!(served.owner().pipeline, Pipeline::Scoring(_));
                ended & answers.iter_mut().all(Answer::poll)
            });
            assert!(scored_beside, "sampled {sampled}: a scoring flight runs while the request drafts");
            assert_eq!(beside.end, Some(Ok(FinishReason::Length)));
            assert!(alone.drafted > 0 && beside.drafted > 0, "sampled {sampled}: each generation drafts by the head");
            assert_eq!(beside.tokens, alone.tokens, "sampled {sampled}: drafting beside scores is unchanged");
            assert_eq!(answers.map(Answer::outcome), reference, "sampled {sampled}: scores beside drafting");
        }
        assert_eq!(occupied(&mut drafting), idle, "every sequence closed with its head's state");
    }

    /// More scores than a flight walks wait in arrival order and complete,
    /// each answered once; scores of equal length are answered as they
    /// arrived, and no more walks than a flight's are ever open.
    #[test]
    fn more_scores_than_walk_at_once_each_complete_once_in_arrival_order() {
        let mut served = Served::new("more-scores-than-walk", EXPORTED);
        let order = Arc::new(Mutex::new(Vec::new()));
        let answers = (0..SCORING_SEQUENCES + 2)
            .map(|index| {
                let (sender, receiver) = mpsc::channel();
                let order = Arc::clone(&order);
                let request = ScoreRequest {
                    tokens: served.tokens(5, index).into_iter().map(TokenId).collect(),
                    top: false,
                    next: false,
                    done: Box::new(move |outcome| {
                        order.lock().unwrap().push(index);
                        sender.send(outcome).unwrap()
                    }),
                };
                let now = served.now;
                served.owner().score(request, now);
                Answer { receiver, answer: None }
            })
            .collect::<Vec<_>>();
        let sequences = served.walks().shape.sequences;
        served.until(|served| {
            assert!(served.walks().active.len() <= sequences, "at most a flight's walks are open");
            order.lock().unwrap().len() == answers.len()
        });
        assert_eq!(*order.lock().unwrap(), (0..answers.len()).collect::<Vec<_>>());
        for answer in answers {
            assert_eq!(answer.outcome().positions.len(), 4);
        }
    }

    /// A score over several flights scores each row as its prefix's score
    /// does: the walk's flights concatenate.
    #[test]
    fn a_score_over_several_flights_agrees_with_its_prefix() {
        let mut served = Served::new("score-over-several-flights", EXPORTED);
        let prefix = SCORING_WIDTH + 2;
        let long = served.tokens(prefix + 3, 3);
        assert!(long.len() > 2 * served.walks().shape.width, "the long score takes several flights");
        let (whole, part) = (served.score(&long, true, true), served.score(&long[..prefix], true, true));
        let (whole, part) = (served.answered(whole), served.answered(part));
        assert_eq!(whole.positions.len(), long.len() - 1);
        assert_eq!(part.positions.len(), prefix - 1);
        for (index, (whole, part)) in whole.positions.iter().zip(&part.positions).enumerate() {
            assert_eq!((whole.rank, whole.top.map(|top| top.token)), (part.rank, part.top.map(|top| top.token)), "position {index}");
            assert!((whole.logprob - part.logprob).abs() < 1e-5, "position {index}: {} and {}", whole.logprob, part.logprob);
        }
    }

    /// Termination answers every waiting and open score once, with its
    /// error; the flight its walks were in reconciles, and every walk's
    /// sequence closes.
    #[test]
    fn terminate_all_answers_waiting_and_active_scores_once() {
        let mut served = Served::new("terminate-scores", EXPORTED);
        let shape = served.walks().shape;
        let answers = (0..shape.sequences + 2)
            .map(|index| {
                let tokens = served.tokens(3 * shape.width + 1, index);
                served.score(&tokens, false, false)
            })
            .collect::<Vec<_>>();
        assert!(served.run().in_flight, "a scoring flight is out");
        assert!(matches!(served.owner().pipeline, Pipeline::Scoring(_)));
        assert_eq!((served.walks().active.len(), served.walks().waiting.len()), (shape.sequences, 2));
        served.owner().terminate_all(RequestError::WorkerClosed);
        for answer in answers {
            assert_eq!(answer.result(), Err(RequestError::WorkerClosed));
        }
        served.until(|served| !served.run().in_flight);
        served.run();
        let closed = served.owner().inspect_domain(|domain| Ok(!domain.holds_state_beside(&BTreeSet::new()))).unwrap();
        assert!(closed, "every walk's sequence closed");
        assert!(served.walks().unclosed.is_empty());
    }

    /// A memory-pressure unload (the host's, as the memory step makes it on
    /// a shortage that persists) while a scoring flight is out answers every
    /// open and waiting score once, with the unload's error. Drained as the
    /// worker drains an unloading owner, until no flight is out, the owner
    /// reconciles the flight and closes every walk's sequence.
    #[test]
    fn a_memory_pressure_unload_answers_every_score_once() {
        let mut served = Served::new("unload-answers-scores", EXPORTED);
        let shape = served.walks().shape;
        let answers = (0..shape.sequences + 2)
            .map(|index| {
                let tokens = served.tokens(3 * shape.width + 1, index);
                served.score(&tokens, false, false)
            })
            .collect::<Vec<_>>();
        assert!(served.run().in_flight && matches!(served.owner().pipeline, Pipeline::Scoring(_)), "a scoring flight is out");
        assert_eq!((served.walks().active.len(), served.walks().waiting.len()), (shape.sequences, 2));
        MemoryHost::unload(&mut served.owner().service);
        let unloaded = RequestError::ModelUnloaded { cause: ModelUnloadCause::MemoryPressure };
        for answer in answers {
            assert_eq!(answer.result(), Err(unloaded.clone()));
        }
        let mut ran = served.run();
        assert!(ran.unload, "the run reports the unload");
        while ran.in_flight {
            ran = served.run();
        }
        let closed = served.owner().inspect_domain(|domain| Ok(!domain.holds_state_beside(&BTreeSet::new()))).unwrap();
        assert!(closed, "every walk's sequence closed");
        assert!(served.walks().active.is_empty() && served.walks().waiting.is_empty() && served.walks().unclosed.is_empty());
    }

    /// Termination between scoring flights, a chat's round in flight beside
    /// open walks, answers every open and waiting score once with its error,
    /// and fails the chat with it. No flight holds the walks' sequences:
    /// they close at the owner's next idle boundary.
    #[test]
    fn terminate_all_between_scoring_flights_answers_each_score_once() {
        let mut served = Served::new("terminate-between-scoring-flights", EXPORTED);
        let shape = served.walks().shape;
        let mut chat = served.admit(5, false);
        served.until(|_| {
            chat.poll();
            !chat.tokens.is_empty()
        });
        let answers = (0..shape.sequences + 1)
            .map(|index| {
                let tokens = served.tokens(3 * shape.width + 1, index);
                served.score(&tokens, false, false)
            })
            .collect::<Vec<_>>();
        served.until(|served| matches!(served.owner().pipeline, Pipeline::Round(..)) && !served.walks().active.is_empty());
        assert!(chat.end.is_none() && !served.walks().waiting.is_empty(), "walks open and wait beside the chat's round");
        served.owner().terminate_all(RequestError::WorkerClosed);
        for answer in answers {
            assert_eq!(answer.result(), Err(RequestError::WorkerClosed));
        }
        assert!(!served.walks().unclosed.is_empty(), "the open walks' sequences close at the next idle boundary");
        served.until(|served| !served.run().in_flight);
        assert!(chat.poll());
        assert_eq!(chat.end, Some(Err(RequestError::WorkerClosed)));
        let closed = served.owner().inspect_domain(|domain| Ok(!domain.holds_state_beside(&BTreeSet::new()))).unwrap();
        assert!(closed, "every walk's sequence closed");
        assert!(served.walks().unclosed.is_empty());
    }

    /// With room beside the load for one walk's open, two walks open (the
    /// first in the load's request slot) and their flight's export slot does
    /// not fit: the newest walk is deferred, its progress dropped, and the
    /// room its open held lets the first fly alone. The deferred walk opens
    /// again past that flight's boundary, and both complete.
    #[test]
    fn a_flight_short_of_memory_defers_its_newest_walk() {
        let mut served = Served::new("score-defers-its-newest-walk", EXPORTED);
        let tokens = [served.tokens(5, 1), served.tokens(5, 2)];
        let open = {
            let domain = &served.owner().service.domain;
            fixtures::reserved_bytes(domain, fixtures::opening_rows(domain, 5))
        };
        leave_room(&mut served.owner().service, open);
        let mut answers = tokens.map(|tokens| served.score(&tokens, false, false));
        assert!(served.run().in_flight, "a flight is out");
        let walks = served.walks();
        assert_eq!((walks.active.len(), walks.waiting.len()), (1, 1), "the newest walk is deferred");
        assert!(walks.waits.is_some(), "it opens again only past a change of ours");
        served.until(|_| answers.iter_mut().all(Answer::poll));
        for answer in answers {
            assert_eq!(answer.outcome().positions.len(), 4);
        }
    }

    /// A flight short of memory sheds its own walks, the newest first,
    /// before any relief evicts generation's state for them. Two chats are
    /// resident between their rounds and evictable, the second's state
    /// beside the load's request slot; two walks' opens take the last room,
    /// and their flight's export claim does not fit: the newest walk is
    /// deferred, and both chats stay resident, neither preempted.
    /// Everything completes.
    #[test]
    fn a_flight_short_of_memory_sheds_walks_before_it_evicts_a_chat() {
        let mut served = Served::new("score-sheds-walks-before-a-chat", EXPORTED);
        let mut chats = [served.admit(5, false), served.admit(5, false)];
        served.until(|served| {
            let decoding = chats.iter_mut().all(|chat| !chat.poll() && !chat.tokens.is_empty());
            decoding && matches!(served.owner().pipeline, Pipeline::Round(..))
        });
        let open = {
            let domain = &served.owner().service.domain;
            fixtures::reserved_bytes(domain, fixtures::opening_rows(domain, 5))
        };
        leave_room(&mut served.owner().service, 2 * open);
        let tokens = [served.tokens(5, 1), served.tokens(5, 2)];
        let mut answers = tokens.map(|tokens| served.score(&tokens, false, false));
        // The chats' round reconciles, and scoring takes its turn.
        assert!(served.run().in_flight);
        assert!(matches!(served.owner().pipeline, Pipeline::Scoring(_)), "scoring takes the turn after the chats' round");
        let walks = served.walks();
        assert_eq!((walks.active.len(), walks.waiting.len()), (1, 1), "the newest walk is deferred");
        for chat in &chats {
            let request = &served.owner().service.requests[&chat.id];
            assert_eq!(request.record.preemption_debt, 0, "no relief preempts a chat for the walks");
            assert!(request.state.generation().is_resident(), "the chat stays resident");
        }
        served.until(|_| chats.iter_mut().all(Stream::poll) & answers.iter_mut().all(Answer::poll));
        for chat in chats {
            assert_eq!(chat.end, Some(Ok(FinishReason::Length)));
        }
        for answer in answers {
            assert_eq!(answer.outcome().positions.len(), 4);
        }
    }

    /// A chat waiting on memory opens before new walks: no walk opens while
    /// a generation request waits on memory, so the room closing walks free
    /// is the chat's first, and scores arriving all the while never starve
    /// it. Two walks hold the load's request slot and the room beside it,
    /// their ends a flight apart. A chat waits whose open needs the room of
    /// three walks: what one walk's end frees, with the flight's export
    /// claim the chat's open may take, is two walks' room. Scores keep
    /// arriving, each walking two flights, so walks would end one a flight,
    /// each end's room taken by a new walk. No walk opens while the chat
    /// waits; it is admitted once the open walks end, and completes, and so
    /// does every score.
    #[test]
    fn a_chat_waiting_on_memory_opens_before_new_walks() {
        let mut served = Served::within("chat-opens-before-new-walks", 16 * CONTEXT, 1, EXPORTED);
        // Walks of two flights, the first of three.
        let width = served.walks().shape.width;
        let (walk, first) = (2 * width, 3 * width);
        // The shortest prompt whose open needs the room of three walks' opens.
        let context = served.context;
        let prompt = {
            let domain = &served.owner().service.domain;
            let bytes = |rows: usize| fixtures::reserved_bytes(domain, fixtures::opening_rows(domain, rows as u64));
            assert_eq!(bytes(walk), bytes(first), "every walk opens in the same room");
            (walk..context).find(|&rows| bytes(rows) == 3 * bytes(walk)).expect("a prompt within the context needs three walks' room")
        };
        assert!(prompt + GENERATED <= context);
        let mut answers = Vec::new();
        let mut score = |served: &mut Served, count: usize| {
            let tokens = served.tokens(count, answers.len());
            answers.push((count, served.score(&tokens, false, false)));
        };
        score(&mut served, first);
        score(&mut served, walk);
        assert!(served.run().in_flight && matches!(served.owner().pipeline, Pipeline::Scoring(_)));
        assert_eq!(served.walks().active.len(), 2, "both walks fly");
        // The flight out holds its export claim: what is taken is the room
        // left beside the walks.
        take_room(&mut served.owner().service);
        let mut chat = served.admit(prompt, false);
        let waits = |served: &mut Served, chat: &Stream| {
            let request = served.owner().service.requests.get(&chat.id);
            request.is_some_and(|request| matches!(request.state.wait(), Some(Wait { on: WaitsOn::Memory, .. })))
        };
        let mut waited = false;
        for _ in 0..2_000 {
            if chat.poll() {
                break;
            }
            // Scores keep arriving: a flight's worth always waits.
            while served.walks().waiting.len() < served.walks().shape.sequences {
                score(&mut served, walk);
            }
            let (waiting, open) = (waits(&mut served, &chat), served.walks().sequences());
            served.run();
            waited |= waiting;
            let opened = !served.walks().sequences().is_subset(&open);
            assert!(!(waiting && opened && waits(&mut served, &chat)), "a walk opened while the chat waited on memory");
        }
        assert!(waited, "the chat waited on memory beside the walks");
        assert_eq!(chat.end, Some(Ok(FinishReason::Length)), "the chat is admitted beside arriving scores, and completes");
        served.until(|_| answers.iter_mut().all(|(_, answer)| answer.poll()));
        for (count, answer) in answers {
            assert_eq!(answer.outcome().positions.len(), count - 1);
        }
    }

    /// Scoring takes no turn while a prompt chain runs, whose chunks any
    /// other group's provisioning would conclude: a score arriving mid-chain
    /// walks once the chain ends, and both complete.
    #[test]
    fn a_score_takes_no_turn_while_a_prompt_chain_runs() {
        let mut served = Served::chaining("score-beside-a-chain", 3, EXPORTED);
        let mut chained = served.admit(16, false);
        served.until(|served| served.owner().service.domain.chain_holder().is_some());
        let tokens = served.tokens(5, 5);
        let mut score = served.score(&tokens, false, false);
        let mut chained_flights = 0;
        served.until(|served| {
            let owner = served.owner();
            let chain = owner.service.domain.chain_holder().is_some();
            assert!(!(chain && matches!(owner.pipeline, Pipeline::Scoring(_))), "a scoring flight runs while a chain is under way");
            chained_flights += usize::from(chain && !score.poll());
            chained.poll() & score.poll()
        });
        assert!(chained_flights > 0, "the score arrived mid-chain");
        assert_eq!(chained.end, Some(Ok(FinishReason::Length)));
        assert_eq!(score.outcome().positions.len(), tokens.len() - 1);
    }

    /// A load that exports no logits refuses a score at once, as invalid.
    #[test]
    fn an_owner_exporting_no_logits_refuses_a_score_at_once() {
        let mut served = Served::new("score-without-logits", 0);
        assert!(served.owner().service.scoring.is_none());
        let mut answer = served.score(&[1, 2, 3], false, true);
        assert!(answer.poll(), "answered before any run");
        assert!(matches!(answer.result(), Err(RequestError::Input(_))));
    }

    /// With every byte of room taken, a score whose flight cannot fit even
    /// alone fails with the shortage, one score at a time, the owner never
    /// stopping; once the room is free the next score completes.
    #[test]
    fn a_score_that_cannot_fit_alone_fails_with_the_shortage() {
        let mut served = Served::new("score-short-alone", EXPORTED);
        take_room(&mut served.owner().service);
        let tokens = served.tokens(5, 4);
        let mut answers = [served.score(&tokens, false, false), served.score(&tokens, false, false)];
        // The first walk opens in the load's request slot, and the second's
        // open waits beside it; the first fails alone, and the second opens
        // in its place within the same run, no event needed to wake it.
        served.run();
        assert!(answers.iter_mut().all(Answer::poll), "both fail within one run");
        for answer in answers {
            let failed = answer.result();
            assert!(matches!(failed, Err(RequestError::Capacity(_))), "{failed:?}");
        }
        served.owner().service.domain.release_measurement_room();
        let next = served.score(&tokens, false, false);
        assert_eq!(served.answered(next).positions.len(), tokens.len() - 1);
    }
}
