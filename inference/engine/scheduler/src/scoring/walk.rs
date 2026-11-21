//! Score walks, device-free: one score's walk through its tokens (the rows
//! each scoring flight forwards of it, and what it takes from a flight's
//! rows), the shape of a load's scoring flights, and the owner's walks,
//! active and waiting.

use super::{argmax, reduce_row, ScoreOutcome, ScoredPosition, SCORING_SEQUENCES, SCORING_WIDTH};
use crate::policy::AvailabilityEpoch;
use crate::protocol::ScoreRequest;
use crate::publication::RequestError;
use magnitude_executor::{Demand, InvariantError, Operation, RequestId, ResourceLimits, TokenId, WorkKind};
use magnitude_family_contracts::ModelDefinition;
use std::collections::{BTreeSet, VecDeque};
use std::ops::Range;

/// What a score's tokens must keep within: the load's context, and the
/// model's vocabulary.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bounds {
    pub(crate) context: usize,
    pub(crate) vocabulary: usize,
}

impl Bounds {
    /// The bounds of a load of `definition`.
    pub(crate) fn of(definition: &ModelDefinition) -> Result<Self, String> {
        let host = |value: u64, what: &str| {
            usize::try_from(value).map_err(|_| format!("the model's {what} exceeds the host domain"))
        };
        Ok(Self {
            context: host(definition.decoder.context_limit, "context")?,
            vocabulary: host(definition.decoder.vocabulary, "vocabulary")?,
        })
    }

    /// Why a score of `tokens` cannot be walked, if it cannot.
    pub(crate) fn refusal(self, tokens: &[TokenId]) -> Option<RequestError> {
        if tokens.is_empty() {
            return Some(RequestError::Input("a score needs at least one token".into()));
        }
        if tokens.len() > self.context {
            return Some(RequestError::Input(format!(
                "the score has {} tokens but the load's context is {}",
                tokens.len(),
                self.context
            )));
        }
        let outside = |token: &&TokenId| usize::try_from(token.0).map_or(true, |id| id >= self.vocabulary);
        tokens.iter().find(outside).map(|token| {
            RequestError::Input(format!(
                "token {} is outside the model's vocabulary of {}",
                token.0, self.vocabulary
            ))
        })
    }
}

/// A load's scoring flights: up to `sequences` walks each, `width` rows of
/// each walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shape {
    pub(crate) sequences: usize,
    pub(crate) width: usize,
}

impl Shape {
    /// The widest flights a load within `limits` launches, within
    /// [`SCORING_SEQUENCES`] walks of [`SCORING_WIDTH`] rows: every row
    /// exported, each walk a request of the launch, and the launch within
    /// its rows. None where the load exports no logits.
    pub(crate) fn of(limits: ResourceLimits) -> Option<Self> {
        let rows = limits.exported_logits_rows.min(limits.max_launch_rows);
        let sequences = SCORING_SEQUENCES.min(limits.max_launch_slots).min(rows);
        (sequences > 0).then(|| Self { sequences, width: SCORING_WIDTH.min(rows / sequences) })
    }
}

/// One opened score's walk through its tokens.
pub(crate) struct Walk {
    request: ScoreRequest,
    sequence: RequestId,
    /// The first row not yet forwarded; every row before it is scored.
    next_row: usize,
    positions: Vec<ScoredPosition>,
    /// The most likely token after the last one, once its row is read.
    next: Option<TokenId>,
}

/// What one flight left of a walk.
pub(crate) enum Absorbed {
    Continues,
    Ended(Result<ScoreOutcome, RequestError>),
}

impl Walk {
    /// A fresh walk of `request` on `sequence`: nothing forwarded yet.
    pub(crate) fn new(request: ScoreRequest, sequence: RequestId) -> Self {
        let positions = Vec::with_capacity(request.tokens.len().saturating_sub(1));
        Self { request, sequence, next_row: 0, positions, next: None }
    }

    pub(crate) fn sequence(&self) -> RequestId {
        self.sequence
    }

    /// The request, its progress dropped: a deferred walk starts again.
    pub(crate) fn into_request(self) -> ScoreRequest {
        self.request
    }

    /// Answer the request: the walk's one call of its callback.
    pub(crate) fn answer(self, result: Result<ScoreOutcome, RequestError>) {
        (self.request.done)(result)
    }

    /// The walk's part of its next flight: up to `width` of its next rows
    /// teacher-forced as one decode, every row demanding its logits and none
    /// selecting. Row `p`'s logits predict token `p + 1`, so a walk of `n`
    /// tokens scores `n - 1` positions, and its last row's logits give the
    /// next token.
    pub(crate) fn forward(&self, width: usize) -> Operation {
        let rows = self.next_row..(self.next_row + width).min(self.request.tokens.len());
        let tokens = self.request.tokens[rows.clone()].to_vec();
        Operation::Forward {
            request: self.sequence,
            kind: WorkKind::Decode,
            position: rows.start,
            conditioning: None,
            demand: Demand::LOGITS,
            select: Vec::new(),
            committed: tokens.len(),
            tokens,
            prime: None,
        }
    }

    /// Whether the walk reads the logits of its row at `position`: the row
    /// predicts one of its tokens, or it is the last row and the next token
    /// is asked for.
    pub(crate) fn reads(&self, position: usize) -> bool {
        position + 1 < self.request.tokens.len() || (self.request.next && position + 1 == self.request.tokens.len())
    }

    /// Take the `rows` a flight `forwarded` of the walk, each the logits the
    /// walk reads of it ([`Self::reads`]): each row that predicts a token
    /// scores it, and the last row's most likely token is the next one where
    /// it is asked for. A row the reduction refuses ends this walk alone;
    /// rows that break the flight's shape are an error.
    pub(crate) fn absorb(&mut self, forwarded: Range<usize>, rows: Vec<Option<Vec<f32>>>) -> Result<Absorbed, String> {
        if forwarded.start != self.next_row || forwarded.is_empty() || forwarded.end > self.request.tokens.len() {
            return Err(format!(
                "a flight forwarded rows {forwarded:?} of a walk whose next row is {} of {}",
                self.next_row,
                self.request.tokens.len()
            ));
        }
        if rows.len() != forwarded.len() {
            return Err(format!(
                "a flight returned {} rows of a walk it forwarded {} rows of",
                rows.len(),
                forwarded.len()
            ));
        }
        let ended = |detail: String| {
            Ok(Absorbed::Ended(Err(RequestError::Invariant(InvariantError { context: "scoring a logits row", detail }))))
        };
        let end = forwarded.end;
        for (position, row) in forwarded.zip(rows) {
            if let Some(&real) = self.request.tokens.get(position + 1) {
                let logits = row.ok_or("a scoring row returned no logits")?;
                match reduce_row(&logits, real, self.request.top) {
                    Ok(scored) => self.positions.push(scored),
                    Err(detail) => return ended(detail),
                }
            } else if self.request.next {
                let logits = row.ok_or("the next-token row returned no logits")?;
                let next = argmax(&logits).and_then(|(index, _)| {
                    u32::try_from(index).map_err(|_| "argmax exceeds the token id domain".to_owned())
                });
                match next {
                    Ok(token) => self.next = Some(TokenId(token)),
                    Err(detail) => return ended(detail),
                }
            }
        }
        self.next_row = end;
        Ok(if end == self.request.tokens.len() {
            Absorbed::Ended(Ok(ScoreOutcome { positions: std::mem::take(&mut self.positions), next: self.next }))
        } else {
            Absorbed::Continues
        })
    }
}

/// Whose flight the pipeline takes next where generation and scoring both
/// have work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Turn {
    Generation,
    Scoring,
}

/// A load's score walks: those open, each holding a sequence of the domain,
/// and the scores waiting to open, in arrival order.
pub(crate) struct Walks {
    pub(crate) shape: Shape,
    pub(crate) bounds: Bounds,
    pub(crate) waiting: VecDeque<ScoreRequest>,
    /// Open walks, in arrival order: at most `shape.sequences`.
    pub(crate) active: Vec<Walk>,
    /// Walks short of memory wait for a change of ours past this: until the
    /// epoch moves past it, no walk opens and no scoring flight forms.
    pub(crate) waits: Option<AvailabilityEpoch>,
    /// The sequences of walks answered by termination, which a flight may
    /// still hold: closed at the owner's next idle boundary.
    pub(crate) unclosed: Vec<RequestId>,
    pub(crate) turn: Turn,
}

impl Walks {
    /// No walk yet, for flights of `shape` within `bounds`.
    pub(crate) fn new(shape: Shape, bounds: Bounds) -> Self {
        Self {
            shape,
            bounds,
            waiting: VecDeque::new(),
            active: Vec::with_capacity(shape.sequences),
            waits: None,
            unclosed: Vec::new(),
            turn: Turn::Generation,
        }
    }

    /// The next scoring flight's operations: one forward of each open walk,
    /// in arrival order.
    pub(crate) fn forwards(&self) -> Vec<Operation> {
        self.active.iter().map(|walk| walk.forward(self.shape.width)).collect()
    }

    /// Whether no wait stands at `epoch`: none was set, or the epoch moved
    /// past it, which ends it.
    pub(crate) fn waited(&mut self, epoch: AvailabilityEpoch) -> bool {
        if self.waits.is_some_and(|at| !epoch.changed_since(at)) {
            return false;
        }
        self.waits = None;
        true
    }

    /// The open walks' sequences.
    pub(crate) fn sequences(&self) -> BTreeSet<RequestId> {
        self.active.iter().map(Walk::sequence).collect()
    }

    /// Defer the newest open walk: its request returns to the front of the
    /// queue, its progress dropped. The sequence it held, to close.
    pub(crate) fn defer_newest(&mut self) -> Option<RequestId> {
        let walk = self.active.pop()?;
        let sequence = walk.sequence();
        self.waiting.push_front(walk.into_request());
        Some(sequence)
    }

    /// Answer every open and waiting score with `error`, once: the owner
    /// terminates its work. The open walks' sequences are left to close.
    pub(crate) fn fail_all(&mut self, error: &RequestError) {
        for walk in self.active.drain(..) {
            self.unclosed.push(walk.sequence());
            walk.answer(Err(error.clone()));
        }
        for request in self.waiting.drain(..) {
            (request.done)(Err(error.clone()));
        }
        self.waits = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    type Outcomes = mpsc::Receiver<Result<ScoreOutcome, RequestError>>;

    fn tokens(ids: impl IntoIterator<Item = u32>) -> Vec<TokenId> {
        ids.into_iter().map(TokenId).collect()
    }

    fn request(tokens: Vec<TokenId>, top: bool, next: bool) -> (ScoreRequest, Outcomes) {
        let (sender, receiver) = mpsc::channel();
        let request = ScoreRequest { tokens, top, next, done: Box::new(move |outcome| sender.send(outcome).unwrap()) };
        (request, receiver)
    }

    fn walk(ids: impl IntoIterator<Item = u32>, top: bool, next: bool) -> (Walk, Outcomes) {
        let (request, outcomes) = request(tokens(ids), top, next);
        (Walk::new(request, RequestId(7)), outcomes)
    }

    /// A logits row of `vocabulary` entries peaking at `peak`.
    fn logits(vocabulary: usize, peak: usize) -> Vec<f32> {
        (0..vocabulary).map(|index| if index == peak { 4.0 } else { index as f32 * 0.25 }).collect()
    }

    fn input(error: Option<RequestError>) -> String {
        match error {
            Some(RequestError::Input(reason)) => reason,
            other => panic!("expected an input refusal, got {other:?}"),
        }
    }

    fn limits(exported: usize, rows: usize, slots: usize) -> ResourceLimits {
        ResourceLimits {
            max_launch_rows: rows,
            max_launch_slots: slots,
            max_selected_rows: 2,
            max_drafting_slots: 2,
            exported_logits_rows: exported,
            max_images_per_request: 0,
            max_image_cells: 0,
            lookahead: false,
            chain_capacity: std::num::NonZeroUsize::MIN,
            conversations: std::num::NonZeroUsize::MIN,
        }
    }

    #[test]
    fn a_score_must_hold_a_token_and_fit_the_context_and_vocabulary() {
        let bounds = Bounds { context: 4, vocabulary: 10 };
        assert!(input(bounds.refusal(&[])).contains("at least one token"));
        let reason = input(bounds.refusal(&tokens([1, 2, 3, 4, 5])));
        assert!(reason.contains("5 tokens") && reason.contains("context is 4"), "{reason}");
        assert!(input(bounds.refusal(&tokens([1, 10]))).contains("token 10"));
        assert!(bounds.refusal(&tokens([0])).is_none());
        assert!(bounds.refusal(&tokens([9, 1, 2, 3])).is_none());
    }

    /// The generation load's limits give the widest flights; a load that
    /// launches fewer rows or requests gives narrower ones; one exporting
    /// no logits scores none.
    #[test]
    fn a_loads_limits_shape_its_scoring_flights() {
        assert_eq!(Shape::of(limits(256, 512, 512)), Some(Shape { sequences: SCORING_SEQUENCES, width: SCORING_WIDTH }));
        assert_eq!(Shape::of(limits(4, 4, 2)), Some(Shape { sequences: 2, width: 2 }));
        assert_eq!(Shape::of(limits(256, 64, 512)), Some(Shape { sequences: 8, width: 8 }));
        assert_eq!(Shape::of(limits(3, 4, 8)), Some(Shape { sequences: 3, width: 1 }));
        assert_eq!(Shape::of(limits(0, 512, 512)), None);
    }

    #[test]
    fn a_walk_forwards_its_next_rows_as_one_decode_exporting_every_row() {
        let width = 4;
        let (mut walk, _) = walk(0..width as u32 + 1, false, true);
        for (next_row, rows) in [(0, 0..width), (width, width..width + 1)] {
            walk.next_row = next_row;
            let forward = walk.forward(width);
            forward.validate().unwrap();
            let Operation::Forward { request, kind, tokens: forced, position, demand, select, committed, .. } = forward else {
                panic!("a walk forwards");
            };
            assert_eq!(request, RequestId(7));
            assert_eq!(kind, WorkKind::Decode);
            assert_eq!(position, rows.start);
            assert_eq!(forced, tokens(rows.start as u32..rows.end as u32));
            assert_eq!(committed, rows.len());
            assert_eq!(demand, Demand::LOGITS);
            assert!(select.is_empty(), "no row selects");
        }
    }

    #[test]
    fn a_walk_scores_each_token_after_the_first_and_takes_the_next_from_the_last_row() {
        let vocabulary = 8;
        let (mut walk, _) = walk([5, 1, 2], true, true);
        assert!((0..3).all(|position| walk.reads(position)), "every row is read where the next token is asked for");
        let (first, second, last) = (logits(vocabulary, 3), logits(vocabulary, 2), logits(vocabulary, 6));
        let rows = vec![Some(first.clone()), Some(second.clone()), Some(last)];
        let Absorbed::Ended(Ok(outcome)) = walk.absorb(0..3, rows).unwrap() else {
            panic!("a walk of one flight ends with its outcome");
        };
        assert_eq!(outcome.positions, vec![reduce_row(&first, TokenId(1), true).unwrap(), reduce_row(&second, TokenId(2), true).unwrap()]);
        assert_eq!(outcome.positions[0].top.unwrap().token, TokenId(3));
        assert_eq!(outcome.positions[1].rank, 1);
        assert_eq!(outcome.next, Some(TokenId(6)));
    }

    /// The next token is the row's most likely, the lowest index winning a
    /// tie, as the reduction's top token is.
    #[test]
    fn the_next_token_ties_to_the_lowest_index() {
        let (mut walk, _) = walk([3], false, true);
        let Absorbed::Ended(Ok(outcome)) = walk.absorb(0..1, vec![Some(vec![1.0, 2.0, 2.0, 0.5])]).unwrap() else {
            panic!("a one-token walk ends in its first flight");
        };
        assert_eq!(outcome.next, Some(TokenId(1)));
    }

    #[test]
    fn a_long_walk_continues_until_every_row_is_scored() {
        let (vocabulary, width) = (4, 4);
        let length = width + 3;
        let (mut walk, _) = walk(std::iter::repeat_n(1, length), false, false);
        assert!(!walk.reads(length - 1), "the last row is not read where the next token is not asked for");
        let rows = |range: Range<usize>| range.map(|position| (position + 1 < length).then(|| logits(vocabulary, 0))).collect::<Vec<_>>();
        assert!(matches!(walk.absorb(0..width, rows(0..width)).unwrap(), Absorbed::Continues));
        let Operation::Forward { position, tokens: forced, .. } = walk.forward(width) else { panic!("a walk forwards") };
        assert_eq!((position, forced.len()), (width, length - width));
        let Absorbed::Ended(Ok(outcome)) = walk.absorb(width..length, rows(width..length)).unwrap() else {
            panic!("the second flight ends the walk");
        };
        assert_eq!(outcome.positions.len(), length - 1);
        assert_eq!(outcome.next, None, "the next token was not asked for");
    }

    #[test]
    fn a_one_token_walk_has_no_positions_and_may_ask_for_the_next() {
        let (mut walk, _) = walk([3], false, true);
        let Absorbed::Ended(Ok(outcome)) = walk.absorb(0..1, vec![Some(logits(4, 2))]).unwrap() else {
            panic!("a one-token walk ends in its first flight");
        };
        assert_eq!(outcome.positions, Vec::new());
        assert_eq!(outcome.next, Some(TokenId(2)));
    }

    #[test]
    fn a_row_the_reduction_refuses_ends_only_its_walk() {
        let (mut scoring, _) = walk([0, 1], false, false);
        let mut row = logits(4, 0);
        row[2] = f32::NAN;
        let verdict = scoring.absorb(0..2, vec![Some(row), None]);
        assert!(matches!(verdict.unwrap(), Absorbed::Ended(Err(RequestError::Invariant(_)))));
        let (mut next, _) = walk([0], false, true);
        let verdict = next.absorb(0..1, vec![Some(vec![f32::NEG_INFINITY; 4])]);
        assert!(matches!(verdict.unwrap(), Absorbed::Ended(Err(RequestError::Invariant(_)))), "a next-token row with no finite logit");
    }

    #[test]
    fn rows_that_break_the_flights_shape_are_an_error() {
        let (mut walk, _) = walk([0, 1, 2], false, true);
        assert!(walk.absorb(0..3, vec![Some(logits(4, 0))]).is_err(), "too few rows");
        assert!(walk.absorb(1..3, vec![Some(logits(4, 0)); 2]).is_err(), "rows other than the walk's next");
        assert!(walk.absorb(0..3, vec![None, Some(logits(4, 0)), Some(logits(4, 0))]).is_err(), "a scoring row without logits");
        assert!(walk.absorb(0..3, vec![Some(logits(4, 0)), Some(logits(4, 0)), None]).is_err(), "a next-token row without logits");
    }

    #[test]
    fn a_deferred_walk_returns_its_request_unanswered_to_the_front() {
        let shape = Shape { sequences: 2, width: 2 };
        let mut walks = Walks::new(shape, Bounds { context: 8, vocabulary: 8 });
        let (first, first_outcomes) = walk([0, 1, 2], false, false);
        let (second, second_outcomes) = request(tokens([3, 4]), false, false);
        let (waiting, _) = request(tokens([5]), false, false);
        walks.active.push(first);
        walks.active.push(Walk::new(second, RequestId(8)));
        walks.waiting.push_back(waiting);
        assert_eq!(walks.forwards().iter().map(Operation::request).collect::<Vec<_>>(), [RequestId(7), RequestId(8)]);
        assert_eq!(walks.defer_newest(), Some(RequestId(8)));
        assert_eq!(walks.waiting.front().map(|request| request.tokens.clone()), Some(tokens([3, 4])));
        assert!(second_outcomes.try_recv().is_err() && first_outcomes.try_recv().is_err(), "a deferral answers nothing");
        assert_eq!(walks.sequences(), BTreeSet::from([RequestId(7)]));
    }

    #[test]
    fn a_wait_ends_once_the_epoch_moves_past_it() {
        let mut walks = Walks::new(Shape { sequences: 2, width: 2 }, Bounds { context: 8, vocabulary: 8 });
        let mut epoch = AvailabilityEpoch::default();
        assert!(walks.waited(epoch), "no wait stands");
        walks.waits = Some(epoch);
        assert!(!walks.waited(epoch) && walks.waits.is_some(), "a wait stands until the epoch moves");
        epoch.advance();
        assert!(walks.waited(epoch) && walks.waits.is_none(), "a change past it ends it");
    }

    #[test]
    fn failing_every_walk_answers_each_once_and_leaves_their_sequences_to_close() {
        let mut walks = Walks::new(Shape { sequences: 2, width: 2 }, Bounds { context: 8, vocabulary: 8 });
        let (open, open_outcomes) = walk([0, 1], false, false);
        let (waiting, waiting_outcomes) = request(tokens([2]), false, false);
        walks.active.push(open);
        walks.waiting.push_back(waiting);
        walks.fail_all(&RequestError::WorkerClosed);
        for outcomes in [open_outcomes, waiting_outcomes] {
            assert_eq!(outcomes.try_recv().unwrap(), Err(RequestError::WorkerClosed));
            assert!(outcomes.try_recv().is_err(), "answered once");
        }
        assert!(walks.active.is_empty() && walks.waiting.is_empty());
        assert_eq!(walks.unclosed, [RequestId(7)]);
    }
}
