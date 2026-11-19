//! One score's walk through its tokens, device-free: the rows each step
//! forwards, and what the walk takes from a step's rows.

use crate::options::SCORING_WIDTH;
use magnitude_executor::{
    Demand, InvariantError, Operation, RequestId, Sampling, SelectSpec, Shaping, TokenId, WorkKind,
};
use magnitude_family_contracts::ModelDefinition;
use magnitude_scheduler::{
    protocol::ScoreRequest,
    publication::RequestError,
    scoring::{reduce_row, ScoreOutcome, ScoredPosition},
};
use std::ops::Range;

/// What a score's tokens must keep within: the load's context, and the
/// model's vocabulary.
#[derive(Clone, Copy, Debug)]
pub(super) struct Bounds {
    pub(super) context: usize,
    pub(super) vocabulary: usize,
}

impl Bounds {
    /// The bounds of a load of `definition`.
    pub(super) fn of(definition: &ModelDefinition) -> Result<Self, String> {
        let host = |value: u64, what: &str| {
            usize::try_from(value)
                .map_err(|_| format!("the model's {what} exceeds the host domain"))
        };
        Ok(Self {
            context: host(definition.decoder.context_limit, "context")?,
            vocabulary: host(definition.decoder.vocabulary, "vocabulary")?,
        })
    }

    /// Why a score of `tokens` cannot be walked, if it cannot.
    pub(super) fn refusal(self, tokens: &[TokenId]) -> Option<RequestError> {
        if tokens.is_empty() {
            return Some(RequestError::Input(
                "a score needs at least one token".into(),
            ));
        }
        if tokens.len() > self.context {
            return Some(RequestError::Input(format!(
                "the score has {} tokens but the scoring load's context is {}",
                tokens.len(),
                self.context
            )));
        }
        let outside =
            |token: &&TokenId| usize::try_from(token.0).map_or(true, |id| id >= self.vocabulary);
        tokens.iter().find(outside).map(|token| {
            RequestError::Input(format!(
                "token {} is outside the model's vocabulary of {}",
                token.0, self.vocabulary
            ))
        })
    }
}

/// One admitted score's walk through its tokens.
pub(super) struct Walk {
    request: ScoreRequest,
    sequence: RequestId,
    /// The first row not yet forwarded; every row before it is scored.
    next_row: usize,
    positions: Vec<ScoredPosition>,
    /// The greedy token after the last one, once its row is forwarded.
    next: Option<TokenId>,
}

/// One forwarded row on the host: its logits when it predicts a token of
/// the walk, and its greedy selection.
pub(super) struct Row {
    pub(super) logits: Option<Vec<f32>>,
    pub(super) selected: Option<TokenId>,
}

/// What one step left of a walk.
pub(super) enum Absorbed {
    Continues,
    Ended(Result<ScoreOutcome, RequestError>),
}

impl Walk {
    /// A fresh walk of `request` on `sequence`: nothing forwarded yet.
    pub(super) fn new(request: ScoreRequest, sequence: RequestId) -> Self {
        let positions = Vec::with_capacity(request.tokens.len().saturating_sub(1));
        Self {
            request,
            sequence,
            next_row: 0,
            positions,
            next: None,
        }
    }

    pub(super) fn sequence(&self) -> RequestId {
        self.sequence
    }

    /// The request, its progress dropped: a deferred walk starts again.
    pub(super) fn into_request(self) -> ScoreRequest {
        self.request
    }

    /// Answer the request: the walk's one call of its callback.
    pub(super) fn answer(self, result: Result<ScoreOutcome, RequestError>) {
        (self.request.done)(result)
    }

    /// The rows the walk's next step forwards.
    pub(super) fn rows(&self) -> Range<usize> {
        self.next_row..(self.next_row + SCORING_WIDTH).min(self.request.tokens.len())
    }

    /// Whether the row at `position` predicts one of the walk's tokens; the
    /// last row predicts the token after them.
    pub(super) fn predicts(&self, position: usize) -> bool {
        position + 1 < self.request.tokens.len()
    }

    /// The walk's next step: its next rows teacher-forced, every row's logits
    /// exported and a token selected greedily at each. One row is a decode;
    /// more are one verification forward (the drafting verify shapes), as
    /// `magnitude-harness logits` forwards them.
    pub(super) fn forward(&self) -> Operation {
        let rows = self.rows();
        let tokens = self.request.tokens[rows.clone()].to_vec();
        Operation::Forward {
            request: self.sequence,
            kind: if tokens.len() == 1 {
                WorkKind::Decode
            } else {
                WorkKind::Verify
            },
            position: rows.start,
            conditioning: None,
            demand: Demand::LOGITS | Demand::SELECT,
            select: rows.map(greedy).collect(),
            committed: tokens.len(),
            tokens,
            prime: None,
        }
    }

    /// Take the step's `rows`: each row that predicts a token scores it, and
    /// the last row's selection is the next token when it is asked for. A
    /// row the reduction refuses ends this walk alone; rows that break the
    /// step's shape are an error.
    pub(super) fn absorb(&mut self, rows: Vec<Row>) -> Result<Absorbed, String> {
        let forwarded = self.rows();
        if rows.len() != forwarded.len() {
            return Err(format!(
                "a step returned {} rows of a walk it forwarded {} rows of",
                rows.len(),
                forwarded.len()
            ));
        }
        let end = forwarded.end;
        for (position, row) in forwarded.zip(rows) {
            if let Some(&real) = self.request.tokens.get(position + 1) {
                let logits = row.logits.ok_or("a scoring row returned no logits")?;
                match reduce_row(&logits, real, self.request.top) {
                    Ok(scored) => self.positions.push(scored),
                    Err(detail) => {
                        return Ok(Absorbed::Ended(Err(RequestError::Invariant(
                            InvariantError {
                                context: "scoring a logits row",
                                detail,
                            },
                        ))))
                    }
                }
            } else if self.request.next {
                self.next = Some(row.selected.ok_or("the next-token row selected no token")?);
            }
        }
        self.next_row = end;
        Ok(if end == self.request.tokens.len() {
            Absorbed::Ended(Ok(ScoreOutcome {
                positions: std::mem::take(&mut self.positions),
                next: self.next,
            }))
        } else {
            Absorbed::Continues
        })
    }
}

/// The greedy selection at `position`, unshaped: a load that exports logits
/// selects from them as they are.
fn greedy(position: usize) -> SelectSpec {
    SelectSpec {
        sampling: Sampling::Greedy,
        seed: 0,
        position,
        domain: 0,
        mask: None,
        shaping: Shaping::default(),
        history: None,
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use std::sync::mpsc;

    pub(in crate::worker::scoring) type Outcomes =
        mpsc::Receiver<Result<ScoreOutcome, RequestError>>;

    pub(in crate::worker::scoring) fn tokens(ids: impl IntoIterator<Item = u32>) -> Vec<TokenId> {
        ids.into_iter().map(TokenId).collect()
    }

    pub(in crate::worker::scoring) fn request(
        tokens: Vec<TokenId>,
        top: bool,
        next: bool,
    ) -> (ScoreRequest, Outcomes) {
        let (sender, receiver) = mpsc::channel();
        let request = ScoreRequest {
            tokens,
            top,
            next,
            done: Box::new(move |outcome| sender.send(outcome).unwrap()),
        };
        (request, receiver)
    }

    fn walk(ids: impl IntoIterator<Item = u32>, top: bool, next: bool) -> (Walk, Outcomes) {
        let (request, outcomes) = request(tokens(ids), top, next);
        (Walk::new(request, RequestId(7)), outcomes)
    }

    /// A logits row of `vocabulary` entries peaking at `peak`.
    fn logits(vocabulary: usize, peak: usize) -> Vec<f32> {
        (0..vocabulary)
            .map(|index| {
                if index == peak {
                    4.0
                } else {
                    index as f32 * 0.25
                }
            })
            .collect()
    }

    fn scoring(logits: Vec<f32>) -> Row {
        Row {
            logits: Some(logits),
            selected: None,
        }
    }

    fn next_row(token: u32) -> Row {
        Row {
            logits: None,
            selected: Some(TokenId(token)),
        }
    }

    fn input(error: Option<RequestError>) -> String {
        match error {
            Some(RequestError::Input(reason)) => reason,
            other => panic!("expected an input refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_score_must_hold_a_token_and_fit_the_context_and_vocabulary() {
        let bounds = Bounds {
            context: 4,
            vocabulary: 10,
        };
        assert!(input(bounds.refusal(&[])).contains("at least one token"));
        let reason = input(bounds.refusal(&tokens([1, 2, 3, 4, 5])));
        assert!(
            reason.contains("5 tokens") && reason.contains("context is 4"),
            "{reason}"
        );
        assert!(input(bounds.refusal(&tokens([1, 10]))).contains("token 10"));
        assert!(bounds.refusal(&tokens([0])).is_none());
        assert!(bounds.refusal(&tokens([9, 1, 2, 3])).is_none());
    }

    /// Refusals are request-class: the host reports them as invalid requests.
    #[test]
    fn a_refused_score_reaches_the_host_as_an_invalid_request() {
        let bounds = Bounds {
            context: 1,
            vocabulary: 10,
        };
        let refusal = bounds.refusal(&tokens([1, 2])).unwrap();
        assert!(matches!(
            crate::error::RequestError::from(refusal),
            crate::error::RequestError::InvalidRequest { .. }
        ));
    }

    #[test]
    fn a_walk_forwards_w_rows_as_one_verification_and_a_lone_row_as_a_decode() {
        let length = SCORING_WIDTH as u32 + 1;
        let (mut walk, _) = walk(0..length, false, true);
        let first = walk.forward();
        first.validate().unwrap();
        let Operation::Forward {
            request,
            kind,
            tokens: forced,
            position,
            demand,
            select,
            committed,
            ..
        } = first
        else {
            panic!("a walk forwards");
        };
        assert_eq!(request, RequestId(7));
        assert_eq!(kind, WorkKind::Verify);
        assert_eq!(position, 0);
        assert_eq!(forced, tokens(0..SCORING_WIDTH as u32));
        assert_eq!(committed, SCORING_WIDTH);
        assert!(demand.contains(Demand::LOGITS) && demand.contains(Demand::SELECT));
        assert_eq!(
            select.iter().map(|spec| spec.position).collect::<Vec<_>>(),
            (0..SCORING_WIDTH).collect::<Vec<_>>()
        );
        assert!(select
            .iter()
            .all(|spec| matches!(spec.sampling, Sampling::Greedy) && spec.mask.is_none()));
        walk.next_row = SCORING_WIDTH;
        let last = walk.forward();
        last.validate().unwrap();
        let Operation::Forward {
            kind,
            tokens: forced,
            position,
            select,
            committed,
            ..
        } = last
        else {
            panic!("a walk forwards");
        };
        assert_eq!(kind, WorkKind::Decode);
        assert_eq!((position, committed), (SCORING_WIDTH, 1));
        assert_eq!(forced, tokens([SCORING_WIDTH as u32]));
        assert_eq!(select.len(), 1);
        assert_eq!(select[0].position, SCORING_WIDTH);
    }

    #[test]
    fn a_walk_scores_each_token_after_the_first_and_takes_the_next_from_the_last_row() {
        let vocabulary = 8;
        let (mut walk, _) = walk([5, 1, 2], true, true);
        let (first, second) = (logits(vocabulary, 3), logits(vocabulary, 2));
        let rows = vec![scoring(first.clone()), scoring(second.clone()), next_row(6)];
        let Absorbed::Ended(Ok(outcome)) = walk.absorb(rows).unwrap() else {
            panic!("a walk of one step ends with its outcome");
        };
        assert_eq!(
            outcome.positions,
            vec![
                reduce_row(&first, TokenId(1), true).unwrap(),
                reduce_row(&second, TokenId(2), true).unwrap()
            ]
        );
        assert_eq!(outcome.positions[0].top.unwrap().token, TokenId(3));
        assert_eq!(outcome.positions[1].rank, 1);
        assert_eq!(outcome.next, Some(TokenId(6)));
    }

    #[test]
    fn a_long_walk_continues_until_every_row_is_scored() {
        let vocabulary = 4;
        let length = SCORING_WIDTH + 8;
        let (mut walk, _) = walk(std::iter::repeat_n(1, length), false, false);
        let rows = |count: usize, last: bool| {
            (0..count)
                .map(|index| {
                    if last && index + 1 == count {
                        next_row(0)
                    } else {
                        scoring(logits(vocabulary, 0))
                    }
                })
                .collect::<Vec<_>>()
        };
        assert!(matches!(
            walk.absorb(rows(SCORING_WIDTH, false)).unwrap(),
            Absorbed::Continues
        ));
        assert_eq!(walk.rows(), SCORING_WIDTH..length);
        let Absorbed::Ended(Ok(outcome)) = walk.absorb(rows(8, true)).unwrap() else {
            panic!("the second step ends the walk");
        };
        assert_eq!(outcome.positions.len(), length - 1);
        assert_eq!(outcome.next, None, "the next token was not asked for");
    }

    #[test]
    fn a_one_token_walk_has_no_positions_and_may_ask_for_the_next() {
        let (mut walk, _) = walk([3], false, true);
        assert_eq!(walk.rows(), 0..1);
        let Absorbed::Ended(Ok(outcome)) = walk.absorb(vec![next_row(2)]).unwrap() else {
            panic!("a one-token walk ends in its first step");
        };
        assert_eq!(outcome.positions, Vec::new());
        assert_eq!(outcome.next, Some(TokenId(2)));
    }

    #[test]
    fn a_row_the_reduction_refuses_ends_only_its_walk() {
        let (mut walk, _) = walk([0, 1], false, false);
        let mut row = logits(4, 0);
        row[2] = f32::NAN;
        let verdict = walk.absorb(vec![scoring(row), scoring(logits(4, 0))]);
        assert!(matches!(
            verdict.unwrap(),
            Absorbed::Ended(Err(RequestError::Invariant(_)))
        ));
    }

    #[test]
    fn rows_that_break_the_steps_shape_are_an_error() {
        let (mut walk, _) = walk([0, 1, 2], false, true);
        assert!(
            walk.absorb(vec![scoring(logits(4, 0))]).is_err(),
            "too few rows"
        );
        let missing = vec![next_row(0), scoring(logits(4, 0)), next_row(0)];
        assert!(
            walk.absorb(missing).is_err(),
            "a scoring row without logits"
        );
    }

    #[test]
    fn a_deferred_walk_returns_its_request_unanswered() {
        let (walk, outcomes) = walk([0, 1, 2], false, false);
        let request = walk.into_request();
        assert_eq!(request.tokens, tokens([0, 1, 2]));
        assert!(outcomes.try_recv().is_err(), "a deferral answers nothing");
    }
}
