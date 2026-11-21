//! Teacher-forced scoring's values: the reduction of one logits row to the real token's log-probability, its
//! rank and the top token, and the score of a whole prompt. Device-free, so they cross the worker protocol.
//! The owner's walks that score are in [`walk`].

pub(crate) mod walk;

use magnitude_executor::TokenId;
use serde::{Deserialize, Serialize};

/// The most walks one scoring flight steps.
pub const SCORING_SEQUENCES: usize = 8;
/// The most rows of one walk one scoring flight forwards.
pub const SCORING_WIDTH: usize = 32;
/// The most logits rows one scoring flight exports: what a load that scores at its widest exports a launch.
pub const SCORING_ROWS: usize = SCORING_SEQUENCES * SCORING_WIDTH;

/// The highest-probability token of a row and its log-probability.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopToken {
    pub token: TokenId,
    pub logprob: f64,
}

/// One scored prompt position: the real token's log-probability and 1-based rank, and the top token when it
/// was asked for and differs from the real one.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoredPosition {
    pub logprob: f64,
    pub rank: u32,
    pub top: Option<TopToken>,
}

/// The score of a whole prompt: one position per scored token, and the sampled next token when requested.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreOutcome {
    pub positions: Vec<ScoredPosition>,
    pub next: Option<TokenId>,
}

/// Reduce one logits row to the log-softmax of `real`, its rank (1 + the count of strictly larger logits) and,
/// when `top` is asked for and the argmax differs from `real`, the argmax token. The lowest index wins an argmax tie.
pub fn reduce_row(row: &[f32], real: TokenId, top: bool) -> Result<ScoredPosition, String> {
    let index = usize::try_from(real.0).map_err(|_| "token id exceeds the host domain".to_owned())?;
    let real_logit = f64::from(*row.get(index).ok_or("token outside the logits row")?);
    let (likeliest, max) = argmax(row)?;
    if real_logit == f64::NEG_INFINITY {
        return Err(format!("token {index} has a -inf logit, so its probability is zero"));
    }
    let sum: f64 = row.iter().map(|&x| (f64::from(x) - max).exp()).sum();
    let lse = max + sum.ln();
    let larger = row.iter().filter(|&&x| f64::from(x) > real_logit).count();
    let rank = u32::try_from(larger + 1).map_err(|_| "row too large to rank".to_owned())?;
    let top = if top && likeliest != index {
        let id = u32::try_from(likeliest).map_err(|_| "argmax exceeds the token id domain".to_owned())?;
        Some(TopToken { token: TokenId(id), logprob: max - lse })
    } else {
        None
    };
    Ok(ScoredPosition { logprob: real_logit - lse, rank, top })
}

/// The most likely entry of a logits row and its logit: the lowest index wins a tie. A row with a NaN, or whose
/// maximum is not finite, has none.
pub(crate) fn argmax(row: &[f32]) -> Result<(usize, f64), String> {
    let (mut max, mut argmax) = (f64::NEG_INFINITY, 0usize);
    for (i, &x) in row.iter().enumerate() {
        if x.is_nan() {
            return Err(format!("logits row has a NaN at index {i}"));
        }
        if f64::from(x) > max {
            max = f64::from(x);
            argmax = i;
        }
    }
    if !max.is_finite() {
        return Err("logits row maximum is not finite (all -inf or a +inf entry)".to_owned());
    }
    Ok((argmax, max))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(id: u32) -> TokenId {
        TokenId(id)
    }

    #[test]
    fn the_real_tokens_logprob_is_its_log_softmax() {
        let row = [1.0f32, 2.0, 3.0];
        let p = reduce_row(&row, token(1), false).unwrap();
        let lse = (1f64.exp() + 2f64.exp() + 3f64.exp()).ln();
        assert!((p.logprob - (2.0 - lse)).abs() < 1e-12);
        assert_eq!(p.rank, 2);
        assert_eq!(p.top, None);
    }

    #[test]
    fn the_top_token_is_reported_only_when_it_differs() {
        let row = [0.5f32, 4.0, -1.0];
        let other = reduce_row(&row, token(0), true).unwrap();
        assert_eq!(other.top.map(|t| t.token), Some(token(1)));
        let same = reduce_row(&row, token(1), true).unwrap();
        assert_eq!(same.rank, 1);
        assert_eq!(same.top, None);
    }

    #[test]
    fn ties_rank_by_strictly_larger_logits() {
        let row = [2.0f32, 2.0, 1.0];
        assert_eq!(reduce_row(&row, token(1), false).unwrap().rank, 1);
        assert_eq!(reduce_row(&row, token(2), false).unwrap().rank, 3);
    }

    #[test]
    fn large_logits_do_not_overflow() {
        let row = [1000.0f32, 999.0];
        let p = reduce_row(&row, token(0), false).unwrap();
        assert!(p.logprob.is_finite() && p.logprob < 0.0 && p.logprob > -1.0);
    }

    #[test]
    fn a_token_outside_the_row_is_an_error() {
        assert!(reduce_row(&[0.0f32; 4], token(4), false).is_err());
    }

    #[test]
    fn argmax_ties_go_to_the_lowest_index() {
        let row = [2.0f32, 2.0, 1.0];
        let p = reduce_row(&row, token(2), true).unwrap();
        assert_eq!(p.top.map(|t| t.token), Some(token(0)));
    }

    #[test]
    fn the_top_logprob_is_the_argmax_log_softmax() {
        let row = [0.5f32, 4.0, -1.0];
        let p = reduce_row(&row, token(0), true).unwrap();
        let lse = (0.5f64.exp() + 4f64.exp() + (-1f64).exp()).ln();
        assert!((p.top.unwrap().logprob - (4.0 - lse)).abs() < 1e-12);
    }

    #[test]
    fn an_empty_row_is_an_error() {
        assert!(reduce_row(&[], token(0), false).is_err());
    }

    #[test]
    fn a_nan_logit_is_an_error() {
        assert!(reduce_row(&[1.0f32, f32::NAN, 0.0], token(0), false).is_err());
    }

    #[test]
    fn a_non_finite_maximum_is_an_error() {
        assert!(reduce_row(&[f32::NEG_INFINITY; 3], token(0), false).is_err());
        assert!(reduce_row(&[0.0f32, f32::INFINITY], token(0), false).is_err());
    }

    #[test]
    fn a_real_token_with_zero_probability_is_an_error() {
        assert!(reduce_row(&[1.0f32, f32::NEG_INFINITY], token(1), false).is_err());
    }

    #[test]
    fn masked_entries_with_a_finite_maximum_score_normally() {
        let row = [f32::NEG_INFINITY, 1.0, 2.0];
        let p = reduce_row(&row, token(1), false).unwrap();
        let lse = (1f64.exp() + 2f64.exp()).ln();
        assert!((p.logprob - (1.0 - lse)).abs() < 1e-12);
        assert_eq!(p.rank, 2);
    }
}
