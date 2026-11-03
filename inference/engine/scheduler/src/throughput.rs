//! A rolling accepted-output token counter over a fixed ten-second window.

use std::collections::VecDeque;
use std::time::Duration;

/// The fixed observation window in milliseconds.
pub const WINDOW_MS: u64 = 10_000;
const WINDOW: Duration = Duration::from_millis(WINDOW_MS);

/// A live snapshot of the rolling accepted-token counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThroughputSnapshot {
    pub window_ms: u64,
    pub ready: bool,
    pub accepted_tokens: u64,
}

impl ThroughputSnapshot {
    pub fn warming_up() -> Self {
        Self {
            window_ms: WINDOW_MS,
            ready: false,
            accepted_tokens: 0,
        }
    }
}

/// A private rolling accumulator. All methods are pure over elapsed time so
/// tests can drive them with a fake clock.
pub struct ThroughputAccumulator {
    entries: VecDeque<(Duration, u64)>,
}

impl Default for ThroughputAccumulator {
    fn default() -> Self {
        Self {
            entries: VecDeque::new(),
        }
    }
}

impl ThroughputAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `tokens` accepted output at the given elapsed time.
    /// Zero increments are omitted. Identical timestamps are coalesced.
    /// Entries older than the window are pruned.
    pub fn append(&mut self, elapsed: Duration, tokens: u64) {
        if tokens == 0 {
            return;
        }
        self.prune(elapsed);
        if let Some(last) = self.entries.back_mut() {
            if last.0 == elapsed {
                last.1 = last.1.saturating_add(tokens);
                return;
            }
        }
        self.entries.push_back((elapsed, tokens));
    }

    /// A snapshot at the given elapsed time. Ready once the full window
    /// has elapsed. `accepted_tokens` is the sum of entries in
    /// `(elapsed - WINDOW, elapsed]`.
    pub fn snapshot(&self, elapsed: Duration) -> ThroughputSnapshot {
        let ready = elapsed >= WINDOW;
        let horizon = elapsed.saturating_sub(WINDOW);
        let accepted_tokens = self
            .entries
            .iter()
            .filter(|(t, _)| *t > horizon)
            .map(|(_, tokens)| *tokens)
            .fold(0u64, u64::saturating_add);
        ThroughputSnapshot {
            window_ms: WINDOW_MS,
            ready,
            accepted_tokens,
        }
    }

    fn prune(&mut self, elapsed: Duration) {
        let horizon = elapsed.saturating_sub(WINDOW);
        while self.entries.front().is_some_and(|(t, _)| *t <= horizon) {
            self.entries.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn throughput_warming_up_before_ten_seconds() {
        let mut acc = ThroughputAccumulator::new();
        let snap = acc.snapshot(ms(9999));
        assert!(!snap.ready);
        assert_eq!(snap.window_ms, 10_000);
        assert_eq!(snap.accepted_tokens, 0);
    }

    #[test]
    fn throughput_ready_at_exactly_ten_seconds() {
        let mut acc = ThroughputAccumulator::new();
        acc.append(ms(5000), 100);
        let snap = acc.snapshot(ms(10_000));
        assert!(snap.ready);
        assert_eq!(snap.accepted_tokens, 100);
    }

    #[test]
    fn throughput_boundary_exclusion_at_exact_window_edge() {
        let mut acc = ThroughputAccumulator::new();
        acc.append(ms(0), 50);
        let snap = acc.snapshot(ms(10_000));
        // Entry at t=0 is at the boundary: (now - 10s, now] excludes it.
        assert_eq!(snap.accepted_tokens, 0);
    }

    #[test]
    fn throughput_coalesces_identical_timestamps() {
        let mut acc = ThroughputAccumulator::new();
        acc.append(ms(1000), 10);
        acc.append(ms(1000), 20);
        acc.append(ms(1000), 30);
        assert_eq!(acc.entries.len(), 1);
        let snap = acc.snapshot(ms(10_500));
        assert_eq!(snap.accepted_tokens, 60);
    }

    #[test]
    fn throughput_zero_increments_omitted() {
        let mut acc = ThroughputAccumulator::new();
        acc.append(ms(1000), 0);
        assert!(acc.entries.is_empty());
    }

    #[test]
    fn throughput_idle_expiry_prunes_old_entries() {
        let mut acc = ThroughputAccumulator::new();
        acc.append(ms(1000), 200);
        let snap = acc.snapshot(ms(11_500));
        assert!(snap.ready);
        assert_eq!(snap.accepted_tokens, 0);
    }

    #[test]
    fn throughput_two_requests_same_window() {
        let mut acc = ThroughputAccumulator::new();
        acc.append(ms(2000), 200);
        acc.append(ms(5000), 200);
        let snap = acc.snapshot(ms(10_000));
        assert!(snap.ready);
        assert_eq!(snap.accepted_tokens, 400);
    }

    #[test]
    fn throughput_retained_window() {
        let mut acc = ThroughputAccumulator::new();
        acc.append(ms(9_500), 100);
        acc.append(ms(12_000), 50);
        // At 19499ms: horizon = 9499ms. Both entries are > 9499ms.
        let snap = acc.snapshot(ms(19_499));
        assert_eq!(snap.accepted_tokens, 150);
        // At 19500ms: horizon = 9500ms. The 9500ms entry is at the boundary, excluded.
        let snap = acc.snapshot(ms(19_500));
        assert_eq!(snap.accepted_tokens, 50);
    }
}
