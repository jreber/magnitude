//! The memory condition every execution owner drives: its observation
//! cadence, the condition one probe of the loaded model's memory domains
//! moves it to, and when a persisting condition unloads the model.
//!
//! An owner observes memory every [`OBSERVATION_INTERVAL_NS`], and when a
//! growth it attempts is refused for memory. In Reclaim, admission and growth
//! pause while the owner releases what it can; the model unloads once Reclaim
//! persists for [`MEMORY_ESCALATION_NS`] after releases are exhausted. Failed
//! observations are tolerated for one escalation interval, then count as
//! Reclaim whose unload clock has already expired.

use magnitude_executor::DomainError;

/// How long Reclaim may persist after releases are exhausted, and how long
/// observations may fail continuously, before the model unloads.
pub const MEMORY_ESCALATION_NS: u64 = 1_000_000_000;

/// How often the owner observes memory without another reason to run.
pub const OBSERVATION_INTERVAL_NS: u64 = 100_000_000;

/// One probe of every domain the loaded model uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryObservation {
    /// Every domain's headroom is above its planning reserve.
    Normal,
    /// Some domain's headroom is at or below its planning reserve.
    Reclaim,
    /// A required observation failed.
    Blind,
}

impl MemoryObservation {
    /// The observation a probe of the domain's memory made; a probe that
    /// failed otherwise is no observation.
    pub fn of(probe: Result<(), DomainError>) -> Result<Self, String> {
        match probe {
            Ok(()) => Ok(Self::Normal),
            Err(DomainError::Reclaim) => Ok(Self::Reclaim),
            Err(DomainError::Blind(_)) => Ok(Self::Blind),
            Err(error) => Err(error.to_string()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryCondition {
    Normal,
    /// Admission and growth pause while releases run. `since` starts the
    /// unload clock: when the band was entered, restarted whenever a release
    /// frees memory, so the model unloads once Reclaim persists for
    /// [`MEMORY_ESCALATION_NS`] after releases are exhausted.
    Reclaim {
        since: u64,
    },
    /// Observations have failed continuously since `since`.
    Blind {
        since: u64,
    },
}

impl MemoryCondition {
    pub fn observe(self, observation: MemoryObservation, now: u64) -> Self {
        match observation {
            MemoryObservation::Normal => Self::Normal,
            MemoryObservation::Reclaim => match self {
                Self::Reclaim { since } => Self::Reclaim { since },
                _ => Self::Reclaim { since: now },
            },
            MemoryObservation::Blind => match self {
                // One continuous escalation interval of Blind is Reclaim
                // whose unload clock has already expired.
                Self::Blind { since } if now.saturating_sub(since) >= MEMORY_ESCALATION_NS => {
                    Self::Reclaim { since }
                }
                Self::Blind { since } => Self::Blind { since },
                Self::Reclaim { since } => Self::Reclaim { since },
                Self::Normal => Self::Blind { since: now },
            },
        }
    }

    /// A release freed memory at `now`: releases are not exhausted yet.
    pub fn released(self, now: u64) -> Self {
        match self {
            Self::Reclaim { .. } => Self::Reclaim { since: now },
            other => other,
        }
    }

    pub fn should_unload(self, now: u64) -> bool {
        matches!(self, Self::Reclaim { since } if now.saturating_sub(since) >= MEMORY_ESCALATION_NS)
    }
}

/// When the owner next observes memory without another reason to run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObservationCadence {
    next: u64,
}

impl ObservationCadence {
    /// Whether an observation is due at `now`; a due one is taken, and the
    /// next falls one interval later.
    pub fn take(&mut self, now: u64) -> bool {
        if now < self.next {
            return false;
        }
        self.next = now.saturating_add(OBSERVATION_INTERVAL_NS);
        true
    }

    /// The time of the next observation.
    pub fn next(self) -> u64 {
        self.next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_observation_is_due_at_once_then_one_interval_later() {
        let mut cadence = ObservationCadence::default();
        assert!(cadence.take(5));
        assert_eq!(cadence.next(), 5 + OBSERVATION_INTERVAL_NS);
        assert!(!cadence.take(5 + OBSERVATION_INTERVAL_NS - 1));
        assert!(cadence.take(5 + OBSERVATION_INTERVAL_NS));
    }

    #[test]
    fn a_probe_that_fails_otherwise_is_no_observation() {
        assert_eq!(MemoryObservation::of(Ok(())), Ok(MemoryObservation::Normal));
        assert_eq!(
            MemoryObservation::of(Err(DomainError::Reclaim)),
            Ok(MemoryObservation::Reclaim)
        );
        assert_eq!(
            MemoryObservation::of(Err(DomainError::Blind("no reading".into()))),
            Ok(MemoryObservation::Blind)
        );
        assert!(MemoryObservation::of(Err(DomainError::Input("other".into()))).is_err());
    }

    #[test]
    fn continuous_blind_escalates_to_immediately_eligible_reclaim() {
        let blind = MemoryCondition::Normal.observe(MemoryObservation::Blind, 7);
        assert_eq!(blind, MemoryCondition::Blind { since: 7 });
        let before = blind.observe(MemoryObservation::Blind, 7 + MEMORY_ESCALATION_NS - 1);
        assert_eq!(before, blind);
        assert!(!before.should_unload(7 + MEMORY_ESCALATION_NS - 1));
        let escalated = before.observe(MemoryObservation::Blind, 7 + MEMORY_ESCALATION_NS);
        assert_eq!(escalated, MemoryCondition::Reclaim { since: 7 });
        assert!(escalated.should_unload(7 + MEMORY_ESCALATION_NS));
    }

    #[test]
    fn interrupted_blind_restarts_its_interval() {
        let blind = MemoryCondition::Normal.observe(MemoryObservation::Blind, 7);
        let normal = blind.observe(MemoryObservation::Normal, 8);
        assert_eq!(normal, MemoryCondition::Normal);
        let renewed = normal.observe(MemoryObservation::Blind, 7 + MEMORY_ESCALATION_NS);
        assert_eq!(
            renewed,
            MemoryCondition::Blind {
                since: 7 + MEMORY_ESCALATION_NS
            }
        );
        assert!(!renewed.should_unload(7 + MEMORY_ESCALATION_NS));
    }

    #[test]
    fn reclaim_unloads_one_interval_after_releases_are_exhausted() {
        let reclaim = MemoryCondition::Normal.observe(MemoryObservation::Reclaim, 5);
        assert_eq!(reclaim, MemoryCondition::Reclaim { since: 5 });
        // Continued Reclaim keeps the clock; a release that frees memory
        // restarts it.
        assert_eq!(reclaim.observe(MemoryObservation::Reclaim, 9), reclaim);
        let released = reclaim.released(100);
        assert_eq!(released, MemoryCondition::Reclaim { since: 100 });
        assert!(!released.should_unload(100 + MEMORY_ESCALATION_NS - 1));
        assert!(released.should_unload(100 + MEMORY_ESCALATION_NS));
        // A failed reading during Reclaim neither resets nor extends it.
        assert_eq!(released.observe(MemoryObservation::Blind, 200), released);
    }

    #[test]
    fn recovery_clears_reclaim_and_a_new_episode_starts_fresh() {
        let reclaim = MemoryCondition::Normal.observe(MemoryObservation::Reclaim, 5);
        let normal = reclaim.observe(MemoryObservation::Normal, 10);
        assert_eq!(normal, MemoryCondition::Normal);
        assert!(!normal.should_unload(10 + MEMORY_ESCALATION_NS));
        assert_eq!(normal.released(11), MemoryCondition::Normal);
        let renewed = normal.observe(MemoryObservation::Reclaim, 11);
        assert_eq!(renewed, MemoryCondition::Reclaim { since: 11 });
    }
}
