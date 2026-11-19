//! How a loaded model serves at one observation: the places its prompt
//! groups hold, and where its routed experts are, each holding with its
//! last resize and why the owner made it (`ExpertResidency`).

use magnitude_executor::{ExecutorDomain, ExpertResidency, ProgramFamily};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServingCensus {
    /// The places of the prompt-group class held now: one, or an open
    /// group's (`ExecutorDomain::open_group`).
    pub group_places: std::num::NonZeroUsize,
    /// The most places a prompt group may take: the load's chain capacity.
    pub group_capacity: std::num::NonZeroUsize,
    /// None for a load without an elastic expert bank.
    pub experts: Option<ExpertResidency>,
}

impl ServingCensus {
    /// How `domain`'s load serves now, whatever owner drives it.
    pub fn of<F: ProgramFamily>(domain: &ExecutorDomain<F>) -> Result<Self, String> {
        Ok(Self {
            group_places: domain.group_places(),
            group_capacity: domain.chain_capacity(),
            experts: domain.expert_residency().map_err(|error| error.to_string())?,
        })
    }
}
