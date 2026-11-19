//! Family-neutral request scheduling and execution ownership.
//!
//! The crate observes logical generation work, byte-denominated capacity, and
//! completion events. Model-family execution and protocol publication remain
//! in composition crates above this boundary.

pub mod cold_tier;
pub mod domain;
pub mod memory_condition;
pub mod owner;
pub mod policy;
pub mod prefix_cache;
pub mod protocol;
pub mod publication;
pub mod round_driver;
pub mod scoring;
pub mod throughput;
pub mod worker;

pub use policy::ServiceLimits;
pub use throughput::ThroughputSnapshot;
