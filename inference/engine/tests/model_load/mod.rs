//! How the model-backed tests load a model: as a load in its own process
//! does, whatever the model, and one at a time.
//!
//! Every load plans on the machine's figures (prices), which its kernel
//! cache holds; a load without one plans only a model held whole, and a load
//! whose figures are not measured yet waits for them. So the tests keep a
//! kernel cache and settle the prices before they open anything, through the
//! engine's own step for it ([`settle_prices`]).
//!
//! A process holds one load: its memory ledger opens one state pool per
//! store, as a worker process serves one load. Cargo runs tests on threads
//! of one process, so a test's first load takes the process's one load
//! ([`resolve`]) and holds it until the test's thread ends, after everything
//! the test loaded is dropped.

use magnitude_engine::composition::{settle_prices, EngineConfiguration, ResolvedEngineConfiguration};
use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, PoisonError};

/// The process's one load.
static LOAD: Mutex<()> = Mutex::new(());

thread_local! {
    /// The process's load, held by the test running on this thread.
    static HELD: RefCell<Option<MutexGuard<'static, ()>>> = const { RefCell::new(None) };
}

/// `MAGNITUDE_TEST_KERNEL_CACHE`, else the tests' own kernel cache in Cargo's
/// directory for integration tests' data: it outlives a run, so the
/// machine's figures are measured once.
pub fn kernel_cache() -> PathBuf {
    std::env::var_os("MAGNITUDE_TEST_KERNEL_CACHE")
        .map_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("kernels"), PathBuf::from)
}

/// Resolve `configuration` and settle the prices its load plans on, holding
/// the process's one load for the rest of the calling test (another test's
/// failure poisons nothing it holds). The first time on a machine the
/// prices' measurement jobs run here, which no other load may stand beside.
pub fn resolve(configuration: EngineConfiguration) -> ResolvedEngineConfiguration {
    HELD.with(|held| {
        held.borrow_mut()
            .get_or_insert_with(|| LOAD.lock().unwrap_or_else(PoisonError::into_inner));
    });
    let resolved = configuration.resolve().unwrap();
    let measured = settle_prices(&resolved.manifest).unwrap();
    if measured > 0 {
        eprintln!("settled the load's prices: {measured} measurement runs");
    }
    resolved
}
