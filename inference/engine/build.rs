//! The engine build identity (`worker::protocol::EngineBuild`): the sources and
//! locked versions of the engine's dependency closure, which holds both ends of
//! the worker protocol and everything a worker computes, and the compiler.

fn main() {
    // This value is an input to the cold-tier cache identity (`options.rs`
    // `cache_identity`): any source change starts a new, empty
    // `cache/state/IDENTITY_HEX` dir, and old dirs are reclaimed by
    // `collect_identity_dirs`.
    println!(
        "cargo:rustc-env=MAGNITUDE_ENGINE_BUILD={}",
        magnitude_source_identity::of_closure(env!("CARGO_PKG_NAME"))
    );
}
