//! What one allocation is charged: the bytes a backend's allocator takes for
//! a buffer of `bytes` (`DeviceService::allocate`), its rounding included,
//! which its entry is charged while the buffer lives and freed of as it
//! drops. The one rule the backends allocate by and a load's plan prices its
//! allocations by ([`allocation_charge`]), so what is planned, what is
//! charged and what a release frees are one figure. A granule pool's blocks
//! are sized by their own rule (`seismic_memory::pool::block_bytes`).

use seismic_lang::registry::BackendName;
use seismic_topology::Contract;

/// The largest CPU buffer carved from the block its entry is filling; a
/// larger one takes whole host pages of its own (`seismic_cpu`'s carver).
/// Faulting fresh pages costs far more than zeroing held ones, so every
/// buffer a heap serves from memory it keeps is carved, and the heap keeps
/// what is under its mmap threshold ceiling (32 MiB); this is 1 MiB because
/// the block an entry holds idle is as large as the largest buffer carved,
/// and a buffer of more (weights, state) lives long enough for its faults
/// to amortize.
pub const CPU_CARVED_LIMIT: u64 = 1 << 20;

/// The host page, which a page-backed allocation is a whole number of.
///
/// # Panics
/// Where the host reports no page: every allocation of the process stands on it.
pub fn host_page() -> u64 {
    seismic_memory::host_page().unwrap_or_else(|error| panic!("invariant violated: the host reports no page: {error}"))
}

/// `bytes` (at least one) in whole host pages: memory of its own, mapped by
/// the page (a CUDA pinned host allocation, the CPU's and Metal's larger
/// buffers).
///
/// # Panics
/// Where they overflow u64: no allocation within a device's limits does.
pub fn host_pages(bytes: u64) -> u64 {
    bytes
        .max(1)
        .checked_next_multiple_of(host_page())
        .unwrap_or_else(|| panic!("invariant violated: an allocation of {bytes} bytes in whole host pages overflows u64"))
}

/// What Metal charges a shared buffer of `bytes` (at least one): its own
/// count of it (`allocatedSize`), the bytes themselves up to a host page,
/// which Metal places within pages it shares, and whole pages beyond.
pub fn metal_buffer(bytes: u64) -> u64 {
    match bytes.max(1) {
        bytes if bytes <= host_page() => bytes,
        bytes => host_pages(bytes),
    }
}

/// What the CPU charges a buffer of `bytes`: up to [`CPU_CARVED_LIMIT`] its
/// bytes of the block its entry holds whole, beyond it whole host pages of
/// its own.
pub fn cpu_buffer(bytes: u64) -> u64 {
    match bytes {
        bytes if bytes <= CPU_CARVED_LIMIT => bytes,
        bytes => host_pages(bytes),
    }
}

/// What `backend` charges one allocation of `bytes` of `contract`
/// (`DeviceService::allocate` in the first kind the contract places in):
/// Metal's and the CPU's one kind as [`metal_buffer`] and [`cpu_buffer`]
/// say; a CUDA device range or a Vulkan range its bytes (at least one),
/// carved from a block; CUDA's pinned host memory, which every contract but
/// `DeviceOnly` places in, whole host pages.
pub fn allocation_charge(backend: BackendName, contract: Contract, bytes: u64) -> u64 {
    match (backend, contract) {
        (BackendName::Metal, _) => metal_buffer(bytes),
        (BackendName::Cpu, _) => cpu_buffer(bytes),
        (BackendName::Cuda, Contract::DeviceOnly) | (BackendName::Vulkan, _) => bytes.max(1),
        (BackendName::Cuda, Contract::Upload | Contract::Shared | Contract::Words) => host_pages(bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_backend_charges_its_own_rounding() {
        let page = host_page();
        let over = CPU_CARVED_LIMIT + 1;
        assert_eq!(allocation_charge(BackendName::Metal, Contract::DeviceOnly, page), page);
        assert_eq!(allocation_charge(BackendName::Metal, Contract::Shared, page + 1), 2 * page);
        assert_eq!(allocation_charge(BackendName::Cpu, Contract::DeviceOnly, CPU_CARVED_LIMIT), CPU_CARVED_LIMIT);
        assert_eq!(allocation_charge(BackendName::Cpu, Contract::DeviceOnly, over), over.next_multiple_of(page));
        assert_eq!(allocation_charge(BackendName::Cuda, Contract::DeviceOnly, over), over);
        assert_eq!(allocation_charge(BackendName::Cuda, Contract::Words, 1), page);
        assert_eq!(allocation_charge(BackendName::Vulkan, Contract::Shared, 0), 1);
    }
}
