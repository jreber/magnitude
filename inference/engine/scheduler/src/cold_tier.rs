//! The SSD cold tier's pure core: the entry key, the byte encoding of one
//! entry file, and the in-memory index with prefix lookup and LRU victim
//! selection. No threads, no file or directory access.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use magnitude_family_contracts::TokenId;
use magnitude_state::ExportedState;

const MAGIC: [u8; 8] = *b"MAGCOLD1";
/// 2: lossless checkpoints — the meta JSON carries the region geometry
/// (`row_bytes`, `ranges`, head `tape`/`starts`) and the MTP draft rows that
/// version 1 dropped. 3: the draft's skipped target rows.
const FORMAT_VERSION: u32 = 3;

/// The 32-byte identity of one cold entry on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntryKey(pub [u8; 32]);

impl EntryKey {
    /// The key's 32 raw bytes.
    pub fn bytes(&self) -> [u8; 32] {
        self.0
    }

    /// The key as 64 lowercase hex characters.
    pub fn hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// Parses exactly 64 characters of `[0-9a-f]`; anything else is `None`.
    pub fn from_hex(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() != 64 || !bytes.iter().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            return None;
        }
        let mut value = [0u8; 32];
        for (i, byte) in value.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
        }
        Some(Self(value))
    }
}

/// One named blob carried by an entry, stored back to back after the table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobRegion {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// Everything one cold entry file carries: its key, the token path it was
/// spilled for, and the region blobs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpillPayload {
    pub key: EntryKey,
    pub tokens: Vec<TokenId>,
    pub position: usize,
    pub meta_json: String,
    pub regions: Vec<BlobRegion>,
}

/// Whether restoring `bytes` from disk beats the prefill it replaces:
/// the read time at 3 GB/s against `position` tokens at 1.5 ms/token.
pub fn restore_worthwhile(bytes: u64, position: usize) -> bool {
    bytes as f64 / 3e9 < position as f64 * 1.5e-3
}

/// Whether to restore a disk entry `disk_depth` tokens deep rather than the
/// memory hit `mem_depth` deep: it must save tokens, and the read must cost
/// less than the prefill of just those extra tokens.
pub fn prefer_disk(mem_depth: usize, disk_depth: usize, bytes: u64) -> bool {
    disk_depth > mem_depth && restore_worthwhile(bytes, disk_depth - mem_depth)
}

/// The key an entry is stored under: blake3 over each input prefixed by its
/// byte length (u64 little-endian), in order: identity, the token bytes,
/// conditioning, method UTF-8, position.
pub fn entry_key(
    identity: &[u8],
    tokens: &[TokenId],
    conditioning: &[u8],
    method: &str,
    position: usize,
) -> EntryKey {
    let mut digest = blake3::Hasher::new();
    let prefixed = |digest: &mut blake3::Hasher, bytes: &[u8]| {
        let len = (bytes.len() as u64).to_le_bytes();
        digest.update(&len);
        digest.update(bytes);
    };
    prefixed(&mut digest, identity);
    let token_bytes: Vec<u8> = tokens.iter().flat_map(|token| token.0.to_le_bytes()).collect();
    prefixed(&mut digest, &token_bytes);
    prefixed(&mut digest, conditioning);
    prefixed(&mut digest, method.as_bytes());
    prefixed(&mut digest, &(position as u64).to_le_bytes());
    EntryKey(*digest.finalize().as_bytes())
}

/// One row of the entry's JSON table. Offsets are relative to the first byte
/// after the table.
#[derive(serde::Serialize, serde::Deserialize)]
struct TableRegion {
    name: String,
    offset: u64,
    len: u64,
    blake3: String,
}

/// The entry's JSON table.
#[derive(serde::Serialize, serde::Deserialize)]
struct Table {
    key: String,
    tokens: Vec<u32>,
    position: usize,
    meta_json: String,
    regions: Vec<TableRegion>,
}

/// Encodes the payload as: 8-byte magic, u32 version, u32 header check,
/// u64 table length, the UTF-8 JSON table, then the region blobs back to
/// back. All integers little-endian.
pub fn encode_entry(payload: &SpillPayload) -> Vec<u8> {
    let mut blobs = Vec::new();
    let mut offset = 0u64;
    let table_regions = payload
        .regions
        .iter()
        .map(|region| {
            let row = TableRegion {
                name: region.name.clone(),
                offset,
                len: region.bytes.len() as u64,
                blake3: blake3::hash(&region.bytes).to_hex().to_string(),
            };
            offset += region.bytes.len() as u64;
            blobs.extend_from_slice(&region.bytes);
            row
        })
        .collect();
    let table = Table {
        key: payload.key.hex(),
        tokens: payload.tokens.iter().map(|token| token.0).collect(),
        position: payload.position,
        meta_json: payload.meta_json.clone(),
        regions: table_regions,
    };
    // Strings, integers and arrays only: serialization cannot fail.
    let table_bytes = serde_json::to_vec(&table).unwrap_or_default();
    let table_hash = blake3::hash(&table_bytes);
    let table_digest = table_hash.as_bytes();
    let header_check = u32::from_le_bytes([
        table_digest[0],
        table_digest[1],
        table_digest[2],
        table_digest[3],
    ]);
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&header_check.to_le_bytes());
    out.extend_from_slice(&(table_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&table_bytes);
    out.extend_from_slice(&blobs);
    out
}

/// Decodes and verifies an entry file. Every malformed shape returns `Err`;
/// the function never panics or indexes out of bounds.
pub fn decode_entry(bytes: &[u8]) -> Result<SpillPayload, String> {
    if bytes.len() < 24 {
        return Err("input under 24 bytes".into());
    }
    if bytes[0..8] != MAGIC {
        return Err("wrong magic".into());
    }
    let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if version != FORMAT_VERSION {
        return Err("wrong version".into());
    }
    let header_check = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let table_len = u64::from_le_bytes([
        bytes[16],
        bytes[17],
        bytes[18],
        bytes[19],
        bytes[20],
        bytes[21],
        bytes[22],
        bytes[23],
    ]);
    let table_end = 24u64.checked_add(table_len).ok_or("table length overflow")?;
    if table_end > bytes.len() as u64 {
        return Err("table length past end of input".into());
    }
    let table_bytes = &bytes[24..table_end as usize];
    let table_hash = blake3::hash(table_bytes);
    let table_digest = table_hash.as_bytes();
    let actual_check = u32::from_le_bytes([
        table_digest[0],
        table_digest[1],
        table_digest[2],
        table_digest[3],
    ]);
    if actual_check != header_check {
        return Err("bad header check".into());
    }
    let table: Table = serde_json::from_slice(table_bytes).map_err(|error| format!("bad table JSON: {error}"))?;
    let key = EntryKey::from_hex(&table.key).ok_or("bad key hex in table")?;
    let tokens = table.tokens.iter().map(|token| TokenId(*token)).collect();
    let blobs_start = table_end as usize;
    let mut regions = Vec::with_capacity(table.regions.len());
    for region in &table.regions {
        let start = blobs_start
            .checked_add(region.offset as usize)
            .ok_or("region offset overflow")?;
        let end = start
            .checked_add(region.len as usize)
            .ok_or("region end overflow")?;
        if end > bytes.len() {
            return Err("region out of bounds".into());
        }
        let blob = &bytes[start..end];
        let digest = blake3::hash(blob).to_hex().to_string();
        if digest != region.blake3 {
            return Err("region blake3 mismatch".into());
        }
        regions.push(BlobRegion {
            name: region.name.clone(),
            bytes: blob.to_vec(),
        });
    }
    let used = blobs_start
        .checked_add(table.regions.iter().map(|region| region.len as usize).sum::<usize>())
        .ok_or("blob total overflow")?;
    if used != bytes.len() {
        return Err("trailing bytes after the last region".into());
    }
    Ok(SpillPayload {
        key,
        tokens,
        position: table.position,
        meta_json: table.meta_json,
        regions,
    })
}

/// The method family a spilled entry was made with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MethodTag {
    Plain,
    Mtp,
    DFlash,
}

/// Where in the request the spilled prefix ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointTag {
    Branch,
    PromptBoundary,
    Terminal,
}

/// A checkpoint spilled losslessly: everything `StateStore::import` and
/// `DraftCheckpoint::from_spill` need to rebuild it on the device.
#[derive(Clone, Debug, PartialEq)]
pub struct SpilledCheckpoint {
    pub method: MethodTag,
    pub point: PointTag,
    pub target: ExportedState,
    pub head: Option<ExportedState>,
    /// The MTP draft's pending rows: one token per feature row, plus the
    /// feature bytes.
    pub pending: Option<(Vec<TokenId>, Vec<u8>)>,
    /// The MTP draft's open feature row bytes.
    pub open: Option<Vec<u8>>,
    /// The target rows the draft's history omits.
    pub skipped: usize,
}

/// The geometry of one exported history region, without its bytes.
#[derive(serde::Serialize, serde::Deserialize)]
struct MetaRegion {
    domain: usize,
    region: usize,
    row_bytes: usize,
    ranges: Vec<(usize, usize)>,
}

/// The geometry of one exported store.
#[derive(serde::Serialize, serde::Deserialize)]
struct MetaCheckpoint {
    position: usize,
    tape: usize,
    starts: Vec<usize>,
    history: Vec<MetaRegion>,
    bank: usize,
}

/// The lossless meta JSON of an entry.
#[derive(serde::Serialize, serde::Deserialize)]
struct SpillMeta {
    method: String,
    point: String,
    target: MetaCheckpoint,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    head: Option<MetaCheckpoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pending: Option<Vec<u32>>,
    open: bool,
    skipped: usize,
}

/// Builds the lossless payload for `checkpoint`: its regions become the
/// entry's blobs in fixed order — target history `h{i}`, target bank `b{i}`,
/// head history `hd{i}`, head bank `hb{i}`, then `pending` and `open` when
/// present — and its geometry goes into the meta JSON.
pub fn encode_spill(
    key: EntryKey,
    tokens: Vec<TokenId>,
    position: usize,
    checkpoint: SpilledCheckpoint,
) -> SpillPayload {
    let mut regions = Vec::new();
    let meta_target = checkpoint
        .target
        .history
        .iter()
        .map(|r| MetaRegion {
            domain: r.domain,
            region: r.region,
            row_bytes: r.row_bytes,
            ranges: r.ranges.clone(),
        })
        .collect::<Vec<_>>();
    let head_meta = checkpoint.head.as_ref().map(|head| {
        head.history
            .iter()
            .map(|r| MetaRegion {
                domain: r.domain,
                region: r.region,
                row_bytes: r.row_bytes,
                ranges: r.ranges.clone(),
            })
            .collect::<Vec<_>>()
    });
    let meta = SpillMeta {
        method: match checkpoint.method {
            MethodTag::Plain => "Plain".into(),
            MethodTag::Mtp => "Mtp".into(),
            MethodTag::DFlash => "DFlash".into(),
        },
        point: match checkpoint.point {
            PointTag::Branch => "Branch".into(),
            PointTag::PromptBoundary => "PromptBoundary".into(),
            PointTag::Terminal => "Terminal".into(),
        },
        target: MetaCheckpoint {
            position: checkpoint.target.position,
            tape: checkpoint.target.tape,
            starts: checkpoint.target.starts.clone(),
            history: meta_target,
            bank: checkpoint.target.bank.len(),
        },
        head: head_meta.map(|history| {
            let head = &checkpoint.head.as_ref().expect("checked above");
            MetaCheckpoint {
                position: head.position,
                tape: head.tape,
                starts: head.starts.clone(),
                history,
                bank: head.bank.len(),
            }
        }),
        pending: checkpoint.pending.as_ref().map(|(tokens, _)| {
            tokens.iter().map(|token| token.0).collect()
        }),
        open: checkpoint.open.is_some(),
        skipped: checkpoint.skipped,
    };
    let target = &checkpoint.target;
    for (i, region) in target.history.iter().enumerate() {
        regions.push(BlobRegion {
            name: format!("h{i}"),
            bytes: region.bytes.clone(),
        });
    }
    for (i, row) in target.bank.iter().enumerate() {
        regions.push(BlobRegion {
            name: format!("b{i}"),
            bytes: row.clone(),
        });
    }
    if let Some(head) = &checkpoint.head {
        for (i, region) in head.history.iter().enumerate() {
            regions.push(BlobRegion {
                name: format!("hd{i}"),
                bytes: region.bytes.clone(),
            });
        }
        for (i, row) in head.bank.iter().enumerate() {
            regions.push(BlobRegion {
                name: format!("hb{i}"),
                bytes: row.clone(),
            });
        }
    }
    if let Some((_, bytes)) = &checkpoint.pending {
        regions.push(BlobRegion {
            name: "pending".into(),
            bytes: bytes.clone(),
        });
    }
    if let Some(bytes) = &checkpoint.open {
        regions.push(BlobRegion {
            name: "open".into(),
            bytes: bytes.clone(),
        });
    }
    SpillPayload {
        key,
        tokens,
        position,
        meta_json: serde_json::to_string(&meta).unwrap_or_default(),
        regions,
    }
}

/// Rebuilds the spilled checkpoint from a decoded entry. The region names
/// must be exactly the ones [`encode_spill`] wrote, in order.
pub fn decode_spill(payload: &SpillPayload) -> Result<SpilledCheckpoint, String> {
    let meta: SpillMeta = serde_json::from_str(&payload.meta_json)
        .map_err(|error| format!("bad spill meta JSON: {error}"))?;
    let method = match meta.method.as_str() {
        "Plain" => MethodTag::Plain,
        "Mtp" => MethodTag::Mtp,
        "DFlash" => MethodTag::DFlash,
        other => return Err(format!("unknown spill method {other:?}")),
    };
    let point = match meta.point.as_str() {
        "Branch" => PointTag::Branch,
        "PromptBoundary" => PointTag::PromptBoundary,
        "Terminal" => PointTag::Terminal,
        other => return Err(format!("unknown spill point {other:?}")),
    };
    let has_pending = meta.pending.is_some();
    let has_open = meta.open;
    let expected: Vec<String> = {
        let mut names = Vec::new();
        for i in 0..meta.target.history.len() {
            names.push(format!("h{i}"));
        }
        for i in 0..meta.target.bank {
            names.push(format!("b{i}"));
        }
        if let Some(head) = &meta.head {
            for i in 0..head.history.len() {
                names.push(format!("hd{i}"));
            }
            for i in 0..head.bank {
                names.push(format!("hb{i}"));
            }
        }
        if has_pending {
            names.push("pending".into());
        }
        if has_open {
            names.push("open".into());
        }
        names
    };
    let names: Vec<String> = payload.regions.iter().map(|r| r.name.clone()).collect();
    if names != expected {
        return Err("spill regions do not match the meta".into());
    }
    let rebuild =
        |meta: &MetaCheckpoint, offset: usize, regions: &[BlobRegion]| -> Result<ExportedState, String> {
            let mut cursor = offset;
            let history = meta
                .history
                .iter()
                .map(|m| {
                    let blob = &regions[cursor];
                    cursor += 1;
                    let rows = m.ranges.iter().map(|(_, count)| count).sum::<usize>();
                    let want = rows.checked_mul(m.row_bytes).ok_or("region size overflow")?;
                    if blob.bytes.len() != want {
                        return Err("region byte length disagrees with its ranges".into());
                    }
                    Ok(magnitude_state::ExportedRegion {
                        domain: m.domain,
                        region: m.region,
                        row_bytes: m.row_bytes,
                        ranges: m.ranges.clone(),
                        bytes: blob.bytes.clone(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let bank = regions[cursor..cursor + meta.bank]
                .iter()
                .map(|r| r.bytes.clone())
                .collect::<Vec<_>>();
            Ok(ExportedState {
                position: meta.position,
                tape: meta.tape,
                starts: meta.starts.clone(),
                history,
                bank,
            })
        };
    let target = rebuild(&meta.target, 0, &payload.regions)?;
    let mut cursor = meta.target.history.len() + meta.target.bank;
    let head = meta
        .head
        .as_ref()
        .map(|m| rebuild(m, cursor, &payload.regions))
        .transpose()?;
    if meta.head.is_some() {
        cursor += meta.head.as_ref().unwrap().history.len() + meta.head.as_ref().unwrap().bank;
    }
    let pending = match &meta.pending {
        Some(tokens) => {
            let blob = payload.regions.get(cursor).ok_or("missing pending blob")?;
            cursor += 1;
            Some((
                tokens.iter().map(|token| TokenId(*token)).collect(),
                blob.bytes.clone(),
            ))
        }
        None => None,
    };
    let open = if meta.open {
        let blob = payload.regions.get(cursor).ok_or("missing open blob")?;
        cursor += 1;
        Some(blob.bytes.clone())
    } else {
        None
    };
    if cursor != payload.regions.len() {
        return Err("trailing spill regions".into());
    }
    Ok(SpilledCheckpoint {
        method,
        point,
        target,
        head,
        pending,
        open,
        skipped: meta.skipped,
    })
}

/// One index row: the token path, the entry's size, and its last-use tick.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct IndexEntry {
    pub tokens: Vec<u32>,
    pub bytes: u64,
    pub last_use: u64,
    /// The checkpoint's method ("Plain", "Mtp", "DFlash"); rows written
    /// before this field existed read as `None` and match any method.
    #[serde(default)]
    pub method: Option<String>,
}

/// The in-memory picture of the cold tier: which keys are on disk, how large
/// they are, and which to evict first.
#[derive(Debug, Default)]
pub struct ColdIndex {
    entries: HashMap<EntryKey, IndexEntry>,
    clock: u64,
    /// Keys whose file write has not finished: indexed (they count toward
    /// the budget) but not yet loadable.
    pending: HashSet<EntryKey>,
}

impl ColdIndex {
    /// Records the key; `last_use` takes the next clock tick.
    pub fn insert(&mut self, key: EntryKey, tokens: &[TokenId], bytes: u64) {
        self.clock += 1;
        self.entries.insert(
            key,
            IndexEntry {
                tokens: tokens.iter().map(|token| token.0).collect(),
                bytes,
                last_use: self.clock,
                method: None,
            },
        );
    }

    /// Records the method of an indexed key.
    pub fn set_method(&mut self, key: &EntryKey, method: &str) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.method = Some(method.to_owned());
        }
    }

    /// Marks the key's file as not yet written.
    pub fn mark_pending(&mut self, key: EntryKey) {
        self.pending.insert(key);
    }

    /// Marks the key's file write as finished, successfully or not.
    pub fn clear_pending(&mut self, key: &EntryKey) {
        self.pending.remove(key);
    }

    pub fn is_pending(&self, key: &EntryKey) -> bool {
        self.pending.contains(key)
    }

    /// Bumps the key's `last_use` to the next clock tick; false if absent.
    pub fn touch(&mut self, key: &EntryKey) -> bool {
        match self.entries.get_mut(key) {
            Some(entry) => {
                self.clock += 1;
                entry.last_use = self.clock;
                true
            }
            None => false,
        }
    }

    /// Forgets the key, returning its row.
    pub fn remove(&mut self, key: &EntryKey) -> Option<IndexEntry> {
        self.pending.remove(key);
        self.entries.remove(key)
    }

    /// The bytes the indexed entries occupy.
    pub fn total_bytes(&self) -> u64 {
        self.entries.values().map(|entry| entry.bytes).sum()
    }

    /// The keys to evict, least-recently used first, until the remaining
    /// bytes fit the budget. A zero budget names every key.
    pub fn victims(&self, budget: u64) -> Vec<EntryKey> {
        let total = self.total_bytes();
        let mut order: Vec<&EntryKey> = self.entries.keys().collect();
        order.sort_by_key(|key| self.entries[*key].last_use);
        let mut victims = Vec::new();
        let mut freed = 0u64;
        for key in order {
            if total - freed <= budget {
                break;
            }
            freed += self.entries[key].bytes;
            victims.push(*key);
        }
        victims
    }

    /// The entry whose token path is the longest prefix of the query shorter
    /// than `bound`, whose method is `method` (a row with no recorded method
    /// matches any) and whose file is written; ties go to the larger
    /// `last_use`. Empty-token entries never match. Returns the key, the
    /// prefix token count, and the entry's on-disk byte count.
    pub fn lookup_deepest(
        &self,
        tokens: &[TokenId],
        bound: usize,
        method: Option<&str>,
    ) -> Option<(EntryKey, usize, u64)> {
        let query: Vec<u32> = tokens.iter().map(|token| token.0).collect();
        let mut best: Option<(&EntryKey, &IndexEntry)> = None;
        for (key, entry) in &self.entries {
            if entry.tokens.is_empty()
                || entry.tokens.len() >= bound
                || self.pending.contains(key)
                || !query.starts_with(&entry.tokens)
                || matches!((method, entry.method.as_deref()), (Some(want), Some(have)) if want != have)
            {
                continue;
            }
            let better = match best {
                None => true,
                Some((_, current)) => entry.tokens.len() > current.tokens.len()
                    || (entry.tokens.len() == current.tokens.len()
                        && entry.last_use > current.last_use),
            };
            if better {
                best = Some((key, entry));
            }
        }
        best.map(|(key, entry)| (*key, entry.tokens.len(), entry.bytes))
    }

    /// Every indexed key.
    pub fn keys(&self) -> Vec<EntryKey> {
        self.entries.keys().copied().collect()
    }

    /// Serializes the entries and the clock; keys as hex strings.
    pub fn to_json(&self) -> String {
        #[derive(serde::Serialize)]
        struct Snapshot {
            clock: u64,
            entries: Vec<Row>,
        }
        #[derive(serde::Serialize)]
        struct Row {
            key: String,
            entry: IndexEntry,
        }
        let snapshot = Snapshot {
            clock: self.clock,
            entries: self
                .entries
                .iter()
                .map(|(key, entry)| Row {
                    key: key.hex(),
                    entry: entry.clone(),
                })
                .collect(),
        };
        // Strings, integers and arrays only: serialization cannot fail.
        serde_json::to_string(&snapshot).unwrap_or_default()
    }

    /// Reverses `to_json`; a key that is not 64 lowercase hex is an error.
    pub fn from_json(s: &str) -> Result<Self, String> {
        #[derive(serde::Deserialize)]
        struct Snapshot {
            clock: u64,
            entries: Vec<Row>,
        }
        #[derive(serde::Deserialize)]
        struct Row {
            key: String,
            entry: IndexEntry,
        }
        let snapshot: Snapshot = serde_json::from_str(s).map_err(|error| error.to_string())?;
        let mut entries = HashMap::new();
        for row in snapshot.entries {
            let key = EntryKey::from_hex(&row.key).ok_or("bad key hex in index JSON")?;
            entries.insert(key, row.entry);
        }
        Ok(Self { entries, clock: snapshot.clock, pending: HashSet::new() })
    }
}

/// One unit of work the writer thread performs. Only `Vec<u8>`, `String`,
/// `PathBuf` and `EntryKey` cross the thread boundary.
pub enum WriteJob {
    Write { key: EntryKey, bytes: Vec<u8> },
    Delete { key: EntryKey },
    Index { json: String },
}

/// The writer's report for one `Write` job.
pub enum WriteResult {
    Written { key: EntryKey },
    Failed { key: EntryKey, error: String },
}

/// Whether `pid` names a running process.
fn pid_alive(pid: u32) -> bool {
    // SAFETY: signal 0 sends nothing; it only asks the kernel for its
    // permission check, and we interpret only the errno it leaves behind.
    if unsafe { libc::kill(pid as libc::pid_t, 0) } == -1
        && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
    {
        false
    } else {
        true
    }
}

/// Claims `<dir>/.lock`. Returns `(can_spill, holds_lock)`: a fresh or stale
/// lock is taken and `true`/`true`; a live lock (or a lost retry race) is
/// restore-only.
fn acquire_lock(dir: &Path) -> std::io::Result<(bool, bool)> {
    let path = dir.join(".lock");
    let claim = |path: &Path| -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(std::process::id().to_string().as_bytes())
    };
    match claim(&path) {
        Ok(()) => Ok((true, true)),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            let stale = match std::fs::read_to_string(&path).ok().and_then(|pid| pid.trim().parse::<u32>().ok()) {
                Some(pid) => !pid_alive(pid),
                None => true,
            };
            if stale {
                let _ = std::fs::remove_file(&path);
                match claim(&path) {
                    Ok(()) => Ok((true, true)),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                        tracing::warn!("cold tier lock held at {}; restore-only", path.display());
                        Ok((false, false))
                    }
                    Err(e) => Err(e),
                }
            } else {
                tracing::warn!("cold tier lock held at {}; restore-only", path.display());
                Ok((false, false))
            }
        }
        Err(e) => Err(e),
    }
}

/// Removes torn state: `.tmp` files under `dir` and `dir/entries`, index
/// rows whose `.bin` is missing, and `.bin` files with an invalid or
/// unknown stem. Per-file removal failures are ignored; read errors
/// propagate. Never decodes files.
fn recover(dir: &Path, index: &mut ColdIndex) -> std::io::Result<()> {
    for location in [dir, &dir.join("entries")] {
        for entry in std::fs::read_dir(location)? {
            let entry = entry?;
            if entry.file_name().to_string_lossy().ends_with(".tmp") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    for key in index.keys() {
        if !dir.join("entries").join(format!("{}.bin", key.hex())).exists() {
            index.remove(&key);
        }
    }
    let entries = dir.join("entries");
    for entry in std::fs::read_dir(&entries)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let stem = match name.strip_suffix(".bin") {
            Some(stem) => stem,
            None => continue,
        };
        let known = EntryKey::from_hex(stem).is_some_and(|key| index.keys().contains(&key));
        if !known {
            let _ = std::fs::remove_file(entry.path());
        }
    }
    Ok(())
}

/// Writes `bytes` to `path` atomically: a `.tmp` sibling, `sync_all`, rename
/// onto `path`, then a sync of the parent directory. Never creates directories.
fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    std::fs::rename(&tmp, path)?;
    if let Some(parent) = path.parent() {
        std::fs::File::open(parent)?.sync_all()?;
    }
    Ok(())
}

/// The SSD cold tier: `ColdIndex` and the byte budget live on the caller
/// thread; every file write and delete is handed to one writer thread. No
/// startup recovery or locking here (unit 03c).
pub struct ColdTier {
    dir: PathBuf,
    budget: u64,
    index: ColdIndex,
    jobs: Option<mpsc::Sender<WriteJob>>,
    results: mpsc::Receiver<WriteResult>,
    writer: Option<std::thread::JoinHandle<()>>,
    can_spill: bool,
    holds_lock: bool,
}

impl ColdTier {
    /// Creates `dir/entries`, takes the lock (a live one makes this tier
    /// restore-only), loads `dir/index.json`, repairs torn state when
    /// allowed to spill, and starts the writer thread.
    pub fn open(dir: PathBuf, budget_bytes: u64) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir.join("entries"))?;
        let (can_spill, holds_lock) = acquire_lock(&dir)?;
        let mut index = std::fs::read_to_string(dir.join("index.json"))
            .ok()
            .and_then(|json| ColdIndex::from_json(&json).ok())
            .unwrap_or_default();
        if can_spill {
            recover(&dir, &mut index)?;
            atomic_write(&dir.join("index.json"), index.to_json().as_bytes())?;
        } else {
            for key in index.keys() {
                if !dir.join("entries").join(format!("{}.bin", key.hex())).exists() {
                    index.remove(&key);
                }
            }
        }
        if !can_spill {
            return Ok(Self {
                dir,
                budget: budget_bytes,
                index,
                jobs: None,
                results: mpsc::channel::<WriteResult>().1,
                writer: None,
                can_spill,
                holds_lock,
            });
        }
        let (jobs_tx, jobs_rx) = mpsc::channel();
        let (results_tx, results_rx) = mpsc::channel();
        let writer_dir = dir.clone();
        let writer = seismic_threads::spawn("cold-tier writer", move || {
            for job in jobs_rx {
                match job {
                    WriteJob::Write { key, bytes } => {
                        let path = writer_dir.join("entries").join(format!("{}.bin", key.hex()));
                        let started = std::time::Instant::now();
                        let size = bytes.len();
                        let result = match atomic_write(&path, &bytes) {
                            Ok(()) => {
                                tracing::info!(
                                    bytes = size,
                                    write_ms = started.elapsed().as_millis() as u64,
                                    "cold-tier write"
                                );
                                WriteResult::Written { key }
                            }
                            Err(e) => WriteResult::Failed { key, error: e.to_string() },
                        };
                        let _ = results_tx.send(result);
                    }
                    WriteJob::Delete { key } => {
                        let path = writer_dir.join("entries").join(format!("{}.bin", key.hex()));
                        if let Err(e) = std::fs::remove_file(&path) {
                            if e.kind() != std::io::ErrorKind::NotFound {
                                tracing::warn!("cold tier delete {}: {e}", key.hex());
                            }
                        }
                    }
                    WriteJob::Index { json } => {
                        let path = writer_dir.join("index.json");
                        if let Err(e) = atomic_write(&path, json.as_bytes()) {
                            tracing::warn!("cold tier index write: {e}");
                        }
                    }
                }
            }
        })?;
        Ok(Self {
            dir,
            budget: budget_bytes,
            index,
            jobs: Some(jobs_tx),
            results: results_rx,
            writer: Some(writer),
            can_spill,
            holds_lock,
        })
    }

    /// Whether `spill` will write; `open` always sets it true in this unit.
    pub fn can_spill(&self) -> bool {
        self.can_spill
    }

    /// Records the payload, queues its write, evicts down to the budget, and
    /// queues the index update. A dropped writer is ignored.
    pub fn spill(&mut self, payload: SpillPayload, method: &str) {
        if !self.can_spill {
            return;
        }
        let key = payload.key;
        let bytes = encode_entry(&payload);
        self.index.insert(key, &payload.tokens, bytes.len() as u64);
        self.index.set_method(&key, method);
        if let Some(tx) = &self.jobs {
            self.index.mark_pending(key);
            let _ = tx.send(WriteJob::Write { key, bytes });
        }
        for victim in self.index.victims(self.budget) {
            self.index.remove(&victim);
            if let Some(tx) = &self.jobs {
                let _ = tx.send(WriteJob::Delete { key: victim });
            }
        }
        if let Some(tx) = &self.jobs {
            let _ = tx.send(WriteJob::Index { json: self.index.to_json() });
        }
    }

    /// Reconciles completed writes: a `Failed` result is dropped from the
    /// index, its file deleted, and the index re-sent.
    pub fn drain_results(&mut self) {
        while let Ok(result) = self.results.try_recv() {
            if let WriteResult::Written { key } = result {
                self.index.clear_pending(&key);
            } else if let WriteResult::Failed { key, error } = result {
                self.index.remove(&key);
                if let Some(tx) = &self.jobs {
                    let _ = tx.send(WriteJob::Delete { key });
                    let _ = tx.send(WriteJob::Index { json: self.index.to_json() });
                }
                tracing::warn!("cold tier write {}: {error}", key.hex());
            }
        }
    }

    /// The deepest indexed prefix of `tokens`: the key, its prefix token
    /// count, and the entry's on-disk byte count.
    pub fn lookup_deepest(
        &self,
        tokens: &[TokenId],
        bound: usize,
        method: Option<&str>,
    ) -> Option<(EntryKey, usize, u64)> {
        self.index.lookup_deepest(tokens, bound, method)
    }

    /// Reads and decodes the entry `key` names on a spawned reader thread;
    /// the caller blocks on the result. On any error the key is evicted and
    /// its file deleted; on success the key is touched and the index re-sent.
    pub fn load(&mut self, key: &EntryKey) -> Result<SpillPayload, String> {
        if self.index.is_pending(key) {
            // The file is still being written: not a corrupt entry, so keep it.
            return Err("entry write still pending".into());
        }
        let path = self.dir.join("entries").join(format!("{}.bin", key.hex()));
        let read_key = *key;
        let (tx, rx) = std::sync::mpsc::channel();
        seismic_threads::spawn("cold-tier reader", move || {
            let outcome = (|| -> Result<SpillPayload, String> {
                let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
                let decoded = decode_entry(&bytes)?;
                if decoded.key != read_key {
                    return Err("entry key mismatch".into());
                }
                Ok(decoded)
            })();
            let _ = tx.send(outcome);
        })
        .map_err(|error| error.to_string())?;
        let outcome = rx.recv().expect("cold-tier read thread panicked");
        match outcome {
            Ok(decoded) => {
                self.index.touch(key);
                if let Some(tx) = &self.jobs {
                    let _ = tx.send(WriteJob::Index { json: self.index.to_json() });
                }
                Ok(decoded)
            }
            Err(error) => {
                self.index.remove(key);
                if let Some(tx) = &self.jobs {
                    let _ = tx.send(WriteJob::Delete { key: *key });
                    let _ = tx.send(WriteJob::Index { json: self.index.to_json() });
                }
                Err(error)
            }
        }
    }

    /// Wait for every queued write, deletion and index update to reach the
    /// disk; the tier spills nothing after. An ending owner finishes its
    /// tier before its stop is announced, since the process may end without
    /// dropping it.
    pub fn finish(&mut self) {
        self.jobs = None;
        self.can_spill = false;
        if let Some(handle) = self.writer.take() {
            let _ = handle.join();
        }
    }

    /// Evicts the key: drops its index row and queues the file deletion and the
    /// index update. A missing key is a no-op.
    pub fn remove(&mut self, key: &EntryKey) {
        if self.index.remove(key).is_none() {
            return;
        }
        if let Some(tx) = &self.jobs {
            let _ = tx.send(WriteJob::Delete { key: *key });
            let _ = tx.send(WriteJob::Index { json: self.index.to_json() });
        }
    }
}

impl Drop for ColdTier {
    fn drop(&mut self) {
        self.finish();
        if self.holds_lock {
            let _ = std::fs::remove_file(self.dir.join(".lock"));
        }
    }
}

/// Removes identity dirs under `state_dir`, least recently used first (by
/// `index.json` mtime), until the sum of all dirs fits `total_budget`. The
/// `current` identity and any dir whose `.lock` names a live process are kept.
/// Returns the dirs it removed.
pub fn collect_identity_dirs(
    state_dir: &Path,
    current: &str,
    total_budget: u64,
) -> std::io::Result<Vec<PathBuf>> {
    let is_identity = |name: &str| name.len() == 64 && name.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    let size = |dir: &Path| -> u64 {
        let entries = std::fs::read_dir(dir.join("entries"))
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| entry.metadata().ok())
            .map(|meta| meta.len())
            .sum::<u64>();
        entries + std::fs::metadata(dir.join("index.json")).map_or(0, |meta| meta.len())
    };
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(state_dir)?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !is_identity(&name) || !entry.path().is_dir() {
            continue;
        }
        let used = std::fs::metadata(entry.path().join("index.json"))
            .and_then(|meta| meta.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        dirs.push((used, name, entry.path()));
    }
    let sizes: Vec<u64> = dirs.iter().map(|(_, _, path)| size(path)).collect();
    let mut total: u64 = sizes.iter().sum();
    let mut order: Vec<usize> = (0..dirs.len()).collect();
    order.sort_by_key(|&i| dirs[i].0);
    let mut removed = Vec::new();
    for i in order {
        if total <= total_budget {
            break;
        }
        let (used, name, path) = &dirs[i];
        let locked = std::fs::read_to_string(path.join(".lock"))
            .ok()
            .and_then(|pid| pid.trim().parse::<u32>().ok())
            .is_some_and(pid_alive);
        if name == current || locked {
            continue;
        }
        match std::fs::remove_dir_all(path) {
            Ok(()) => {
                total -= sizes[i];
                tracing::info!(
                    dir = %path.display(),
                    bytes = sizes[i],
                    age_s = used.elapsed().map_or(0, |age| age.as_secs()),
                    "cold-tier gc removed"
                );
                removed.push(path.clone());
            }
            Err(error) => tracing::warn!(dir = %path.display(), %error, "cold-tier gc could not remove"),
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(ids: &[u32]) -> Vec<TokenId> {
        ids.iter().map(|id| TokenId(*id)).collect()
    }

    #[test]
    fn key_is_stable_and_input_sensitive() {
        let base = entry_key(b"id", &tokens(&[1, 2, 3]), b"cond", "auto", 42);
        assert_eq!(base, entry_key(b"id", &tokens(&[1, 2, 3]), b"cond", "auto", 42));
        assert_ne!(base, entry_key(b"id2", &tokens(&[1, 2, 3]), b"cond", "auto", 42));
        assert_ne!(base, entry_key(b"id", &tokens(&[1, 2, 4]), b"cond", "auto", 42));
        assert_ne!(base, entry_key(b"id", &tokens(&[1, 2, 3]), b"cond2", "auto", 42));
        assert_ne!(base, entry_key(b"id", &tokens(&[1, 2, 3]), b"cond", "draft", 42));
        assert_ne!(base, entry_key(b"id", &tokens(&[1, 2, 3]), b"cond", "auto", 43));
        assert_eq!(EntryKey::from_hex(&base.hex()), Some(base));
        assert_eq!(EntryKey::from_hex("zz"), None);
        assert_eq!(EntryKey::from_hex(&base.hex()[..63]), None);
        assert_eq!(EntryKey::from_hex(&base.hex().to_uppercase()), None);
    }

    #[test]
    fn entry_round_trips() {
        let payload = SpillPayload {
            key: entry_key(b"model", &tokens(&[7, 8]), b"ctx", "auto", 1),
            tokens: tokens(&[7, 8, 9]),
            position: 3,
            meta_json: "{\"k\":\"v\"}".into(),
            regions: vec![
                BlobRegion {
                    name: "a".into(),
                    bytes: vec![1, 2, 3],
                },
                BlobRegion {
                    name: "empty".into(),
                    bytes: vec![],
                },
            ],
        };
        let decoded = decode_entry(&encode_entry(&payload)).expect("round trip");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn decode_rejects_corruption() {
        let payload = SpillPayload {
            key: entry_key(b"m", &tokens(&[1]), b"c", "auto", 0),
            tokens: tokens(&[1, 2]),
            position: 2,
            meta_json: "{}".into(),
            regions: vec![BlobRegion {
                name: "r".into(),
                bytes: vec![9, 9],
            }],
        };
        let bytes = encode_entry(&payload);

        let mut bad = bytes.clone();
        bad[0] ^= 0xff; // magic
        assert!(decode_entry(&bad).is_err());

        bad = bytes.clone();
        bad[8] ^= 0xff; // version
        assert!(decode_entry(&bad).is_err());

        bad = bytes.clone();
        bad[24] ^= 0xff; // table
        assert!(decode_entry(&bad).is_err());

        bad = bytes.clone();
        bad[bytes.len() - 1] ^= 0xff; // blob
        assert!(decode_entry(&bad).is_err());

        assert!(decode_entry(&bytes[..bytes.len() - 1]).is_err());
        let mut long = bytes.clone();
        long.push(0);
        assert!(decode_entry(&long).is_err());
    }

    #[test]
    fn decode_rejects_version_one_files() {
        let payload = SpillPayload {
            key: entry_key(b"m", &tokens(&[1]), b"c", "Plain", 0),
            tokens: tokens(&[1]),
            position: 0,
            meta_json: "{}".into(),
            regions: Vec::new(),
        };
        let mut bytes = encode_entry(&payload);
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        let error = decode_entry(&bytes).unwrap_err();
        assert_eq!(error, "wrong version");
    }

    #[test]
    fn spill_round_trips_a_full_mtp_checkpoint() {
        use magnitude_state::ExportedRegion;
        fn region(domain: usize, region: usize, row_bytes: usize, ranges: &[(usize, usize)]) -> ExportedRegion {
            let bytes: Vec<u8> = (0..ranges.iter().map(|(_, count)| count).sum::<usize>() * row_bytes)
                .map(|i| i as u8)
                .collect();
            ExportedRegion { domain, region, row_bytes, ranges: ranges.to_vec(), bytes }
        }
        fn state(position: usize, history: Vec<ExportedRegion>, bank: Vec<Vec<u8>>) -> ExportedState {
            ExportedState {
                position,
                tape: 3,
                starts: vec![0, position],
                history,
                bank,
            }
        }
        let checkpoint = SpilledCheckpoint {
            method: MethodTag::Mtp,
            point: PointTag::Branch,
            target: state(
                10,
                vec![
                    region(0, 0, 7, &[(0, 4), (6, 9)]),
                    region(1, 2, 5, &[(2, 10)]),
                ],
                vec![vec![1, 2, 3], vec![4, 5]],
            ),
            head: Some(state(
                9,
                vec![region(0, 0, 7, &[(0, 9)])],
                vec![vec![9, 9]],
            )),
            pending: Some((vec![TokenId(11), TokenId(12)], vec![7; 21])),
            open: Some(vec![8; 13]),
            skipped: 2,
        };
        let key = entry_key(b"m", &tokens(&[1, 2, 3]), b"", "Mtp", 10);
        let payload = encode_spill(key, tokens(&[1, 2, 3]), 10, checkpoint.clone());
        assert_eq!(payload.regions.len(), 2 + 2 + 1 + 1 + 1 + 1);
        let bytes = encode_entry(&payload);
        let decoded = decode_entry(&bytes).expect("entry decodes");
        assert_eq!(decode_spill(&decoded).expect("spill decodes"), checkpoint);
    }

    #[test]
    fn spill_rejects_plain_and_mismatched_layouts() {
        use magnitude_state::ExportedRegion;
        let checkpoint = SpilledCheckpoint {
            method: MethodTag::Plain,
            point: PointTag::Terminal,
            target: ExportedState {
                position: 1,
                tape: 0,
                starts: vec![0],
                history: vec![ExportedRegion {
                    domain: 0,
                    region: 0,
                    row_bytes: 2,
                    ranges: vec![(0, 3)],
                    bytes: vec![1; 6],
                }],
                bank: vec![vec![2]],
            },
            head: None,
            pending: None,
            open: None,
            skipped: 0,
        };
        let key = entry_key(b"m", &tokens(&[1]), b"", "Plain", 1);
        let payload = encode_spill(key, tokens(&[1]), 1, checkpoint.clone());
        let round = decode_spill(&decode_entry(&encode_entry(&payload))
            .expect("plain entry decodes"))
            .expect("plain spill decodes");
        assert_eq!(round, checkpoint);

        // A plain entry must not gain MTP draft rows.
        let mut mtp = round.clone();
        mtp.method = MethodTag::Mtp;
        mtp.pending = Some((vec![TokenId(5)], vec![0; 4]));
        let forged = encode_spill(key, tokens(&[1]), 1, mtp);
        let decoded = decode_entry(&encode_entry(&forged)).expect("forged entry decodes");
        let rebuilt = decode_spill(&decoded).expect("forged spill decodes");
        assert_eq!(rebuilt.method, MethodTag::Mtp);
        assert_eq!(rebuilt.pending, Some((vec![TokenId(5)], vec![0; 4])));

        // A tampered region name fails the layout check.
        let mut payload = payload.clone();
        payload.regions[0].name = "h00".into();
        assert!(decode_spill(&payload).is_err());

        // A blob shorter than its geometry fails the size check.
        let mut payload = payload.clone();
        payload.regions[0].bytes.truncate(3);
        assert!(decode_spill(&payload).is_err());
    }

    #[test]
    fn lookup_picks_deepest_prefix() {
        let mut index = ColdIndex::default();
        let t3 = tokens(&[1, 2, 3]);
        let t5 = tokens(&[1, 2, 3, 4, 5]);
        let t8 = tokens(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let other = tokens(&[9, 8, 7, 6]);
        let k3 = entry_key(b"a", &t3, b"", "auto", 0);
        let k5 = entry_key(b"b", &t5, b"", "auto", 0);
        let k8 = entry_key(b"c", &t8, b"", "auto", 0);
        index.insert(k3, &t3, 3);
        index.insert(k5, &t5, 5);
        index.insert(k8, &t8, 8);
        index.insert(entry_key(b"d", &other, b"", "auto", 0), &other, 4);

        let query10 = tokens(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(index.lookup_deepest(&query10, usize::MAX, None), Some((k8, 8, 8)));
        let query6 = tokens(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(index.lookup_deepest(&query6, usize::MAX, None), Some((k5, 5, 5)));
        assert_eq!(index.lookup_deepest(&tokens(&[5, 5, 5]), usize::MAX, None), None);
    }

    #[test]
    fn restore_gate_compares_read_time_to_prefill() {
        // 3 GB read (1 ms) vs 3000 tokens of prefill (4.5 ms): worth it.
        assert!(restore_worthwhile(3_000_000_000, 3000));
        // 30 GB read (10 ms) vs 100 tokens of prefill (0.15 ms): not.
        assert!(!restore_worthwhile(30_000_000_000, 100));
        // Nothing to prefill: never worth it.
        assert!(!restore_worthwhile(1, 0));
    }

    #[test]
    fn victims_are_least_recent_first() {
        let mut index = ColdIndex::default();
        let first = entry_key(b"1", &tokens(&[1]), b"", "auto", 0);
        let second = entry_key(b"2", &tokens(&[2]), b"", "auto", 0);
        let third = entry_key(b"3", &tokens(&[3]), b"", "auto", 0);
        index.insert(first, &tokens(&[1]), 10);
        index.insert(second, &tokens(&[2]), 10);
        index.insert(third, &tokens(&[3]), 10);
        assert!(index.touch(&first));
        assert_eq!(index.victims(15), vec![second, third]);
        assert_eq!(index.victims(0), vec![second, third, first]);
    }

    #[test]
    fn index_json_round_trips() {
        let mut index = ColdIndex::default();
        let a = entry_key(b"a", &tokens(&[1]), b"", "auto", 0);
        let b = entry_key(b"b", &tokens(&[2, 2]), b"", "auto", 0);
        index.insert(a, &tokens(&[1]), 5);
        index.insert(b, &tokens(&[2, 2]), 7);
        index.touch(&a);
        let mut decoded = ColdIndex::from_json(&index.to_json()).expect("round trip");
        assert_eq!(decoded.total_bytes(), 12);
        assert_eq!(decoded.lookup_deepest(&tokens(&[2, 2, 3]), usize::MAX, None).map(|(_, n, _)| n), Some(2));
        assert_eq!(decoded.remove(&a).expect("a present").last_use, 3);
        assert!(decoded.remove(&b).is_some());
        assert!(decoded.remove(&b).is_none());
        let bad_key = "{\"clock\":1,\"entries\":[{\"key\":\"zz\",\"entry\":{\"tokens\":[],\"bytes\":1,\"last_use\":1}}]}";
        assert!(ColdIndex::from_json(bad_key).is_err());
    }

    fn sample_payload(n: u32, region_bytes: u64) -> SpillPayload {
        let identity = format!("model-{n}").into_bytes();
        let t = tokens(&[n, n + 1, n + 2]);
        SpillPayload {
            key: entry_key(&identity, &t, b"c", "auto", 0),
            tokens: t,
            position: 0,
            meta_json: "{}".into(),
            regions: vec![BlobRegion {
                name: "kv".into(),
                bytes: vec![0u8; region_bytes as usize],
            }],
        }
    }

    fn tier_dir(test_name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cold-tier-{}-{test_name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn spill_then_load_round_trips() {
        let dir = tier_dir("spill_then_load_round_trips");
        let payload = sample_payload(1, 16);
        {
            let mut tier = ColdTier::open(dir.clone(), 1024 * 1024).expect("open");
            tier.spill(payload.clone(), "Plain");
            tier.drain_results();
        }
        let mut tier = ColdTier::open(dir.clone(), 1024 * 1024).expect("open");
        let (key, len, _) = tier.lookup_deepest(&tokens(&[1, 2, 3]), usize::MAX, None).expect("found");
        assert_eq!(len, 3);
        assert_eq!(tier.load(&key).expect("load"), payload);
        drop(tier);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn budget_evicts_least_recent() {
        let dir = tier_dir("budget_evicts_least_recent");
        let p1 = sample_payload(10, 64);
        let p2 = sample_payload(11, 64);
        let p3 = sample_payload(12, 64);
        let budget = (2 * encode_entry(&p1).len()) as u64;
        {
            let mut tier = ColdTier::open(dir.clone(), budget).expect("open");
            tier.spill(p1.clone(), "Plain");
            tier.spill(p2.clone(), "Plain");
            tier.spill(p3.clone(), "Plain");
            tier.drain_results();
        }
        let mut tier = ColdTier::open(dir.clone(), budget).expect("open");
        assert!(tier.lookup_deepest(&tokens(&[10, 11, 12]), usize::MAX, None).is_none());
        assert!(tier.lookup_deepest(&tokens(&[11, 12, 13]), usize::MAX, None).is_some());
        assert!(tier.lookup_deepest(&tokens(&[12, 13, 14]), usize::MAX, None).is_some());
        let first_bin = dir.join("entries").join(format!("{}.bin", p1.key.hex()));
        assert!(!first_bin.exists());
        drop(tier);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_entry_fails_load_and_is_removed() {
        let dir = tier_dir("corrupt_entry_fails_load_and_is_removed");
        let payload = sample_payload(20, 16);
        {
            let mut tier = ColdTier::open(dir.clone(), 1024 * 1024).expect("open");
            tier.spill(payload.clone(), "Plain");
            tier.drain_results();
        }
        let bin = dir.join("entries").join(format!("{}.bin", payload.key.hex()));
        let mut bytes = std::fs::read(&bin).expect("read bin");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&bin, &bytes).expect("write bin");
        let mut tier = ColdTier::open(dir.clone(), 1024 * 1024).expect("open");
        assert!(tier.load(&payload.key).is_err());
        assert!(tier.lookup_deepest(&tokens(&[20, 21, 22]), usize::MAX, None).is_none());
        drop(tier);
        assert!(!bin.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_recovers_torn_state() {
        let dir = tier_dir("open_recovers_torn_state");
        std::fs::create_dir_all(dir.join("entries")).unwrap();
        std::fs::write(dir.join("entries").join("x.tmp"), b"torn").unwrap();
        let orphan_key = "ab".repeat(32);
        std::fs::write(dir.join("entries").join(format!("{orphan_key}.bin")), b"orphan").unwrap();
        std::fs::write(dir.join("entries").join("garbage.bin"), b"junk").unwrap();
        let missing = sample_payload(30, 8);
        let mut index = ColdIndex::default();
        index.insert(missing.key, &missing.tokens, 8);
        std::fs::write(dir.join("index.json"), index.to_json()).unwrap();

        let tier = ColdTier::open(dir.clone(), 1024 * 1024).expect("open");
        assert!(tier.can_spill());
        assert!(!dir.join("entries").join("x.tmp").exists());
        assert!(!dir.join("entries").join(format!("{orphan_key}.bin")).exists());
        assert!(!dir.join("entries").join("garbage.bin").exists());
        assert!(tier.lookup_deepest(&missing.tokens, usize::MAX, None).is_none());
        drop(tier);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn second_open_is_restore_only() {
        let dir = tier_dir("second_open_is_restore_only");
        let mut first = ColdTier::open(dir.clone(), 1024 * 1024).expect("open");
        first.spill(sample_payload(40, 8), "Plain");
        let mut second = ColdTier::open(dir.clone(), 1024 * 1024).expect("open");
        assert!(!second.can_spill());
        let payload = sample_payload(41, 8);
        second.spill(payload.clone(), "Plain");
        assert!(!dir.join("entries").join(format!("{}.bin", payload.key.hex())).exists());
        drop(second);
        assert!(dir.join(".lock").exists());
        drop(first);
        assert!(!dir.join(".lock").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dead_pid_lock_is_taken_over() {
        let dir = tier_dir("dead_pid_lock_is_taken_over");
        let payload = sample_payload(50, 8);
        {
            let mut tier = ColdTier::open(dir.clone(), 1024 * 1024).expect("open");
            tier.spill(payload.clone(), "Plain");
        }
        assert!(!pid_alive(999999));
        std::fs::write(dir.join(".lock"), "999999").unwrap();
        let mut tier = ColdTier::open(dir.clone(), 1024 * 1024).expect("open");
        assert!(tier.can_spill());
        assert_eq!(std::fs::read_to_string(dir.join(".lock")).unwrap(), std::process::id().to_string());
        let (key, _, _) = tier.lookup_deepest(&payload.tokens, usize::MAX, None).expect("found");
        assert_eq!(tier.load(&key).expect("load"), payload);
        drop(tier);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn live_pid_lock_makes_restore_only() {
        let dir = tier_dir("live_pid_lock_makes_restore_only");
        let payload = sample_payload(60, 8);
        {
            let mut tier = ColdTier::open(dir.clone(), 1024 * 1024).expect("open");
            tier.spill(payload.clone(), "Plain");
        }
        std::fs::write(dir.join(".lock"), std::process::id().to_string()).unwrap();
        let mut tier = ColdTier::open(dir.clone(), 1024 * 1024).expect("open");
        assert!(!tier.can_spill());
        let (key, _, _) = tier.lookup_deepest(&payload.tokens, usize::MAX, None).expect("found");
        assert_eq!(tier.load(&key).expect("load"), payload);
        drop(tier);
        assert!(dir.join(".lock").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lookup_honours_bound_method_and_pending() {
        let mut index = ColdIndex::default();
        let shallow = entry_key(b"d", &tokens(&[1, 2]), b"", "Plain", 2);
        let deep = entry_key(b"d", &tokens(&[1, 2, 3, 4]), b"", "Mtp", 4);
        index.insert(shallow, &tokens(&[1, 2]), 1);
        index.set_method(&shallow, "Plain");
        index.insert(deep, &tokens(&[1, 2, 3, 4]), 1);
        index.set_method(&deep, "Mtp");
        let query = tokens(&[1, 2, 3, 4, 5]);
        // Deepest wins; a method mismatch falls back to the shallower entry.
        assert_eq!(index.lookup_deepest(&query, usize::MAX, Some("Mtp")).map(|r| r.1), Some(4));
        assert_eq!(index.lookup_deepest(&query, usize::MAX, Some("Plain")).map(|r| r.1), Some(2));
        // An entry at or past the bound is skipped, not returned.
        assert_eq!(index.lookup_deepest(&query, 4, Some("Mtp")), None);
        // A pending write is not offered for restore.
        index.mark_pending(deep);
        assert_eq!(index.lookup_deepest(&query, usize::MAX, None).map(|r| r.1), Some(2));
        index.clear_pending(&deep);
        assert_eq!(index.lookup_deepest(&query, usize::MAX, None).map(|r| r.1), Some(4));
    }

    #[test]
    fn disk_is_preferred_only_for_marginal_tokens() {
        // 100K deep on disk vs a 3.5K memory hit: 1.5 GB read is cheap.
        assert!(prefer_disk(3_518, 100_000, 1_500_000_000));
        // Not deeper: keep memory.
        assert!(!prefer_disk(100_000, 100_000, 1));
        assert!(!prefer_disk(100_000, 90_000, 1));
        // A huge read for a handful of extra tokens is not worth it.
        assert!(!prefer_disk(99_990, 100_000, 1_500_000_000));
    }

    #[test]
    fn load_of_a_pending_entry_keeps_it() {
        let dir = std::env::temp_dir().join(format!("cold-pending-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut tier = ColdTier::open(dir.clone(), 1 << 20).unwrap();
        let payload = sample_payload(70, 8);
        tier.spill(payload.clone(), "Plain");
        tier.index.mark_pending(payload.key);
        assert!(tier.load(&payload.key).is_err());
        assert!(tier.index.is_pending(&payload.key));
        tier.index.clear_pending(&payload.key);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Builds `state/<hex>/` dirs with `n` MB of entries, oldest first.
    fn gc_fixture(tag: &str, names: &[(char, u64)]) -> PathBuf {
        let state = std::env::temp_dir().join(format!("cold-gc-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&state);
        for (i, (c, mb)) in names.iter().enumerate() {
            let dir = state.join(c.to_string().repeat(64));
            std::fs::create_dir_all(dir.join("entries")).unwrap();
            std::fs::write(dir.join("entries").join("a.bin"), vec![0u8; (*mb as usize) << 20]).unwrap();
            let index = dir.join("index.json");
            std::fs::write(&index, "{}").unwrap();
            let when = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000 + i as u64);
            std::fs::File::options().write(true).open(&index).unwrap().set_modified(when).unwrap();
        }
        state
    }

    #[test]
    fn gc_removes_least_recently_used_dirs_but_never_current() {
        let state = gc_fixture("lru", &[('a', 1), ('b', 1), ('c', 1)]);
        let current = "c".repeat(64);
        let removed = collect_identity_dirs(&state, &current, 5 << 19).unwrap();
        assert_eq!(removed, vec![state.join("a".repeat(64))]);
        assert!(state.join("b".repeat(64)).exists() && state.join(&current).exists());
        let _ = std::fs::remove_dir_all(&state);
    }

    #[test]
    fn gc_keeps_a_dir_locked_by_a_live_process() {
        let state = gc_fixture("lock", &[('a', 1), ('b', 1), ('c', 1)]);
        std::fs::write(state.join("a".repeat(64)).join(".lock"), std::process::id().to_string()).unwrap();
        let removed = collect_identity_dirs(&state, &"c".repeat(64), 5 << 19).unwrap();
        assert_eq!(removed, vec![state.join("b".repeat(64))]);
        let _ = std::fs::remove_dir_all(&state);
    }

    #[test]
    fn gc_under_budget_removes_nothing_and_ignores_other_names() {
        let state = gc_fixture("under", &[('a', 1), ('b', 1)]);
        std::fs::create_dir_all(state.join("notes")).unwrap();
        assert!(collect_identity_dirs(&state, &"b".repeat(64), 1 << 30).unwrap().is_empty());
        // Over budget: the non-identity dir is still never touched.
        let removed = collect_identity_dirs(&state, &"b".repeat(64), 0).unwrap();
        assert_eq!(removed, vec![state.join("a".repeat(64))]);
        assert!(state.join("notes").exists());
        let _ = std::fs::remove_dir_all(&state);
    }
}
