#!/usr/bin/env python3
"""SSD cold-tier KV cache effectiveness report.

Parses ~/.magnitude-desktop-dev/logs/inference.log* cold-tier lines and
joins with serving-usage.sqlite to answer: is the SSD tier paying for itself?

Read-only. Stdlib only. --since "2026-11-04" filters; --watch N re-runs.
"""
import argparse
import glob
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone

HOME = os.path.expanduser("~/.magnitude-desktop-dev")
LOG_GLOB = os.path.join(HOME, "logs", "inference.log*")
SQLITE_PATH = os.path.join(HOME, "serving-usage.sqlite")
INDEX_GLOB = os.path.join(HOME, "cache", "state", "*", "index.json")

# Second timestamp in relay lines; event payload after the worker.pid= field.
TS_RE = re.compile(r"worker\.pid=\d+ (\S+)")
DECISION_RE = re.compile(
    r"cold-tier decision mem_depth=(\d+) disk_depth=(\d+) disk_bytes=(\d+) "
    r'resume_bound=(\d+) chosen="(\w+)"'
)
RESTORE_RE = re.compile(
    r"cold-tier restore restore_hit=(\w+) position=(\d+) bytes=(\d+) restore_ms=(\d+)"
)
RESTORE_FAIL_RE = re.compile(r"cold-tier restore failed")
SPILL_RE = re.compile(r"cold-tier spill spill_bytes=(\d+) spill_ms=(\d+)")
SPILL_SKIP_RE = re.compile(r"spill skipped.*skip_unsupported=true")
WRITE_RE = re.compile(r"cold-tier write bytes=(\d+) write_ms=(\d+)")


def parse_ts(line):
    m = TS_RE.search(line)
    if not m:
        return None
    try:
        return datetime.fromisoformat(m.group(1).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def median(xs):
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def report(since_ts):
    decisions, restores, spills, writes = [], [], [], []
    restore_fails = 0
    spill_skips = 0
    for path in sorted(glob.glob(LOG_GLOB)):
        with open(path, errors="replace") as f:
            for line in f:
                if "cold-tier" not in line:
                    continue
                ts = parse_ts(line)
                if ts is None or (since_ts and ts < since_ts):
                    continue
                if m := DECISION_RE.search(line):
                    decisions.append(
                        dict(ts=ts, mem_depth=int(m[1]), disk_depth=int(m[2]),
                             disk_bytes=int(m[3]), chosen=m[5]))
                elif m := RESTORE_RE.search(line):
                    restores.append(dict(ts=ts, hit=m[1] == "true",
                                         position=int(m[2]), bytes=int(m[3]),
                                         restore_ms=int(m[4])))
                elif RESTORE_FAIL_RE.search(line):
                    restore_fails += 1
                elif m := SPILL_RE.search(line):
                    spills.append(dict(ts=ts, bytes=int(m[1]), spill_ms=int(m[2])))
                elif SPILL_SKIP_RE.search(line):
                    spill_skips += 1
                elif m := WRITE_RE.search(line):
                    writes.append(dict(ts=ts, bytes=int(m[1]), write_ms=int(m[2])))

    total_events = len(decisions) + len(restores) + len(spills) + len(writes) + restore_fails + spill_skips
    print(f"Logs: {LOG_GLOB}  (since {datetime.fromtimestamp(since_ts, timezone.utc) if since_ts else 'all'})")
    print()

    if total_events == 0:
        print("VERDICT: no cold-tier events found. Tier disabled, not firing, or logs rotated.")
        return

    # --- Decisions ---
    by_choice = {}
    for d in decisions:
        by_choice[d["chosen"]] = by_choice.get(d["chosen"], 0) + 1
    print(f"Decisions ({len(decisions)}): " +
          ", ".join(f"{k}={v} ({100*v/len(decisions):.0f}%)" for k, v in sorted(by_choice.items())))
    disk_zero_mem = [d for d in decisions if d["chosen"] == "disk" and d["mem_depth"] == 0]
    n_disk = by_choice.get("disk", 0)
    if n_disk:
        print(f"  disk decisions with mem_depth=0 (cross-restart resume): {len(disk_zero_mem)}/{n_disk}")

    # --- Restores: achieved bandwidth ---
    hits = [r for r in restores if r["hit"]]
    print(f"Restores: {len(restores)} ({len(hits)} hit, {restore_fails} failed/fresh-open)")
    if hits:
        gbps = [r["bytes"] / max(r["restore_ms"], 1) * 1e3 / 1e9 for r in hits]
        print(f"  restore_ms median={median([r['restore_ms'] for r in hits]):.0f}  "
              f"achieved GB/s median={median(gbps):.2f} (model assumes 3.0)")

    # --- ms/token from fresh-cohort TTFT in sqlite ---
    ms_per_token = None
    if os.path.exists(SQLITE_PATH):
        db = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True)
        fresh_ts = [d["ts"] for d in decisions if d["chosen"] == "fresh"]
        ttfts = []
        if fresh_ts:
            rows = {r[0] / 1000.0: (r[1], r[2]) for r in db.execute(  # completed_at is epoch ms
                "SELECT completed_at, first_token_ms, input FROM usage "
                "WHERE first_token_ms IS NOT NULL AND input > 0")}
            rows_ts = sorted(rows)
            for ts in fresh_ts:
                # nearest usage row within 30s (decision precedes completion)
                best = None
                for rts in rows_ts:
                    if rts < ts - 10 or rts > ts + 120:
                        continue
                    if best is None or abs(rts - ts) < abs(best - ts):
                        best = rts
                if best:
                    ft_ms, inp = rows[best]
                    ttfts.append(ft_ms / inp)
        if ttfts:
            ms_per_token = median(ttfts)
            print(f"  ms/token from fresh-cohort TTFT (n={len(ttfts)}): {ms_per_token:.3f} (model assumes 1.5)")
        db.close()

    # --- Net win per restore ---
    if hits and ms_per_token:
        wins = [r["position"] * ms_per_token - r["restore_ms"] for r in hits]
        saved_ms = sum(w for w in wins if w > 0)
        wasted_ms = sum(-w for w in wins if w < 0)
        print(f"  net win per restore: median={median(wins):.0f} ms  "
              f"total saved={saved_ms/1000:.1f}s  total wasted={wasted_ms/1000:.1f}s  "
              f"losing restores={sum(1 for w in wins if w < 0)}/{len(wins)}")

    # --- Spills / writes ---
    spill_gb = sum(s["bytes"] for s in spills) / 1e9
    write_ms_total = sum(w["write_ms"] for w in writes)
    print(f"Spills: {len(spills)} ({spill_gb:.2f} GB), writes: {len(writes)} "
          f"(disk time total {write_ms_total/1000:.1f}s), spill skipped: {spill_skips}")

    # --- Reuse ratio ---
    if spills:
        print(f"Reuse: {len(hits)/len(spills):.2f} restores per spill "
              f"({len(hits)} restores / {len(spills)} spills)")
    idx_paths = glob.glob(INDEX_GLOB)
    if idx_paths:
        try:
            with open(idx_paths[0]) as f:
                idx = json.load(f)
            entries = idx.get("entries", idx if isinstance(idx, list) else [])
            print(f"ColdIndex ({os.path.basename(os.path.dirname(idx_paths[0]))}): {len(entries)} entries on disk")
        except (json.JSONDecodeError, OSError, TypeError) as e:
            print(f"ColdIndex: unreadable ({e})")

    # --- Verdict ---
    print()
    if not hits and not spills:
        verdict = "DISUSED: decisions only, no data ever moved."
    elif hits and ms_per_token and sum(w for w in (r["position"] * ms_per_token - r["restore_ms"] for r in hits) if w > 0) > write_ms_total:
        verdict = "HELPING: restore savings exceed write cost."
    elif hits:
        verdict = "MARGINAL: restores happen but net benefit unclear (check medians above)."
    else:
        verdict = "DISUSED: spills happen, nothing is ever restored."
    print(f"VERDICT: {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", help="ISO date/datetime, e.g. 2026-11-04")
    ap.add_argument("--watch", type=int, metavar="SECS", help="re-run every N seconds")
    args = ap.parse_args()
    since_ts = None
    if args.since:
        s = args.since
        dt = datetime.fromisoformat(s) if "T" in s else datetime.fromisoformat(s + "T00:00:00")
        since_ts = dt.replace(tzinfo=dt.tzinfo or timezone.utc).timestamp()
    while True:
        report(since_ts)
        if not args.watch:
            break
        time.sleep(args.watch)
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
