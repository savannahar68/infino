// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Drain profiler: where one hidden-cell drain spends its time.
//!
//! Gated on `diagnostics.drain_profile`. Off, every entry point is a no-op
//! behind one boolean check.
//!
//! Two kinds of measurement:
//!
//! - **Steps** — the drain's sequential phases (load the manifest, open a
//!   batch, assign, upload, commit, settle ...). Each records its wall time,
//!   the process CPU it burned (so `cores = cpu / wall` separates CPU-bound
//!   from waiting), RSS, and the object-store and disk-cache traffic it
//!   caused. With `diagnostics.io_timeline` on, each step also carries a
//!   per-request latency distribution and the implied request concurrency.
//!   A step ends when the next one begins, so the steps tile the drain.
//! - **Sub-steps** — work inside a parallel fan-out or a deep helper (one
//!   superfile open, one cell build, one fsync). Recorded as count / total /
//!   max per name, under the step that was current when they ran. Their
//!   totals are summed across threads, so they can exceed the step's wall.
//!
//! A background thread samples RSS, process CPU, drain-scratch bytes and
//! disk-cache bytes once a second, tagged with the current step.
//!
//! The report is one JSON document per drain under
//! `<scratch_root>/infino-drain-profiles/`, plus a summary table on stderr.

use std::{
    collections::BTreeMap,
    fs,
    future::Future,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::Serialize;
use serde_json::{Map, Value};

use crate::{
    config::{self, scratch_root},
    runtime_metrics::{
        cpu::process_cpu_ns,
        io::{UsageMeter, UsageSnapshot},
        rss::current_rss_bytes,
    },
    storage::io_counters::{self, FetchSpan},
    supertable::reader_cache::{CacheStats, DiskCacheStore},
};

/// How often the background sampler records a resource sample.
const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
/// Directory under the scratch root the reports land in.
const REPORT_DIR: &str = "infino-drain-profiles";
/// Bytes in one MiB, for the human-readable summary.
const MIB: f64 = 1024.0 * 1024.0;
/// Nanoseconds in one second.
const NS_PER_SEC: f64 = 1e9;
/// Microseconds in one millisecond.
const US_PER_MS: f64 = 1e3;
/// Percentiles reported for each request kind's latency.
const LATENCY_PERCENTILES: [(&str, f64); 3] = [("p50", 0.50), ("p90", 0.90), ("p99", 0.99)];

/// Whether drain profiling is on (`diagnostics.drain_profile`).
pub(crate) fn enabled() -> bool {
    config::global().diagnostics.drain_profile
}

/// The profile of the drain in flight, if any. Sub-step recorders deep in
/// helpers reach it here rather than through every signature. A drain is
/// single-flight per table, and profiling runs one table, so one slot is
/// enough.
static ACTIVE: Mutex<Option<Arc<Shared>>> = Mutex::new(None);

fn active() -> Option<Arc<Shared>> {
    ACTIVE.lock().ok().and_then(|slot| slot.clone())
}

/// State the drain thread, the fan-out workers and the sampler share.
struct Shared {
    t0: Instant,
    /// Name of the step currently running.
    current: Mutex<String>,
    /// `(step, sub-step) -> aggregate`.
    subs: Mutex<BTreeMap<(String, &'static str), SubAgg>>,
    samples: Mutex<Vec<Sample>>,
    scratch: Mutex<Option<PathBuf>>,
    cache: Option<Arc<DiskCacheStore>>,
    stop: AtomicBool,
}

#[derive(Debug, Default, Clone, Serialize)]
struct SubAgg {
    count: u64,
    total_ms: f64,
    max_ms: f64,
    bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct Sample {
    t_ms: f64,
    step: String,
    rss_mib: Option<f64>,
    cpu_s: Option<f64>,
    scratch_mib: f64,
    cache_mib: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct CacheDelta {
    cold_fetches: u64,
    disk_reuses: u64,
    evictions: u64,
    entries_end: u64,
    bytes_end_mib: f64,
}

#[derive(Debug, Clone, Serialize)]
struct LatencyStats {
    count: usize,
    mib: f64,
    sum_ms: f64,
    p50_ms: f64,
    p90_ms: f64,
    p99_ms: f64,
    max_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct TimelineStats {
    /// First request start to last request end.
    io_wall_ms: f64,
    /// `Σ request durations / io_wall`: ~1 is a serial chain of requests.
    implied_concurrency: f64,
    by_op: BTreeMap<&'static str, LatencyStats>,
}

#[derive(Debug, Clone, Serialize)]
struct StepRecord {
    name: String,
    start_ms: f64,
    wall_ms: f64,
    cpu_s: Option<f64>,
    /// Average cores busy over the step (`cpu_s / wall_s`).
    cores: Option<f64>,
    rss_mib_start: Option<f64>,
    rss_mib_end: Option<f64>,
    io: UsageSnapshot,
    cache: Option<CacheDelta>,
    timeline: Option<TimelineStats>,
    notes: Map<String, Value>,
}

/// An open step: the counters it started from.
struct OpenStep {
    name: String,
    t: Instant,
    cpu_ns: Option<u128>,
    rss: Option<u64>,
    io: Option<UsageSnapshot>,
    cache: Option<CacheStats>,
    notes: Map<String, Value>,
}

struct Active {
    shared: Arc<Shared>,
    meter: Option<Arc<UsageMeter>>,
    open: Option<OpenStep>,
    steps: Vec<StepRecord>,
    notes: Map<String, Value>,
    sampler: Option<JoinHandle<()>>,
    cpu0: Option<u128>,
}

/// One drain's profile. A no-op shell when profiling is off.
pub(crate) struct DrainProfile {
    active: Option<Active>,
}

impl DrainProfile {
    /// Begin profiling a drain. `meter` is the storage provider's ledger (the
    /// user and hidden tables share it); `cache` is the hidden table's disk
    /// cache, if one is attached.
    pub(crate) fn start(
        meter: Option<Arc<UsageMeter>>,
        cache: Option<Arc<DiskCacheStore>>,
    ) -> Self {
        if !enabled() {
            return Self { active: None };
        }
        let shared = Arc::new(Shared {
            t0: Instant::now(),
            current: Mutex::new("start".into()),
            subs: Mutex::new(BTreeMap::new()),
            samples: Mutex::new(Vec::new()),
            scratch: Mutex::new(None),
            cache,
            stop: AtomicBool::new(false),
        });
        if let Ok(mut slot) = ACTIVE.lock() {
            *slot = Some(Arc::clone(&shared));
        }
        let sampler_shared = Arc::clone(&shared);
        let sampler = thread::Builder::new()
            .name("drain-profile-sampler".into())
            .spawn(move || run_sampler(&sampler_shared))
            .ok();
        Self {
            active: Some(Active {
                shared,
                meter,
                open: None,
                steps: Vec::new(),
                notes: Map::new(),
                sampler,
                cpu0: process_cpu_ns(),
            }),
        }
    }

    /// Close the running step (if any) and open `name`.
    pub(crate) fn step(&mut self, name: impl Into<String>) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        active.close_open();
        let name = name.into();
        if let Ok(mut current) = active.shared.current.lock() {
            current.clone_from(&name);
        }
        if io_counters::timeline_enabled() {
            io_counters::timeline_reset();
        }
        active.open = Some(OpenStep {
            name,
            t: Instant::now(),
            cpu_ns: process_cpu_ns(),
            rss: current_rss_bytes(),
            io: active.meter.as_ref().map(|m| m.snapshot()),
            cache: active.shared.cache.as_ref().map(|c| c.stats()),
            notes: Map::new(),
        });
    }

    /// Attach a fact (row count, superfile count, ...) to the running step.
    pub(crate) fn note(&mut self, key: &str, value: impl Into<Value>) {
        if let Some(open) = self.active.as_mut().and_then(|a| a.open.as_mut()) {
            open.notes.insert(key.into(), value.into());
        }
    }

    /// Attach a fact to the drain as a whole.
    pub(crate) fn note_drain(&mut self, key: &str, value: impl Into<Value>) {
        if let Some(active) = self.active.as_mut() {
            active.notes.insert(key.into(), value.into());
        }
    }

    /// Tell the sampler where the drain's local scratch lives.
    pub(crate) fn set_scratch(&mut self, dir: &Path) {
        if let Some(active) = self.active.as_ref()
            && let Ok(mut scratch) = active.shared.scratch.lock()
        {
            *scratch = Some(dir.to_path_buf());
        }
    }

    /// Close the last step and write the report. `outcome` says how the
    /// drain ended (`done`, `nothing_to_drain`, ...).
    pub(crate) fn finish(mut self, outcome: &str) {
        self.finish_inner(outcome);
    }

    fn finish_inner(&mut self, outcome: &str) {
        let Some(mut active) = self.active.take() else {
            return;
        };
        active.close_open();
        active.shared.stop.store(true, Ordering::Relaxed);
        if let Some(sampler) = active.sampler.take() {
            let _ = sampler.join();
        }
        if let Ok(mut slot) = ACTIVE.lock()
            && slot
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, &active.shared))
        {
            *slot = None;
        }
        write_report(&active, outcome);
    }
}

impl Drop for DrainProfile {
    /// A drain that returns early or fails through `?` still reports what it
    /// measured, marked with how it ended.
    fn drop(&mut self) {
        self.finish_inner("returned_early_or_failed");
    }
}

impl Active {
    fn close_open(&mut self) {
        let Some(open) = self.open.take() else {
            return;
        };
        let wall = open.t.elapsed();
        let cpu_s = match (open.cpu_ns, process_cpu_ns()) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start) as f64 / NS_PER_SEC),
            _ => None,
        };
        let wall_s = wall.as_secs_f64();
        let io = match (&self.meter, &open.io) {
            (Some(meter), Some(before)) => meter.snapshot().since(before),
            _ => UsageSnapshot::default(),
        };
        let cache = match (&self.shared.cache, &open.cache) {
            (Some(cache), Some(before)) => {
                let now = cache.stats();
                Some(CacheDelta {
                    cold_fetches: now.n_cold_fetches.saturating_sub(before.n_cold_fetches),
                    disk_reuses: now.n_disk_reuses.saturating_sub(before.n_disk_reuses),
                    evictions: now.n_evictions.saturating_sub(before.n_evictions),
                    entries_end: now.n_entries,
                    bytes_end_mib: now.current_bytes as f64 / MIB,
                })
            }
            _ => None,
        };
        let timeline =
            io_counters::timeline_enabled().then(|| timeline_stats(io_counters::timeline_take()));
        self.steps.push(StepRecord {
            name: open.name,
            start_ms: open
                .t
                .saturating_duration_since(self.shared.t0)
                .as_secs_f64()
                * US_PER_MS,
            wall_ms: wall_s * US_PER_MS,
            cpu_s,
            cores: cpu_s.filter(|_| wall_s > 0.0).map(|cpu| cpu / wall_s),
            rss_mib_start: open.rss.map(|b| b as f64 / MIB),
            rss_mib_end: current_rss_bytes().map(|b| b as f64 / MIB),
            io,
            cache,
            timeline,
            notes: open.notes,
        });
    }
}

/// Time `f` as sub-step `name` of the running step. A plain call when no
/// drain is being profiled.
pub(crate) fn timed<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    if active().is_none() {
        return f();
    }
    let t = Instant::now();
    let out = f();
    record(name, t.elapsed(), 0);
    out
}

/// Async counterpart of [`timed`].
pub(crate) async fn timed_async<T>(name: &'static str, fut: impl Future<Output = T>) -> T {
    if active().is_none() {
        return fut.await;
    }
    let t = Instant::now();
    let out = fut.await;
    record(name, t.elapsed(), 0);
    out
}

/// Record one occurrence of sub-step `name` that took `dur` and moved
/// `bytes` (0 when bytes do not apply).
pub(crate) fn record(name: &'static str, dur: Duration, bytes: u64) {
    let Some(shared) = active() else {
        return;
    };
    let step = shared
        .current
        .lock()
        .map(|current| current.clone())
        .unwrap_or_default();
    let ms = dur.as_secs_f64() * US_PER_MS;
    if let Ok(mut subs) = shared.subs.lock() {
        let agg = subs.entry((step, name)).or_default();
        agg.count += 1;
        agg.total_ms += ms;
        agg.max_ms = agg.max_ms.max(ms);
        agg.bytes += bytes;
    }
}

/// Record that sub-step `name` happened, with no duration of its own (a
/// cache hit, a skipped branch).
pub(crate) fn count(name: &'static str) {
    record(name, Duration::ZERO, 0);
}

fn run_sampler(shared: &Shared) {
    while !shared.stop.load(Ordering::Relaxed) {
        let scratch_bytes = shared
            .scratch
            .lock()
            .ok()
            .and_then(|dir| dir.clone())
            .map(|dir| dir_bytes(&dir))
            .unwrap_or(0);
        let sample = Sample {
            t_ms: shared.t0.elapsed().as_secs_f64() * US_PER_MS,
            step: shared
                .current
                .lock()
                .map(|current| current.clone())
                .unwrap_or_default(),
            rss_mib: current_rss_bytes().map(|b| b as f64 / MIB),
            cpu_s: process_cpu_ns().map(|ns| ns as f64 / NS_PER_SEC),
            scratch_mib: scratch_bytes as f64 / MIB,
            cache_mib: shared
                .cache
                .as_ref()
                .map(|cache| cache.stats().current_bytes as f64 / MIB),
        };
        if let Ok(mut samples) = shared.samples.lock() {
            samples.push(sample);
        }
        thread::sleep(SAMPLE_INTERVAL);
    }
}

/// Bytes under `dir`, recursively. Missing entries count as zero: the drain
/// creates and removes scratch files while this walks.
fn dir_bytes(dir: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| match entry.file_type() {
            Ok(kind) if kind.is_dir() => dir_bytes(&entry.path()),
            Ok(_) => entry.metadata().map(|meta| meta.len()).unwrap_or(0),
            Err(_) => 0,
        })
        .sum()
}

fn timeline_stats(spans: Vec<FetchSpan>) -> TimelineStats {
    let min_start = spans.iter().map(|s| s.start_us).min().unwrap_or(0);
    let max_end = spans.iter().map(|s| s.end_us).max().unwrap_or(0);
    let wall_us = max_end.saturating_sub(min_start);
    let mut by_op: BTreeMap<&'static str, (Vec<f64>, u64)> = BTreeMap::new();
    for span in &spans {
        let (durations, bytes) = by_op.entry(span.op).or_default();
        durations.push(span.end_us.saturating_sub(span.start_us) as f64 / US_PER_MS);
        *bytes += span.len;
    }
    let sum_ms_all: f64 = by_op.values().flat_map(|(d, _)| d.iter()).sum();
    TimelineStats {
        io_wall_ms: wall_us as f64 / US_PER_MS,
        implied_concurrency: if wall_us > 0 {
            sum_ms_all / (wall_us as f64 / US_PER_MS)
        } else {
            0.0
        },
        by_op: by_op
            .into_iter()
            .map(|(op, (mut durations, bytes))| {
                durations.sort_by(f64::total_cmp);
                let pct = |q: f64| percentile(&durations, q);
                let [(_, p50), (_, p90), (_, p99)] = LATENCY_PERCENTILES;
                (
                    op,
                    LatencyStats {
                        count: durations.len(),
                        mib: bytes as f64 / MIB,
                        sum_ms: durations.iter().sum(),
                        p50_ms: pct(p50),
                        p90_ms: pct(p90),
                        p99_ms: pct(p99),
                        max_ms: durations.last().copied().unwrap_or(0.0),
                    },
                )
            })
            .collect(),
    }
}

/// Nearest-rank percentile of an ascending slice.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

#[derive(Serialize)]
struct Report<'a> {
    outcome: &'a str,
    unix_ms: u128,
    total_wall_ms: f64,
    total_cpu_s: Option<f64>,
    notes: &'a Map<String, Value>,
    steps: &'a [StepRecord],
    /// `step -> sub-step -> aggregate`.
    sub_steps: BTreeMap<String, BTreeMap<&'static str, SubAgg>>,
    samples: Vec<Sample>,
}

fn write_report(active: &Active, outcome: &str) {
    let mut sub_steps: BTreeMap<String, BTreeMap<&'static str, SubAgg>> = BTreeMap::new();
    if let Ok(subs) = active.shared.subs.lock() {
        for ((step, name), agg) in subs.iter() {
            sub_steps
                .entry(step.clone())
                .or_default()
                .insert(name, agg.clone());
        }
    }
    let unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let report = Report {
        outcome,
        unix_ms,
        total_wall_ms: active.shared.t0.elapsed().as_secs_f64() * US_PER_MS,
        total_cpu_s: match (active.cpu0, process_cpu_ns()) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start) as f64 / NS_PER_SEC),
            _ => None,
        },
        notes: &active.notes,
        steps: &active.steps,
        sub_steps,
        samples: active
            .shared
            .samples
            .lock()
            .map(|samples| samples.clone())
            .unwrap_or_default(),
    };
    let dir = scratch_root().join(REPORT_DIR);
    let path = dir.join(format!("drain-{unix_ms}.json"));
    let written = fs::create_dir_all(&dir)
        .map_err(|e| e.to_string())
        .and_then(|()| serde_json::to_vec_pretty(&report).map_err(|e| e.to_string()))
        .and_then(|bytes| fs::write(&path, bytes).map_err(|e| e.to_string()));
    eprintln!("{}", summary(&report));
    match written {
        Ok(()) => eprintln!("[drainprof] report: {}", path.display()),
        Err(error) => eprintln!("[drainprof] report write failed: {error}"),
    }
}

/// One summary row: every instance of a step name (a batch step runs once
/// per batch) folded together.
#[derive(Default)]
struct StepGroup {
    runs: usize,
    wall_ms: f64,
    cpu_s: Option<f64>,
    gets: u64,
    get_bytes: u64,
    puts: u64,
    put_bytes: u64,
    /// Request-weighted p50 across the instances' GET distributions.
    get_p50_weighted: f64,
    get_requests: usize,
    /// Worst instance p99 — a fold of percentiles is not a percentile, so
    /// the table reports the bound and the JSON keeps each instance.
    get_p99_max: Option<f64>,
    cold_fetches: Option<u64>,
}

/// Fixed-width table: one row per step name in first-run order, then the
/// sub-steps by total time.
fn summary(report: &Report<'_>) -> String {
    let mut out = format!(
        "[drainprof] outcome={} wall={:.1}s cpu={}\n",
        report.outcome,
        report.total_wall_ms / US_PER_MS,
        report
            .total_cpu_s
            .map(|s| format!("{s:.1}s"))
            .unwrap_or_else(|| "?".into()),
    );
    let mut order: Vec<&str> = Vec::new();
    let mut groups: BTreeMap<&str, StepGroup> = BTreeMap::new();
    for step in report.steps {
        let group = groups.entry(step.name.as_str()).or_insert_with(|| {
            order.push(step.name.as_str());
            StepGroup::default()
        });
        group.runs += 1;
        group.wall_ms += step.wall_ms;
        if let Some(cpu) = step.cpu_s {
            *group.cpu_s.get_or_insert(0.0) += cpu;
        }
        group.gets += step.io.get_count + step.io.bg_get_count;
        group.get_bytes += step.io.get_bytes + step.io.bg_get_bytes;
        group.puts += step.io.put_count;
        group.put_bytes += step.io.put_bytes;
        if let Some(timeline) = &step.timeline {
            for (op, stats) in &timeline.by_op {
                if op.starts_with("get") || *op == "tail" {
                    group.get_p50_weighted += stats.p50_ms * stats.count as f64;
                    group.get_requests += stats.count;
                    let p99 = group.get_p99_max.get_or_insert(0.0);
                    *p99 = p99.max(stats.p99_ms);
                }
            }
        }
        if let Some(cache) = &step.cache {
            *group.cold_fetches.get_or_insert(0) += cache.cold_fetches;
        }
    }
    out.push_str(&format!(
        "[drainprof] {:<48} {:>4} {:>11} {:>6} {:>6} {:>8} {:>10} {:>7} {:>10} {:>8} {:>8} {:>7}\n",
        "step",
        "runs",
        "wall_ms",
        "%",
        "cores",
        "gets",
        "get_mib",
        "puts",
        "put_mib",
        "get_p50",
        "p99max",
        "cold",
    ));
    let total = report.total_wall_ms.max(f64::MIN_POSITIVE);
    for name in order {
        let g = &groups[name];
        let wall_s = g.wall_ms / US_PER_MS;
        out.push_str(&format!(
            "[drainprof] {:<48} {:>4} {:>11.1} {:>6.1} {:>6} {:>8} {:>10.1} {:>7} {:>10.1} {:>8} {:>8} {:>7}\n",
            name,
            g.runs,
            g.wall_ms,
            100.0 * g.wall_ms / total,
            g.cpu_s
                .filter(|_| wall_s > 0.0)
                .map(|cpu| format!("{:.1}", cpu / wall_s))
                .unwrap_or_else(|| "?".into()),
            g.gets,
            g.get_bytes as f64 / MIB,
            g.puts,
            g.put_bytes as f64 / MIB,
            if g.get_requests > 0 {
                format!("{:.1}", g.get_p50_weighted / g.get_requests as f64)
            } else {
                "-".into()
            },
            g.get_p99_max
                .map(|p| format!("{p:.1}"))
                .unwrap_or_else(|| "-".into()),
            g.cold_fetches
                .map(|c| c.to_string())
                .unwrap_or_else(|| "-".into()),
        ));
    }
    let mut subs: Vec<(&str, &str, &SubAgg)> = report
        .sub_steps
        .iter()
        .flat_map(|(step, names)| {
            names
                .iter()
                .map(move |(name, agg)| (step.as_str(), *name, agg))
        })
        .collect();
    subs.sort_by(|a, b| b.2.total_ms.total_cmp(&a.2.total_ms));
    out.push_str(&format!(
        "[drainprof] {:<48} {:<36} {:>9} {:>12} {:>10} {:>10}\n",
        "sub-step of", "name", "count", "total_ms", "max_ms", "mib",
    ));
    for (step, name, agg) in subs {
        out.push_str(&format!(
            "[drainprof] {:<48} {:<36} {:>9} {:>12.1} {:>10.1} {:>10.1}\n",
            step,
            name,
            agg.count,
            agg.total_ms,
            agg.max_ms,
            agg.bytes as f64 / MIB,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_is_nearest_rank() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&sorted, 0.50), 5.0);
        assert_eq!(percentile(&sorted, 0.90), 9.0);
        assert_eq!(percentile(&sorted, 0.99), 10.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn timeline_stats_reports_concurrency_and_bytes() {
        let span = |op, start_us, end_us, len| FetchSpan {
            op,
            uri: String::new(),
            off: 0,
            len,
            start_us,
            end_us,
            background: false,
        };
        // Two fully overlapped 10 ms GETs: wall 10 ms, Σ 20 ms → 2x.
        let stats = timeline_stats(vec![
            span("get", 0, 10_000, 1 << 20),
            span("get", 0, 10_000, 1 << 20),
        ]);
        assert_eq!(stats.io_wall_ms, 10.0);
        assert_eq!(stats.implied_concurrency, 2.0);
        let get = &stats.by_op["get"];
        assert_eq!(get.count, 2);
        assert_eq!(get.mib, 2.0);
        assert_eq!(get.p50_ms, 10.0);
    }
}
