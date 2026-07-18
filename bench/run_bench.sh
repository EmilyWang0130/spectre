#!/usr/bin/env bash
# Distributed under the MIT License.
# See LICENSE.txt for details.
#
# Build and run the SpECTRE (full Marquina + MarquinaCpm) and WHISKY Marquina
# flux-kernel benchmarks, verify correctness, and report median ns/interface
# of each. The primary comparison is MarquinaCpm vs WHISKY (WHISKY's FAST=TRUE
# path is CPM's algorithmic sibling); full Marquina is timed as a regression
# guard.
#
# Correctness gates (any failure aborts the run):
#   - full Marquina output must stay bit-exact vs bench/reference_output.dat
#     (rel 1e-12 / abs 1e-14).
#   - MarquinaCpm must stay within the equivalence envelope of full Marquina
#     (rel 1e-12 / abs 1e-12).
#
# Usage (from the repo root):  bash bench/run_bench.sh
# Environment: BUILD_DIR (default build), JOBS (default 8), RUNS (default 5).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${BUILD_DIR:-build}"
JOBS="${JOBS:-8}"
RUNS="${RUNS:-5}"
BENCH_DIR="${REPO_ROOT}/bench"
RESULTS_LOG="${BENCH_DIR}/RESULTS.log"
SPECTRE_BIN="${BUILD_DIR}/bin/bench_marquina"
CPM_BIN="${BUILD_DIR}/bin/bench_marquina_cpm"
WHISKY_BIN="${BUILD_DIR}/bin/bench_whisky"

log() { printf '%s\n' "$*"; }

# --- Build ---------------------------------------------------------------
log "==> Building benchmarks (JOBS=${JOBS})"
cmake --build "${BUILD_DIR}" \
  --target bench_marquina bench_marquina_cpm bench_whisky -j "${JOBS}"

# --- median of a list of numbers (discard min and max, take median) ------
median_of() {
  local sorted
  mapfile -t sorted < <(printf '%s\n' "$@" | sort -g)
  local n=${#sorted[@]}
  if (( n <= 2 )); then
    printf '%s\n' "${sorted[$((n/2))]}"
    return
  fi
  local trimmed=("${sorted[@]:1:$((n-2))}")
  local m=${#trimmed[@]}
  printf '%s\n' "${trimmed[$((m/2))]}"
}

# Run one benchmark binary RUNS times, verifying exit code, and print the
# median of the ns/interface value tagged by the given awk pattern to stdout.
# All progress goes to stderr so the captured stdout is only the median.
# Args: <label> <binary> <awk-tag> [binary-args...]
run_bench_binary() {
  local label="$1" bin="$2" tag="$3"
  shift 3
  local samples=() out ns
  printf '==> Running %s benchmark (%s runs)\n' "${label}" "${RUNS}" >&2
  for ((i = 1; i <= RUNS; ++i)); do
    out="$("${bin}" "$@")" || {
      printf 'ERROR: %s benchmark failed on run %s (verify or crash).\n' \
        "${label}" "${i}" >&2
      printf '%s\n' "${out}" >&2
      exit 1
    }
    ns="$(printf '%s\n' "${out}" | awk -v t="${tag}" '$1==t{print $2}')"
    printf '    run %s: %s ns/interface\n' "${i}" "${ns}" >&2
    samples+=("${ns}")
  done
  median_of "${samples[@]}"
}

spectre_median="$(run_bench_binary "full Marquina" "${SPECTRE_BIN}" \
  SPECTRE_NS_PER_CALL "${BENCH_DIR}")"
cpm_median="$(run_bench_binary "MarquinaCpm" "${CPM_BIN}" \
  CPM_NS_PER_CALL "${BENCH_DIR}")"
whisky_median="$(run_bench_binary "WHISKY" "${WHISKY_BIN}" WHISKY_NS_PER_CALL)"

# --- Report --------------------------------------------------------------
# Primary ratio: MarquinaCpm vs WHISKY (apples-to-apples FAST-path sibling).
ratio="$(awk -v s="${cpm_median}" -v w="${whisky_median}" \
  'BEGIN { if (w > 0) printf "%.3f", s / w; else print "inf" }')"

summary="CPM: ${cpm_median} ns/call  WHISKY: ${whisky_median} ns/call"
summary="${summary}  RATIO: ${ratio}  (full Marquina: ${spectre_median})"
log ""
log "==> ${summary}"

timestamp="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
gitrev="$(git rev-parse --short HEAD 2>/dev/null || echo 'nogit')"
printf '%s  %s  %s\n' "${timestamp}" "${gitrev}" "${summary}" \
  >> "${RESULTS_LOG}"
log "==> Appended to ${RESULTS_LOG}"
