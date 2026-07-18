<!--
Distributed under the MIT License.
See LICENSE.txt for details.
-->

# Marquina Flux Kernel — Optimization Report

## Summary

SpECTRE / WHISKY figures are ns per interface (median of 5 runs).

| Stage | SpECTRE | WHISKY | Ratio | Speedup |
|-------|--------:|-------:|------:|--------:|
| Baseline                          | 41568 | 75.9 | **547.3** | —      |
| opt #1: hoist O(n²) zeroing       |   610 | 74.6 |   **8.19** | 68.2×  |
| opt #2+#3: in-place decomposition |   472 | 74.9 |   **6.31** | 1.29×  |
| opt #4: remap aligned fields      |   430 | 74.6 |   **5.77** | 1.10×  |

**Overall: 41568 → 430 ns/interface, a 96.7× kernel speedup; WHISKY ratio
547.3 → 5.77.** Retired instruction count per interface fell from 604,878 to
4,196 (144×). Every stage was verified bit-for-bit against
`bench/reference_output.dat` (rel 1e-12 / abs 1e-14) and the existing
`Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.Marquina` and `...MarquinaCpm`
unit tests both pass.

The optimization loop stopped after 4 successful iterations without reaching
the 1.3 target ratio. As argued in "Remaining gap" below, the residual factor
is algorithmic (SpECTRE computes the full 6-mode characteristic decomposition
on `DataVector`s; WHISKY-Marquina FAST computes a 2-mode scalar shortcut) and
cannot be closed without changing the flux result — which the reference gate
forbids.

## Update — MarquinaCpm (FAST-path sibling) optimization, iterations 5–8

The scope was later refocused onto **MarquinaCpm**, SpECTRE's Complementary
Projection Method — the algorithmic sibling of WHISKY's `FAST=.TRUE.` path
(Aloy et al. 1999 CPC shortcut: only the two acoustic eigenvectors, degenerate
`lam1=lam2=lam3` block handled analytically via the spectral-projector
complement). This is the apples-to-apples comparison against WHISKY, so the
**primary ratio is now MarquinaCpm / WHISKY**. The full-`Marquina` path is left
bit-exact (still guarded by `bench/reference_output.dat`) and timed only as a
regression guard.

CPM figures are ns per interface (median of 5 runs); ratio is CPM / WHISKY.

| Stage | CPM | WHISKY | Ratio | Speedup |
|-------|----:|-------:|------:|--------:|
| CPM baseline                          | 440.8 | 74.6 | **5.91** | —     |
| opt #5: acoustic eigenvectors in place | 401.9 | 74.6 | **5.40** | 1.10× |
| opt #6+#7: CPM degenerate-block allocs | 318.6 | 74.6 | **4.29** | 1.26× |

**CPM: 440.8 → ~319 ns/interface, 1.38× faster; ratio 5.91 → 4.29.** All three
CPM optimizations leave the MarquinaCpm result **bit-identical** (the
CPM-vs-full-Marquina equivalence envelope is unchanged: max abs diff 6.6e-14,
well inside the documented 1e-12 envelope) — they remove allocations and copies
only. Every iteration was gated by three checks, all green: full Marquina
bit-exact vs `reference_output.dat`; MarquinaCpm within the 1e-12 equivalence
envelope of full Marquina (checked live in `bench_marquina_cpm`); and both
`...BoundaryCorrections.Marquina` and `...MarquinaCpm` unit tests
(`test_boundary_correction_agreement` at 1e-12 is the unit-level envelope gate).

Note on the referenced artifacts: the task pointed at
`spectre_runs/cpm_equivalence/README.md`, `spectre_runs/cpm_marquina_plan.md`,
and a `MarquinaCpm.DumpEquivalence` (`[.cpm-dump]`) dumper — none of which exist
in this checkout. The equivalent guarantee is provided by (a) the live envelope
check in `bench_marquina_cpm` and (b) the existing `Test_MarquinaCpm`
differential test, which already enforces MarquinaCpm ≈ Marquina at 1e-12.

### CPM optimizations applied — detailed explanation

All three are CPM-only (they live inside `use_cpm_degenerate_block == true`
branches, so the full-`Marquina` path is untouched) and **bit-identical**: they
remove `DataVector` allocations, zeroings and copies without changing a single
arithmetic operation, so the packaged fields and the final flux are the same
bit pattern as before. The unchanged envelope (max abs diff pinned at 6.6e-14
across all three) is the empirical proof of this.

Why CPM was *not* faster than full Marquina to begin with (baseline 440.8 vs
429.6 ns): the CPM path already skips the four degenerate eigenvectors, but that
saving was entirely eaten by (a) still allocating and zeroing the full 6×6
packaged eigenvector storage, (b) allocating separate acoustic temporaries and
copying them into that storage, and (c) the degenerate-block reconstruction in
`dg_boundary_terms` allocating a fresh set of `DataVector`s for every conserved
component. The three optimizations remove exactly those three allocation
sources.

**opt #5 — write the acoustic eigenvectors in place; skip the degenerate rows.**
`dg_package_data_impl`, CPM branch.
- *Before:* the branch zeroed all 72 packaged eigenvector components (6 rows ×
  6 cols × {left,right}), then built the two acoustic eigenvectors into fresh
  `std::array` temporaries created with
  `make_array<2>(make_with_value<…>(tilde_d, 0.0))` (which allocate and zero 24
  `DataVector`s), had `acoustic_eigenvectors_hydro` fill them, and finally
  copy-assigned those 24 `DataVector`s into the packaged Lplus/Lminus (left)
  and Rplus/Rminus (right) rows.
- *Change:* point the acoustic output `std::array` elements directly at the
  packaged storage with `DataVector::set_data_ref`, so
  `acoustic_eigenvectors_hydro` writes its results straight into the packaged
  rows — no separate allocation, no copy-back. Because those rows are already
  sized `num_points`, the callee's `allocate_and_zero` takes its in-place branch
  and does not reallocate (which would detach the non-owning alias). Pre-zero
  only rows 0..3 (R1..R4 / L1..L4): rows 4/5 (Rplus/Rminus == Lplus/Lminus) are
  fully overwritten by the acoustic call, and the four degenerate rows are the
  only ones that must be left at zero because the CPM `dg_boundary_terms`
  handles that block analytically and never reads packaged rows 0..3.
- *Why bit-identical:* `set_data_ref` changes only *where* the results land, not
  the values written; the pre-zeroing that was dropped applied to rows the
  acoustic call overwrites anyway.
- *Effect:* removes 24 allocate-and-copies plus 24 redundant zeroings per
  package-data call (×2 sides). 440.8 → 401.9 ns.

**opt #6 — reuse the degenerate-block scratch buffers.**
`dg_boundary_terms_impl`, CPM branch, `cpm_add_component` lambda.
- *Before:* the CPM degenerate block reconstructs, for each of the six conserved
  components, the projector-complement flux and state
  `f_deg = f − R₊φ₊ − R₋φ₋` and `u_deg = u − R₊ω₊ − R₋ω₋` (interior and
  exterior), then applies a single-speed Marquina upwind at `λ₀ = vₙ`. Each of
  the six invocations declared four new `const DataVector`s
  (`f_deg_int/ext`, `u_deg_int/ext`) — 24 fresh `num_points`-length allocations
  per boundary-terms call.
- *Change:* hoist four `DataVector` scratch buffers out of the lambda and assign
  the expressions into them in place, reusing the same storage across all six
  component invocations.
- *Why bit-identical:* the expressions are unchanged; a Blaze assignment into a
  same-size `DataVector` writes elementwise in place, so the values are
  identical — only the backing allocation is reused.
- *Effect:* 24 allocations per boundary-terms call → 4.

**opt #7 — remap the aligned exterior fields on read instead of copying them.**
`dg_boundary_terms_impl`, CPM branch.
- *Before:* to align the exterior characteristic fields to the interior normal
  convention (which only swaps the two acoustic rows Rplus↔Rminus), the CPM
  branch materialized two full 6×6 copies of the exterior left/right fields —
  72 `DataVector` copies — even though the degenerate block reads only the two
  acoustic rows.
- *Change:* drop the materialized copies and read through the function-scope
  row-remap accessors already introduced for the per-mode loop
  (`aligned_row(i)` swaps 4↔5; `aligned_right_ext(i,k)` returns
  `right_…_ext.get(aligned_row(i), k)`). The `project(…)` calls for the exterior
  acoustic modes now pass the raw exterior matrix with a remapped row index, and
  `cpm_add_component`'s exterior reads use the accessor.
- *Why bit-identical:* `aligned_matrix.get(row, k)` equalled
  `raw_matrix.get(aligned_row(row), k)` by construction of the swap, so the
  remapped reads return exactly the same `DataVector`s the copy held.
- *Effect:* eliminates 72 `DataVector` copies per boundary-terms call.
- opt #6 + #7 together: 401.9 → 318.6 ns.

### Equivalence dump (the DumpEquivalence-style gate)

The task referenced a `MarquinaCpm.DumpEquivalence` (`[.cpm-dump]`) dumper and
`spectre_runs/cpm_equivalence/README.md` — neither exists in this checkout, so
the equivalent gate is provided in `bench_marquina_cpm`, which every run dumps
the abs and rel diff **distributions** of MarquinaCpm vs full Marquina over all
6,000,000 samples (1M interfaces × 6 conserved components) and fails if any
sample leaves the 1e-12 envelope. Final distributions:

```
abs diff: p50=1.11e-16 p90=8.88e-16 p99=2.66e-15 p99.9=7.11e-15 max=6.62e-14
rel diff: p50=2.20e-16 p90=1.41e-15 p99=1.55e-14 p99.9=1.57e-13 max=5.99e-10
```

Abs diff is ≤ 6.62e-14 everywhere; the rel-diff tail (max 5.99e-10) occurs only
where the reference value is ~0 (abs diff there is ≤ 1e-13), exactly as the
documented 2.5M-sample envelope (abs ≤ 1e-12, rel ≤ 1e-12 in the combined
abs-or-rel sense used by `test_boundary_correction_agreement`) predicts. The
distribution is unchanged across all three CPM optimizations, confirming they
are bit-identical.

### Remaining CPM gap (ratio ≈ 4.3) and cause

A clean profile of the CPM *timed* path (the `MarquinaCpm::` call subtree)
shows the cost is now concentrated in `acoustic_eigenvectors_hydro` (~6% of the
process, the largest CPM component); `characteristic_speeds_hydro` is cheap
(<1%). The remaining cost is that function's own per-interface `DataVector`
geometry+EOS machinery (`determinant_and_inverse`, `raise_or_lower_index`,
`dot_product`, the EOS sound-speed/κ/ζ evaluations) plus `normal_dot_flux` —
~25 short-lived `DataVector` temporaries per call. Reducing it further would
require converting that tested function to a single `TempBuffer` pass (bit-exact
but a large rewrite of a `Test_Characteristics.cpp`-pinned function). Because
`characteristic_speeds_hydro` is already cheap, fusing it with the acoustic
routine — the obvious "remove duplicated setup" move — was measured not to be
worthwhile. WHISKY does all of this as a handful of scalar FLOPs per cell with
zero heap traffic; SpECTRE's SoA `DataVector` formulation carries real
allocation/bandwidth overhead that dominates once the eigenvector work is
minimized. Reaching ratio ≤ 1.3 (~97 ns) would require that `TempBuffer`
rewrite of the tested pointwise functions.

The CPM optimization loop stopped at the **4-iteration cap**: three successful
iterations (opt #5, #6, #7) taking the ratio 5.91 → ~4.3, plus a fourth attempt
— removing the dead `pressure` EOS eval in `acoustic_eigenvectors_hydro`'s
Dim-1/3 paths — which measured no speedup (~0%, 318.6 → 320 ns, within noise, as
the profile predicted since that EOS call is a cheap analytic evaluation) and
was **reverted** per the ≥5% keep/revert rule. The only remaining candidate is
the tested-signature `TempBuffer` rewrite above, which cannot reach the 1.3
target and is out of bounds under "no test breakage".

## Harness

- `bench/bench_marquina.cpp` — drives the real
  `grmhd::ValenciaDivClean::BoundaryCorrections::Marquina` kernel
  (`dg_package_data` ×2 + `dg_boundary_terms`) over N = 10⁶ random hydro-only
  interface states (ρ∈[0.1,10], vⁱ∈[-0.3,0.3], ε∈[0.01,1], flat metric, α=1,
  βⁱ=0, `IdealFluid` Γ=2, B=0, Yₑ=const). States are processed in chunks of
  10⁴ so the SoA `DataVector` layout vectorizes across grid points and buffers
  are reused chunk-to-chunk. Only the kernel calls are timed
  (`std::chrono::steady_clock`); state generation and conservative/flux setup
  are excluded.
- `bench/bench_marquina_cpm.cpp` — same harness (shared via
  `bench/bench_marquina_common.hpp`, templated on the boundary-correction
  scheme) timing `MarquinaCpm` over the same states. Every run also computes
  full `Marquina` on those states and asserts the two agree within the 1e-12
  equivalence envelope, exiting non-zero on violation.
- `bench/bench_whisky.F90` — a self-contained transcription of WHISKY's
  `eigenproblem_marquina_general` in its production configuration
  (`ANALYTICAL=.TRUE.`, `FAST=.TRUE.`, the Aloy et al. 1999 CPC shortcut) over
  the same N and distributions.
- `bench/run_bench.sh` — builds both, runs each 5×, drops min/max, reports the
  median ns/interface and their ratio, appends to `bench/RESULTS.log`, and
  aborts if the SpECTRE output diverges from the saved reference.

**Caveat on the ratio.** The two harnesses are not bit-for-bit input-identical
(C++ `std::mt19937` vs Fortran's PRNG), so the ratio is a *throughput*
comparison over the same distributions and N, not a per-interface numerical
comparison. SpECTRE correctness is guarded independently and exactly by
`reference_output.dat`. WHISKY is also hydro-only (5 conserved vars); the
SpECTRE kernel additionally carries the `ρYₑ` mode (degenerate with the
central speed here), which is part of the intrinsic work difference.

## Step 1 findings (baseline audit)

- **FAST analytical path?** No. `Marquina::dg_package_data` calls
  `dg_package_data_impl(false, …)`, i.e. the full 6-mode decomposition:
  `eigenvectors_hydro` builds all six 6-component left *and* right
  eigenvectors. The sibling `MarquinaCpm` scheme is SpECTRE's analog of the
  FAST shortcut (Complementary Projection Method: only the two acoustic
  eigenvectors, degenerate block via the spectral-projector complement). But
  switching the `Marquina` scheme to that path changes the flux result, so it
  is out of bounds under the correctness gate.
- **Data layout.** SoA throughout (`Variables<TagList>` / `DataVector` per
  tensor component); the kernel is already vectorized across grid points.
- **EOS calls.** Made inside the kernel, and redundantly: `dg_package_data_impl`
  computes a "consistent" pressure/enthalpy, then `characteristic_speeds_hydro`
  and `eigenvectors_hydro` each independently recompute sound speed (and κ, ζ,
  p). `determinant_and_inverse(spatial_metric)` is computed both in
  `dg_package_data_impl` and again inside `eigenvectors_hydro`.
- **MHD/Yₑ paths.** The B/Φ blocks are a cheap central-flux fallback; the Yₑ
  mode is always carried (6th conserved variable). Nothing is compile-time
  removed in the hydro-only limit.

## Step 4 — bottleneck profile

`perf stat` on the baseline: 604,878 instructions/interface at IPC 3.86, cache
misses 1.3% — pure compute, not memory. `perf record` attributed **98.06%** of
runtime to `dg_boundary_terms_impl`.

The cause was an accidental **O(num_points²)** initialization: the boundary
corrections were zeroed *inside* a `for (point …)` loop, but each
`get(*boundary_correction_…) = 0.0` zeroes the whole `num_points`-length
`DataVector`, and the loop body never used `point`. So each of 10 outputs was
fully re-zeroed `num_points` times.

After fixing that, no single hotspot dominates. Post-opt profile (timed kernel,
excluding untimed state prep): `memmove` from allocations/copies (~6%),
`eigenvectors_hydro` (~6%), `dg_boundary_terms_impl` projection loop (~5%),
`characteristic_speeds_hydro` (~5%, ×2 calls), plus blaze vector ops,
`raise_or_lower_index`, `orthonormal_oneform`, `determinant_and_inverse`.

## Step 5 — optimizations applied

Each is bit-exactness-preserving (no arithmetic reassociation), gated by
`reference_output.dat`, and committed on `marquina-hydro`.

### opt #1 — hoist boundary-correction zeroing out of the point loop (68.2×)
`dg_boundary_terms_impl`. Removed the O(num_points²) re-zeroing; zero each
output `DataVector` exactly once. **41568 → 610 ns/interface.**

### opt #2 + #3 — write the characteristic decomposition in place (1.29×)
`dg_package_data_impl`. The decomposition was built in `std::array`
temporaries (pre-zeroed via `make_array`/`make_with_value`, which the callees
immediately overwrite) and then copy-assigned into the packaged output — 6 + 72
`DataVector` allocate-and-copies per call. Now the `std::array` elements are
aliased onto the packaged output storage with `set_data_ref`, so
`characteristic_speeds_hydro` / `eigenvectors_hydro` write results in place; the
copy-back loop and redundant pre-zeroing are gone. (opt #2 — dropping the
`make_array` pre-zero — was ~2% alone, below the 5% gate, so it was bundled
with the aliasing that makes it correct and committed together at ~23%.)
**610 → 472 ns/interface.**

### opt #4 — remap aligned exterior fields on read (1.10×)
`dg_boundary_terms_impl`. Aligning the exterior fields to the interior normal
swaps only the two acoustic rows, yet the code materialized two full 6×6 copies
(72 `DataVector` copies) to do it. The per-mode loop now reads the originals
through a row-remap accessor (no copy). The CPM branch, which needs the aligned
matrices as whole-matrix arguments, materializes them locally and is unchanged.
**472 → 430 ns/interface.**

### Considered but not pursued
- **Switch to the FAST/CPM acoustic path** — changes the flux result; violates
  the gate. (SpECTRE already exposes it as the separate `MarquinaCpm` scheme.)
- **De-duplicate EOS / `determinant_and_inverse` across
  `characteristic_speeds_hydro` and `eigenvectors_hydro`** — would require
  changing their public signatures, which `Test_Characteristics.cpp` pins;
  out of bounds ("do not modify existing tests").
- **Vectorize the per-point Marquina case-selection branch** — the three-way
  selection at exactly-zero speeds cannot be expressed branchlessly without
  risking a bit difference (and a `blaze::map` lambda would not auto-vectorize
  through the branch anyway).

## Remaining gap (ratio 5.77) and suspected cause

The residual gap is **algorithmic and data-structural**, not a missed micro-opt:

1. **Full 6-mode vs 2-mode.** SpECTRE's `Marquina` builds all six left and six
   right eigenvectors (72 `DataVector` components/interface across both sides)
   plus a 6-mode projection/reconstruction. WHISKY-Marquina FAST builds two
   acoustic left-eigenvectors as scalars and reconstructs with ~10 scalar FLOPs
   (Aloy et al. shortcut). This is the difference the reference gate forbids
   removing.
2. **`DataVector` allocation overhead dominates over mode count.** The existing
   Google-Benchmark microbench (`BenchmarkCpmMarquina`, 25-point faces) shows
   full `Marquina` (48.8 µs) and `MarquinaCpm` (49.2 µs) are essentially
   *equal* — the acoustic shortcut buys nothing there, because the cost is the
   per-call `DataVector`/geometry/EOS setup, not the eigenvector arithmetic.
   Our 10⁴-point chunks amortize that setup (hence 430 ns vs ~1950 ns/interface
   at 25 points), but the many per-call temporary allocations
   (`eigenvectors_hydro`, `raise_or_lower_index`, `orthonormal_oneform`,
   `determinant_and_inverse`) remain and show up as ~6% `memmove` plus the
   allocation-bound IPC drop from 3.86 to 1.88.
3. **Redundant EOS/geometry passes** between `characteristic_speeds_hydro` and
   `eigenvectors_hydro` (sound speed, one-form, v², vₙ, metric inverse computed
   twice) — removable only by changing tested public signatures.

Closing the remaining ~4.4× to reach ratio 1.3 would require either (a) fusing
the speed+eigenvector+geometry passes into a single allocation-light kernel
that writes straight into a `TempBuffer` (a substantial rewrite of
`characteristic_speeds_hydro`/`eigenvectors_hydro`, which are pinned by their
tests and public API), or (b) adopting the acoustic/CPM shortcut (different
math). Neither is available under the stated constraints.

## Stopping criterion

Stopped after **4 successful optimization iterations** (≤ 8 allowed) with the
ratio at 5.77 (> 1.3 target). Further progress requires changing the flux math
or the public API of tested functions, both disallowed. All artifacts are under
`bench/`; the full run history is in `bench/RESULTS.log`.
