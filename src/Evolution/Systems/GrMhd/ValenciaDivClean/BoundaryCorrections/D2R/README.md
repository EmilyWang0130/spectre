# Experimental compiled SR D2-R

`PlutoD2R` is a selectable SpECTRE ValenciaDivClean boundary correction. The
complete face calculation is C++/C; it does not invoke Python or use HLLx,
an exact solution, or tabulated reference fluxes at runtime.

Select it in the usual boundary-correction option:

```yaml
BoundaryCorrection:
  PlutoD2R:
    MagneticFieldMagnitudeForHydro: 1.0e-30
    LightSpeedDensityCutoff: 1.0e-30
```

These two inherited options select the outer packaging speeds; they do not
set a slow-wave activation threshold. The equation of state must be SpECTRE's
`Equilibrium3D<IdealFluid<true>>`, with `1 < Gamma <= 2`. The supported D2-R
calculation requires Cartesian Minkowski metric, unit lapse, zero shift and
zero normal mesh velocity. The full spatial metric is checked. Other faces
use HLL with causal light-cone bounds and the supplied conservative variables
and physical fluxes. This is an SR implementation, not a GR construction.

## Calculation

The seven states are `U=(D,mn,mt,mb,tau,Bt,Bb)`, where `tau=E-D`, and
`P=(rho,vn,vt,vb,p,Bt,Bb)` with one separate common normal field `Bn`.

1. Obtain the actual stored conservative states, physical fluxes and original
   fan bounds from PLUTO's C mapper/native HLLD. A native magnetosonic-speed
   failure may instead use explicitly labelled HLL(-1,+1), assembled from
   finite actual C physical maps. Its slow correction is disabled.
2. For a mixed fan, form a pressure-coherent baseline using the private v5
   fixed-bound HLLD helper. Its root refinement either succeeds or explicitly
   falls back to HLL. Equal primitives use their physical flux exactly. (The
   batch-level round-off shortcut, `SPECTRE_D2R_ROUNDOFF_TOL`, returns this same
   baseline since 2026-10-20.)
3. Form the numerical entropy model of the **actual stored** original states
   with the entropy variables of the face's primitives, of which they are the
   encoding (design D1, 2026-10-21; originally recovered from the stored
   states). A state whose stored encoding does not resolve its thermal energy
   (below 2^-32 of the recovery's energy terms: a cold, fast state) is recovered
   and checked as originally (D1b). If necessary repair the baseline along
   `-nu (UR-UL)` using the derived entropy and binary64 rounding budget. A
   thermodynamically unsupported original encoding makes the callback reject
   the point; the SpECTRE adapter retains HLL. It is not counted as a supported
   D2-R result.
   The repair is bounded (2026-10-20): a repair that would move the baseline
   by more than `C s_max |UR-UL|` (C = 2, `SPECTRE_D2R_CONSISTENCY`) rejects the
   point as well; without it, nearly uniform pairs got baselines off by
   1e-5..3e-4 of the flux.
4. Use the averaged primitive density and pressure (D4; originally the
   recovered stored midpoint's) and the averaged timelike velocity to form the
   reference. Compute the true SR magnetosonic roots (bracket-safeguarded Newton
   from the mean of PLUTO's speeds of the two sides, D2 and D8) and Saul's
   analytic left/right slow pairs. The complete weighted actions are
   `G=Ws(UR-UL)` and `Q=|A|Ws(UR-UL)`. No characteristic matrix inverse or
   fitted condition-number cutoff is used.
5. Select a numerically supported physical input coefficient `beta` for
   `UL,c=UL+beta G/2`, `UR,c=UR-beta G/2`. Their primitives are the face
   primitives plus the linear update `(dU/dP)^-1 dU` at `beta = 1` (D3); where
   an updated primitive is unphysical or the system singular, recover these
   states with the original `beta` search. If stored or recovered inputs do not
   change, retain the baseline before restoring a formal slow contribution.
6. With the **same original bounds**, call the same fixed helper on the
   original and modified states (the native bounds and the original states'
   fixed-bound solve come from one C call, D6; the helper refines its pressure
   root by Illinois regula falsi, D5). Check both actual C re-encodings. Form
   `V0=K0-(F0,L+F0,R)/2`, `Vc=Kc-(Fc,L+Fc,R)/2` and freeze
   `T=Vc-V0-beta Q/2`.
7. Scale `T` down to the same bound `C s_max |UR-UL|` if it exceeds it
   (`FaceResult::consistency`; the margin is affine along the ray, so the
   entropy condition survives the scaling). Select `alpha` on the
   original-input entropy model and return `F=Fb+alpha T`. Each final affine component is summed as an exact rational
   before rounding once to binary64, with explicit post-checks. Unsupported
   correction steps retain the supported baseline. Exactly one-sided original
   fans retain their upwind baseline.

The shared normal-frame adapter supplies common GLM `Bn`, the normal-field
and cleaning fluxes and GLM energy pairing. On an accepted D2-R mass flux,
it advects the upwind concentration `TildeYe/TildeD` with that same mass flux.
Thus constant Ye obeys `G(DYe)=Ye G(D)`. Rejected points retain the complete
HLL mass/composition pair. This behavior is enabled only for `PlutoD2R`.
`PlutoD2R` explicitly
enables common `Bn` and pairing without changing legacy environment switches.
Strong and weak DG forms use the same returned interface flux.

## Two builds of the kernel (2026-10-19)

The same sources are compiled twice (`Types.hpp` selects the scalar type by the
`D2R_MULTIPRECISION` macro; the `*Mp.cpp` wrappers set it):

- **double** (namespace `d2r_detail`, the default): `Real = double`; the exact
  rational bookkeeping of the correctness backend becomes ordinary arithmetic
  carried in `long double` and rounded once; the entropy margins are accumulated
  in `long double` and compared against a round-off tolerance of
  32 ε (|Δψ| + Σ|dz_i F_i|) instead of exact zero; the recovery's round-trip
  budget is eight times the binary64 model's, because the bisection root is only
  resolved to binary64. Everything else — the steps, the fallbacks, the weights,
  the caps — is the same code.
- **160-digit** (namespace `d2r_detail_mp`): task 3's original correctness backend,
  selected at run time by `SPECTRE_D2R_PRECISION=mp` (also `160`,
  `multiprecision`). It is the oracle: the face harness in the research workspace
  (`notes/projects/active/hllx_rmhd/scripts/d2r_face_harness/`,
  `harness6_double_vs_mp.cpp.txt`) runs both on stored faces and reports the
  largest relative flux difference, the support agreement and the time per face.

The `extern "C"` entry points dispatch to one of the two; the SpECTRE adapter and
the unit tests do not know which. The paragraphs below describe the 160-digit
backend as it was written; they apply to the double build with the
qualifications above.

### The double kernel's cost, before and after the optimization of 2026-10-20

Branch `d2r-opt`; every change is bit-identical to the unchanged kernel
(per-face dumps of 69 000 stored faces and 14 evolution arms, element by
element), so the agreement with the 160-digit oracle is the unchanged kernel's:
worst relative flux difference 2.0e-15 and 3.3e-14 on the two tube face sets,
1.08e-12 on the jet sets. Measured in one job on one node, one pinned core,
median of 7 repetitions (`spectre_runs/branch_testing/run09_d2r_optimization`,
job 79354):

| | before | after | PLUTO HLLD | PLUTO HLLx |
|---|---|---|---|---|
| µs per active face, relativistic large-B_x tube (156 faces) | 37.2–37.7 | 17.0 | 0.70 | 4.25 |
| µs per active face, cold large-B_x tube (159) | 39.7–39.8 | 16.8–17.0 | 0.68 | 4.62 |
| µs per active face, 44 × 88 jet, t = 0.5 … 3.5 (4063 of 53 284) | 40.2 | 19.8 | 0.64 | 6.1 |
| 44 × 88 jet to t = 3, 1 worker (wall, 2 runs) | 27.9 s | 19.1–19.2 s | 11.8 s | 16.0 s |
| 44 × 88 jet to t = 3, 16 workers | 20.2–20.3 s | 8.2 s | 2.0 s | 3.7 s |

What changed (one commit each): the double build takes no lock (its C core keeps
all scratch `_Thread_local`); `recover()` evaluates only the bisection midpoints
inside a window bounded by a safeguarded Newton search and rounding-margin
probes, starting from nearby known roots, and reuses the physical-ray cap's
recoveries (5 instead of 7 per active face); the response's first fixed-bound
solve reuses the refined solve when its inputs are equal; two repeated
computations in the C bridge are dropped; `sqrt` is the instruction and `x / 2`
is `x * 0.5` (the flags' `-ffp-exception-behavior=maytrap` otherwise keeps a
libm call and a division on every bisection step); statuses are string literals
(no heap allocation per face); the expected unsupported spectra are returned
rather than thrown.

Where the time goes now (flat profile on jet faces): primitive recovery ~45 %
(about 115 bisection steps per recovery, kept to stay bit-identical); the C core
~30 %, mostly the fixed-bound solver's refinement bisection (16–27 `Fstar`
evaluations per solve on jet faces); the quartic roots ~10 %. On 16 workers the
jet is limited by its element distribution: one worker carries ~60 % of the D2-R
solves.

### Design changes of 2026-10-21 (branch `d2r-design`)

Not bit-identical: each change keeps the method (HLLD bounds and candidate,
entropy repair, consistency bound, slow-mode response, entropy cap) and changes
how one ingredient is computed, in both builds, so the 160-digit build stays the
oracle of the same design. After D4 no primitive recovery is left on the common
path; the active path makes two C core calls.

| step | what changes | flux change against the step before |
|---|---|---|
| D1 | entropy variables of the face's primitives, not recovered from the stored states | 1.5e-12 |
| D1b | ... except a state whose encoding does not resolve its thermal energy, recovered as before | none on the face sets; restores the unit test's rejection |
| D2 | quartic roots by bracket-safeguarded Newton, not bisection | 3.4e-14 |
| D3 | shifted states' primitives by the linear update `(dU/dP)^-1 dU`, recovery only as fallback | second order in the jump: tubes 5e-6, jet t = 3 3e-8, one strong-jump face 1.6e-3 |
| D4 | reference density and pressure from the face primitives, not the recovered midpoint | second order: tubes 1.5e-6, jet 2.4e-5, one strong-jump face 1.4e-3 |
| D5 | fixed-bound root refined by Illinois regula falsi (same termination) | round-off (4e-14) |
| D6 | native bounds and the first fixed-bound solve in one C call, no native solve | bit-identical |
| D8 | root searches start from PLUTO's speeds of the two sides; shared endpoints evaluated once | round-off (1.5e-14) |

Over all 61 472 stored faces, saved kernel → D8: 8911 faces change, 153 by more
than 1e-6 relative, 5 by more than 1e-4 (worst 2.4e-3, a 95 % density jump of
the t = 35 jet); no support changes. Agreement with the 160-digit build of the
same design: 1.6e-15 / 6.5e-14 on the tube sets, 2.6e-13 on the 22 × 44 jet set.
Measurements, gates and the tube L1 tables against the exact solutions:
`spectre_runs/branch_testing/run09_d2r_optimization` (README, design round).

## Precision, support and cost

This first compiled correctness backend uses fixed 160-digit decimal
arithmetic, with expression templates disabled. Roots, normalization,
thermodynamic recovery and entropy signs are numerical computations, **not
outward interval proofs**. The complete slow actions are converted to
binary64 once; reference averaging uses high precision. The earlier Python
prototype used some long-double intermediate conversions, so differential
checks allow rounding differences and do not claim bitwise port identity.

Exact `Bn=0` and the analytically handled parallel slow/Alfven collision have
zero-action exits. An isolated acoustic slow pair remains active at a
fast/Alfven collision. General nonparallel exact quartic endpoint collisions
are explicitly unsupported, rather than assigning duplicate family roots.

The baseline helper retains inherited PLUTO root decisions, including HLL
fallbacks. Supported flux does not imply every baseline was an HLLD fan.
Original cold/high-Lorentz-factor C encodings can fail admissibility even if
the intended primitive state was physical. A fallback is not a certificate
of positivity for a complete finite-volume or DG update.

The active path normally makes four C core calls and several high-precision
recoveries (on `d2r-design`, 2026-10-21: two core calls and no recovery on the
common path). Calls are serialized in this initial kernel; it is not a
production-speed implementation. Private contexts and root scratch are
thread-local, with capacity at least 1026 to remain compatible with legacy
PLUTO batches. Concurrent initialization alongside unprotected legacy
PLUTO allocation accounting has not been certified.

Factory registration, serialization and face tests are necessary preflight.
Evolved SR shocktubes, multidimensional evolution, GR, general/tabulated EOS,
full GLM entropy analysis and neutron-star/BNS validation remain separate
requirements. Build `Test_ValenciaDivClean` and run the PlutoD2R tests with
`ctest --output-on-failure` in a normally configured SpECTRE build.

## Integration switches (task 2, 2026-10-15)

Two environment variables, both off when unset, added when the kernel was merged
into `hydroYe-mhd-tov` and first evolved:

- `SPECTRE_D2R_STATS=N` prints a cost line to stderr every N batch calls: points,
  solves (equal / active / unsupported), near-equal faces, memo hits, kernel
  seconds against wall seconds, slowest face.
- `SPECTRE_D2R_ROUNDOFF_TOL=tol` returns the baseline of the pair (since 2026-10-20 the
  fixed-bound solve with the pair's native bounds, the same solve as the candidate; before,
  native HLLD), with no
  slow-mode response, on faces whose states differ by at most `tol` relative in
  every component (rho and p by themselves, velocities by c, the field by |B|).
  SpECTRE's FD slab is transversely uniform only to ~1e-12, so a "1-D" tube
  otherwise pays a full 0.15 s solve on every transverse face. The tube arms use
  1e-10. With the same tolerance, points of one batch whose left and right
  inputs agree component by component reuse the first point's flux (the 25
  transverse copies of a tube's x-face differ at 1e-12 too). This is a
  round-off floor, not a physical activation band; with it unset the kernel is
  exactly the audited one.

Independent of these, bit-identical left and right states return their physical
flux at once, and bit-identical faces within one batch are solved once.

## Consistency bound and the near-equal baseline (2026-10-19/20)

Two defects found by the transverse-symmetry investigation of the 1-D cold large-B_x tube on
the 3-D slab (`~/spectre_runs/branch_testing/run07_pluto_d2r/sensitivity/README.md`; with
the kernel of 2026-10-19 the 25 transverse copies of every cell, bit-identical at t = 1,
differed by 0.1–0.3 % in ρ from t ≈ 1.7 on, while HLLD and HLLx stayed at 1e-10):

- **The entropy repair was not tied to the jump.** Its coefficient divides by the pair's
  entropy jump; on pairs differing by 2e-10 it moved the baseline by 1e-5 (transverse normal)
  to 3e-4 (x normal) of the flux, a million times the physical jump. The kernel now enforces
  `|F - F_candidate| <= C s_max |U_R - U_L|` with `C = SPECTRE_D2R_CONSISTENCY` (default 2,
  0 = off): a repair beyond the bound leaves the face unsupported (the shim keeps its HLL flux,
  which is consistent), and a response beyond it is scaled down to it, which keeps the entropy
  condition because the margin is affine along the ray. On the relativistic tube the response
  stays below 0.93 s_max |dU| and the repair below 4e-7 of the flux, so nothing changes there
  (tube L1 errors identical to every printed digit). `FaceResult::consistency` records the
  scaling; the stats line counts `consistency-capped` and `repair-bound`.
- **The near-equal path returned native `hlld.c`, whose pressure iteration stops at 1e-5 of p,
  while the full path's candidate is the fixed-bound solve, whose root is refined to adjacent
  doubles.** A face crossing `SPECTRE_D2R_ROUNDOFF_TOL` therefore jumped by 1e-5 of the flux,
  which is what broke the symmetry once the copies' pressures had drifted that far apart. The
  near-equal path now returns the same fixed-bound solve (native bounds, when they straddle
  zero). Near-uniform transverse pairs: error 1.2e-9 of the flux instead of 9.8e-6 (HLLD's
  own); jitter under one-ulp nudges 4.8e-7 of the mass flux instead of 3.6e-4.
- `SPECTRE_D2R_JUMP_FLOOR=eta` (default 0 = off) multiplies the response by
  `min(1, (|dU| / (eta |U|))^2)`. Tested at 1e-3: neutral for the tube errors (within 0.02) and
  for the symmetry; kept as an experiment switch only.

Result (80-cell cold tube, memo at 1e-10): transverse asymmetry 1.6e-10 of ρ at t = 10
(6.6e-3 before; HLLx 2.7e-10 of the range, HLLD 4e-12), L1 within 0.02 of before, 613 s wall.
Iterations: jobs 79294 (bound), 79295 (+ jump floor), 79296 (+ near-equal baseline). The
x-faces were never the problem: under round-off-sized input changes the kernel's x-flux moves
exactly as little as HLL's (3e-11 on the mass flux for 1e-10 changes; `harness16`).

## The family-filter weight (decided 2026-10-19)

Step 4's weights `Ws` multiply the slow-family projections of the jump. The
default is the Roe–Balsara law,

    w = (b_A / f^2) * alpha_s^2,   alpha_s^2 = (f^2 - c_s^2) / (f^2 - s^2),

per side with that side's rest-frame slow and fast speeds `s`, `f`, the normal
Alfvén speed squared `b_A` and the sound speed `c_s`; `alpha_s` is the standard
normalization factor of the slow magnetosonic eigenvectors, 1/2 at the umbilic
by continuity. It vanishes with the normal field and at slow–Alfvén coalescence
and is 1 where the slow wave is acoustic-like, whatever the temperature.
`SPECTRE_D2R_WEIGHT=polynomial` restores the audited weight of the Newtonian
design, `(b_A/f^2)(1 - c_s^2/f^2)^2 (1 - s^2/f^2)`, which halves on a warm
plasma (c_s^2/f^2 = 0.17 on the relativistic large-B_x tube) and left D2-R with
half of HLLx's gain there; `linear` gives `(b_A/f^2)(1 - c_s^2/f^2)`. Evidence
(relativistic large-B_x tube against its exact GR06 solution, L1 of rho in % of
the exact range, |x| < 0.4): 80 cells HLLD 9.81, polynomial 8.22, HLLx 6.72,
linear 7.17, roe-balsara 6.43; 160 cells 7.02 / 5.95 / 4.91 / 5.26 / 4.73; the
same ordering in p, v and B. What the simple zero gives up against the audited
weight is the quadratic vanishing at slow–Alfvén coalescence: `w Pi_s` stays
bounded there instead of going to zero, with the caps `alpha` and `beta` as the
remaining guards. The method as it runs, step by step, is written up in
`notes/projects/active/hlld_slow_antidiffusion/D2R_SR_METHOD.md` of the research
workspace.
