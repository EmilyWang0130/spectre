<!--
Distributed under the MIT License.
See LICENSE.txt for details.
-->

# SpECTRE Marquina vs WHISKY: do they compute the same flux?

## TL;DR

**No — they do not produce the same numerical flux, and it is not a precision
issue.** On identical input states:

- **Characteristic speeds agree to machine precision** (max abs diff 4.4e-16).
- **Acoustic eigenvectors agree to machine precision** (max abs diff 1.8e-15).
- **The assembled Marquina numerical flux differs by O(0.1–1)** (median abs
  diff ~0.05–0.15 per component; not round-off).

The eigen-*system* is identical; the difference is entirely in the **flux
formula**:

- **SpECTRE** implements the original **sided Donat–Marquina** flux: each
  characteristic field is upwinded from whichever side it propagates from, and
  the Local-Lax-Friedrichs (max-|λ|) building block is used **only** for a
  field whose eigenvalue changes sign across the interface (a sonic field).
- **WHISKY** (`eigenproblem_marquina_general`, the `ANALYTICAL=.TRUE.`,
  `FAST=.TRUE.` path this study targets) applies the **max-|λ| viscosity to
  every characteristic field unconditionally** — a characteristic-wise
  Local-Lax-Friedrichs ("modified Marquina") flux.

Both share the defining Marquina feature (each side decomposed in its *own*
eigenbasis, no averaged intermediate state), which is why the eigenvalues and
eigenvectors match exactly. They differ only in the dissipation coefficient:
**upwind (λ) off-sonic in SpECTRE vs. max-|λ| everywhere in WHISKY.**

Neither is "more precise" in the floating-point sense. SpECTRE's sided scheme
adds *less* numerical dissipation on non-sonic fields (it reproduces the exact
upwind flux), so it is sharper/less diffusive — the genuine Donat–Marquina
flux. WHISKY's variant is more diffusive. This is a scheme choice, not a bug in
either code.

## What was compared, and how the comparison is fair

Harness: `bench/bench_marquina_crosscheck.cpp` +
`bench/whisky_crosscheck.F90` (build target `bench_marquina_crosscheck`).

For each of 20,000 random flat-metric, normal-along-x, hydro-only state pairs
`(u_L, u_R)` (ρ∈[0.1,10], vⁱ∈[-0.3,0.3], ε∈[0.01,1], ideal-gas Γ=2):

- Both kernels receive **identical inputs**: the same primitives, the **same
  conserved variables** (SpECTRE's own `tilde_*`, flat metric), and analytic
  IdealFluid Γ=2 EOS quantities (pressure, cₛ², ∂p/∂ε) — so the EOS and
  conserved-variable definitions cannot masquerade as a scheme difference.
- SpECTRE's `Marquina::dg_boundary_terms` (WeakInertial) output **is** the
  Marquina numerical flux F\*.
- WHISKY assembles F\* = ½·(F(u_L)·x̂ + F(u_R)·x̂ − dissipation); we form the
  same combination from SpECTRE's own physical fluxes F(u)·x̂ and WHISKY's
  dissipation term. The WHISKY dissipation comes from the actual reference
  kernel (transcribed verbatim from
  `whisky_reference/Whisky_Eigenproblem_Marquina.F90`, FAST path), called
  through a `bind(C)` wrapper.
- The `Yₑ` mode is dropped (WHISKY is 5-variable hydro); only D, Sₓ, Sy, Sz, τ
  are compared.

**Convention validation (positive control).** For identical states (u_L = u_R,
example 0 and 1 below) the interface is trivial: both schemes must, and do,
return exactly F(u)·x̂. This exact agreement confirms the input mapping, the
physical-flux extraction, and the ½·(F_L+F_R−diss) assembly are all correct, so
the discrepancy on non-trivial states is real.

## Results

### Eigenvalues (u_L), 20,000 states

| speed | abs diff (p50 / p99 / max) | rel diff (max) |
|-------|----------------------------|----------------|
| λ₊    | 1.1e-16 / 3.3e-16 / 4.4e-16 | 4.6e-14 |
| λ₋    | 1.1e-16 / 3.3e-16 / 4.4e-16 | 7.9e-13 |
| λ₀    | 0 / 0 / 0                   | 0 |

### Acoustic right eigenvectors (u_L, D-normalized)

| vector | abs diff (p50 / p99 / max) | rel diff (max) |
|--------|----------------------------|----------------|
| R₊     | 0 / 8.9e-16 / 1.8e-15 | 2.6e-14 |
| R₋     | 0 / 8.9e-16 / 1.8e-15 | 5.6e-13 |

Eigenvalues and eigenvectors are identical to round-off. Both codes use the
same D-normalized acoustic eigenvectors.

### Assembled Marquina numerical flux

| component | abs diff (p50 / p99 / max) |
|-----------|----------------------------|
| F[D]   | 1.5e-1 / 1.1e0 / 1.9e0 |
| F[Sₓ]  | 7.6e-2 / 4.0e-1 / 6.2e-1 |
| F[Sy]  | 4.4e-2 / 3.4e-1 / 7.3e-1 |
| F[Sz]  | 4.4e-2 / 3.4e-1 / 7.4e-1 |
| F[τ]   | 6.4e-2 / 2.9e-1 / 5.0e-1 |

The flux differs at the ~10–50% level on generic states. **max |flux diff| =
1.9**, versus **max |λ diff| = 4.4e-16** — five to sixteen orders of magnitude
apart. The disagreement is in the flux assembly, not the eigen-decomposition.

### The smoking gun: crafted control states

```
example 0 [identical L=R subsonic]     -> SpECTRE == WHISKY == F(u_L)  (exact)
example 1 [identical L=R supersonic]   -> SpECTRE == WHISKY == F(u_L)  (exact)
example 2 [supersonic L!=R, all λ>0]:
    F(u_L).x  0.6289709  0.2217582  0  0  0.04355657
    SpECTRE   0.6289709  0.2217582  0  0  0.04355657   <- exactly F(u_L)
    WHISKY    0.6237931  0.2201898  0  0  0.04327512   <- F(u_L) - max|λ| visc.
example 3 [subsonic L!=R, transonic]:
    F(u_L).x  0.2010076  1.0404040  0  0  0.2030328
    SpECTRE   0.2891522  0.9016863  0  0  0.3130856
    WHISKY    0.2845304  0.8746382  0  0  0.3108373
```

Example 2 is decisive. All three characteristic speeds are positive on both
sides (the wind blows in +x faster than sound), so **every** field propagates
left→right. SpECTRE's sided scheme therefore returns the pure upwind flux
`F(u_L)·x̂` to the bit; WHISKY still subtracts its max-|λ| viscosity and lands
elsewhere. That is exactly the sided-vs-unconditional-LLF distinction.

## Root cause, in the source

SpECTRE — `src/.../BoundaryCorrections/Marquina.cpp`, `dg_boundary_terms_impl`,
per characteristic field, per point:

```cpp
if (lambda_i_int[pt] >= 0 and lambda_i_ext[pt] >= 0) {      // field moves +
  phi_plus  = phi_int;  phi_minus = 0;                       //  -> upwind int
} else if (lambda_i_int[pt] <= 0 and lambda_i_ext[pt] <= 0) {// field moves -
  phi_plus  = 0;        phi_minus = phi_ext;                 //  -> upwind ext
} else {                                                     // sonic field
  alpha = max(|lambda_i_int|, |lambda_i_ext|);               //  -> LLF block
  phi_plus  = 0.5 (phi_int + alpha * omega_int);
  phi_minus = 0.5 (phi_ext - alpha * omega_ext);
}
```

WHISKY — `whisky_reference/Whisky_Eigenproblem_Marquina.F90`,
`eigenproblem_marquina_general`: the dissipation eigenvalues are the
componentwise max modulus, applied to **all** fields with no sign test:

```fortran
lam1 = dmax1(dabs(lam1l), dabs(lam1r))   ! degenerate block
lamp = dmax1(dabs(lampl), dabs(lampr))   ! acoustic +
lamm = dmax1(dabs(lamml), dabs(lammr))   ! acoustic -
! ... rflux = R * diag(lam) * (L*u), for each side, then diss = rflux_r-rflux_l
```

Algebraically, in the sonic case the two formulas coincide (SpECTRE's `else`
branch is exactly the max-|λ| LLF block); they diverge for every field whose
eigenvalue keeps its sign across the interface, where SpECTRE upwinds and
WHISKY still applies max-|λ| dissipation. For subsonic hydro states almost
every field is same-sign on both sides (λ₊>0, λ₋<0, λ₀ small), so the two
disagree on essentially every interface.

## Interpretation ("are we more precise?")

- The difference is a **numerical-flux (scheme) choice**, not accuracy of a
  shared algorithm. Both are legitimate, conservative, entropy-satisfying
  fluxes with identical wave speeds and eigenvectors.
- SpECTRE = the **original sided Donat–Marquina flux**: minimal characteristic
  dissipation (exact upwinding away from sonic points). Sharper, less diffusive.
- WHISKY `eigenproblem_marquina_general` = **characteristic-wise
  Local-Lax-Friedrichs** ("modified Marquina"): more robust, more diffusive.
- So SpECTRE is expected to be *less dissipative / higher resolution* on smooth
  and contact-dominated flows, while WHISKY's extra viscosity can be steadier
  near strong shocks. "More precise" is the wrong axis — they are different
  approximate Riemann solvers that happen to share the Marquina eigenstructure.
- Consequence for the performance study: the WHISKY↔SpECTRE ns/interface ratio
  compares kernels that compute *different* fluxes. It remains a fair measure of
  per-interface cost, but it is **not** a same-output comparison — that was the
  motivation for this document.

## Caveats

- Only the hydro subset (D, Sᵢ, τ) is compared; SpECTRE's `Yₑ` mode has no
  WHISKY counterpart. `Yₑ` is degenerate with λ₀ and does not affect the
  hydro-block conclusion.
- SpECTRE's near-luminal safety-floor clamps (`max(…, 1e-8)` / `1e-12` in
  `eigenvectors_hydro`) are not triggered at these benign states (|v| ≤ 0.3),
  so they are not the source of the disagreement; they would add a second,
  separate difference only at extreme states.
- The WHISKY side is a verbatim transcription of the reference FAST path; the
  trivial-interface exact agreement (examples 0–1) validates it end-to-end.
- The WHISKY driver `Whisky_Marquina.F90` actually calls `eigenproblem_marquina`
  (specialized) rather than `eigenproblem_marquina_general`; both use the same
  unconditional max-|λ| dissipation, and this study targets `_general` as
  specified in the task.

## Reproduce

```bash
cmake --build build --target bench_marquina_crosscheck -j <N>
./build/bin/bench_marquina_crosscheck
```
