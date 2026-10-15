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
   falls back to HLL. Equal primitives use their physical flux exactly.
3. Recover and check the **actual stored** original states. Form their numerical
   entropy model. If necessary repair the baseline along `-nu (UR-UL)` using
   the derived entropy and binary64 rounding budget. A thermodynamically
   unsupported original encoding makes the callback reject the point; the
   SpECTRE adapter retains HLL. It is not counted as a supported D2-R result.
4. Use the recovered midpoint thermodynamics and averaged timelike velocity
   to form the reference. Compute the true SR magnetosonic roots and Saul's
   analytic left/right slow pairs. The complete weighted actions are
   `G=Ws(UR-UL)` and `Q=|A|Ws(UR-UL)`. No characteristic matrix inverse or
   fitted condition-number cutoff is used.
5. Select a numerically supported physical input coefficient `beta` for
   `UL,c=UL+beta G/2`, `UR,c=UR-beta G/2`. Recover these states. If stored or
   recovered inputs do not change, retain the baseline before restoring a
   formal slow contribution.
6. With the **same original bounds**, call the same fixed helper on recovered
   original and modified states. Check both actual C re-encodings. Form
   `V0=K0-(F0,L+F0,R)/2`, `Vc=Kc-(Fc,L+Fc,R)/2` and freeze
   `T=Vc-V0-beta Q/2`.
7. Select `alpha` on the original-input entropy model and return
   `F=Fb+alpha T`. Each final affine component is summed as an exact rational
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
recoveries. Calls are serialized in this initial kernel; it is not a
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
- `SPECTRE_D2R_ROUNDOFF_TOL=tol` returns the native HLLD flux of the pair, with no
  slow-mode response, on faces whose states differ by at most `tol` relative in
  every component (rho and p by themselves, velocities by c, the field by |B|).
  SpECTRE's FD slab is transversely uniform only to ~1e-12, so a "1-D" tube
  otherwise pays a full 0.15 s solve on every transverse face. The tube arms use
  1e-10. This is a round-off floor, not a physical activation band; with it
  unset the kernel is exactly the audited one.

Independent of these, bit-identical left and right states return their physical
flux at once, and bit-identical faces within one batch are solved once.
