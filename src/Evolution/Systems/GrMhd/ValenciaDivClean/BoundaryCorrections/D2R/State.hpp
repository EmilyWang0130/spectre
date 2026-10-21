// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <string>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/D2R/Types.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE {

// Gamma-law, fixed common Bn, c=1. P=(rho,vn,vt,vb,p,Bt,Bb),
// U/F=(D,mn,mt,mb,tau,Bt,Bb). Numerical high-precision support only:
// no outward intervals, general EOS, CT/GLM/RK, or thermal-relative proof.
// Statuses and reasons are string literals (2026-10-20: std::string members
// cost several heap allocations per face); EntropyModel keeps its composite
// reason as a std::string, built only when a model is unsupported.
struct Admissibility {
  bool admissible = false;
  const char* reason = "";
  Real q = 0;
  Real psi = 0;
};

// Starting points for the root searches of a recovery (2026-10-20): the
// pressure-zero boundary and the energy root, in the unknown x = z - D, NaN
// for none. They only seed the Newton iteration that bounds the bisection's
// foregone decisions, so they change the cost of a recovery, never its result.
struct RootHint {
  double boundary = std::numeric_limits<double>::quiet_NaN();
  double energy = std::numeric_limits<double>::quiet_NaN();
};

struct Recovery {
  bool success = false;
  const char* reason = "";
  State primitive{};
  MpState high_precision_primitive{};
  std::size_t iterations = 0;
  std::size_t boundary_bisections = 0;
  std::size_t energy_bisections = 0;
  const char* boundary_termination = "";
  const char* energy_termination = "";
  bool roundtrip_checked = false;
  double reconstruction_error = 0.;
  double thermal_relative_error = 0.;
  Admissibility admissibility{};
  // The two roots found (the boundary and the lower end of the final energy
  // bracket), to seed the recoveries of nearby states.
  RootHint roots{};
};

struct EntropyPair {
  bool success = false;
  const char* reason = "";
  Recovery recovery{};
  MpState z{};
  Real psi = 0;
  Real eta = 0;
};

struct Reference {
  bool success = false;
  const char* reason = "";
  State primitive{};
  std::array<Real, 4> four_velocity{};
};

struct PhysicalCap {
  bool success = false;
  const char* reason = "";
  double beta = 0.;
  State left{};
  State right{};
  std::size_t evaluations = 0;
  bool full_supported = false;
  // True when the helper primitives below are linear predictions rather than
  // recoveries of left/right. The predicted primitives are physical, and
  // left/right pass admissibility, but their encoding differs from left/right
  // at second order in the shift. No round-trip check is claimed in this case.
  bool primitive_prediction = false;
  // Recoveries of left/right, or physical primitive predictions when
  // primitive_prediction is true, so the caller need not recompute them.
  Recovery left_recovery{};
  Recovery right_recovery{};
};

struct LineResult {
  bool supported = false;
  const char* reason = "";
  State flux{};
  Real coefficient = 0;
  Real margin = 0;
};

bool valid_gamma(double gamma);
bool physical_primitive(const State& p, double bn, double gamma);
State conserved_tau(const State& p, double bn, double gamma);
State flux_tau(const State& p, double bn, double gamma);
Admissibility admissibility(const State& u, double bn, double gamma);
Recovery recover(const State& u, double bn, double gamma,
                 const RootHint& hint = {});
EntropyPair entropy_pair(const State& u, double bn, double gamma,
                         const RootHint& hint = {});
// The entropy pair evaluated directly at these primitives. This does not
// recover or verify any independently stored conserved encoding, and is not
// the entropy model used to certify an actual stored-state pair.
EntropyPair entropy_pair_from_primitive(const State& p, double bn,
                                        double gamma);
// The energy root of the state with these primitives, x = (rho + Gamma p /
// (Gamma - 1)) W^2 - rho W: a start for recovering its conserved state.
RootHint primitive_hint(const State& p, double gamma);
// The reference state of the slow modes (design D4, 2026-10-21): density and
// pressure the averages of the face primitives left, right, the transverse
// field the midpoint of the stored states ul, ur, the velocity from the
// square-root-density weighted average of the four-velocities. (Originally
// density and pressure were recovered from the stored midpoint (ul + ur) / 2;
// the two differ at second order in the jump.)
Reference reference_from_primitives(const State& ul, const State& ur,
                                    const State& left, const State& right,
                                    double bn, double gamma);
PhysicalCap physical_ray_cap(const State& ul, const State& ur,
                             const State& shift, double bn, double gamma,
                             const RootHint& left_hint = {},
                             const RootHint& right_hint = {});
// Design D3 (2026-10-21): try the full shift (beta = 1), requiring the intended
// conserved states ul+shift, ur-shift to pass admissibility. Form separate
// physical helper primitives with dP = (dU/dP)^-1 dU at left/right; these are
// predictions, not recoveries of the shifted states, and their encodings
// differ from the intended shifts at second order. success = false (the caller
// then takes physical_ray_cap) for unsupported shifted states, an unphysical
// primitive prediction, or a singular linear system.
PhysicalCap linear_ray_cap(const State& ul, const State& ur, const State& left,
                           const State& right, const State& shift, double bn,
                           double gamma);

// Exact represented affine arithmetic, rounded once, ties to even. Return
// false for nonfinite input/output. coefficient is its represented decimal
// Real value; direction is imported from represented binary64 values.
bool rounded_ray(State* result, const State& base, const State& direction,
                 const Real& coefficient);
bool causal_hll(State* result, const State& ul, const State& ur,
                const State& fl, const State& fr);

struct EntropyModel {
  bool supported = false;
  std::string reason{};
  EntropyPair left{};
  EntropyPair right{};
  MpState dz{};
  MpState jump{};
  Real dpsi = 0;
  Real h = 0;

  EntropyModel(const State& ul, const State& ur, double bn, double gamma,
               const RootHint& left_hint = {}, const RootHint& right_hint = {});
  // Recover the actual stored states ul, ur before forming their entropy
  // model. Supplied primitives only provide starting hints for the recovery;
  // no thermal-energy threshold substitutes their entropy variables for the
  // recovered ones.
  EntropyModel(const State& ul, const State& ur, const State& left_primitive,
               const State& right_primitive, double bn, double gamma);
  Real margin(const State& flux) const;
  Real tolerance(const State& flux) const;
  LineResult repair(const State& candidate) const;
  LineResult cap(const State& baseline, const State& direction) const;

 private:
  void complete();
  State ul_{};
  State ur_{};
};

}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE
