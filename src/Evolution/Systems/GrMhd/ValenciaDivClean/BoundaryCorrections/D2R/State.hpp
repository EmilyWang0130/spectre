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
  // A literal, or for an unsupported midpoint "stored_tau_midpoint_" + its
  // recovery's reason in this thread's buffer (valid until the next such one).
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
  // The recoveries of left and right that accepted them (set with a nonzero
  // beta), so a caller need not repeat them.
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
// The energy root of the state with these primitives, x = (rho + Gamma p /
// (Gamma - 1)) W^2 - rho W: a start for recovering its conserved state.
RootHint primitive_hint(const State& p, double gamma);
Reference reference_from_tau(const State& ul, const State& ur,
                             const State& left, const State& right, double bn,
                             double gamma, const RootHint& hint = {});
PhysicalCap physical_ray_cap(const State& ul, const State& ur,
                             const State& shift, double bn, double gamma,
                             const RootHint& left_hint = {},
                             const RootHint& right_hint = {});

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
  Real margin(const State& flux) const;
  Real tolerance(const State& flux) const;
  LineResult repair(const State& candidate) const;
  LineResult cap(const State& baseline, const State& direction) const;

 private:
  State ul_{};
  State ur_{};
};

}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE
