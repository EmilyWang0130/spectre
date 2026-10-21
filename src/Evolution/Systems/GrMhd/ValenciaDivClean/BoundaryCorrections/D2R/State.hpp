// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/D2R/Types.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE {

// Gamma-law, fixed common Bn, c=1. P=(rho,vn,vt,vb,p,Bt,Bb),
// U/F=(D,mn,mt,mb,tau,Bt,Bb). Numerical high-precision support only:
// no outward intervals, general EOS, CT/GLM/RK, or thermal-relative proof.
struct Admissibility {
  bool admissible = false;
  std::string reason{};
  Real q = 0;
  Real psi = 0;
};

struct Recovery {
  bool success = false;
  std::string reason{};
  State primitive{};
  MpState high_precision_primitive{};
  std::size_t iterations = 0;
  std::size_t boundary_bisections = 0;
  std::size_t energy_bisections = 0;
  std::string boundary_termination{};
  std::string energy_termination{};
  bool roundtrip_checked = false;
  double reconstruction_error = 0.;
  double thermal_relative_error = 0.;
  Admissibility admissibility{};
};

struct EntropyPair {
  bool success = false;
  std::string reason{};
  Recovery recovery{};
  MpState z{};
  Real psi = 0;
  Real eta = 0;
};

struct Reference {
  bool success = false;
  std::string reason{};
  State primitive{};
  std::array<Real, 4> four_velocity{};
};

struct PhysicalCap {
  bool success = false;
  std::string reason{};
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
  std::string reason{};
  State flux{};
  Real coefficient = 0;
  Real margin = 0;
};

bool valid_gamma(double gamma);
bool physical_primitive(const State& p, double bn, double gamma);
State conserved_tau(const State& p, double bn, double gamma);
State flux_tau(const State& p, double bn, double gamma);
Admissibility admissibility(const State& u, double bn, double gamma);
Recovery recover(const State& u, double bn, double gamma);
EntropyPair entropy_pair(const State& u, double bn, double gamma);
Reference reference_from_tau(const State& ul, const State& ur,
                             const State& left, const State& right, double bn,
                             double gamma);
PhysicalCap physical_ray_cap(const State& ul, const State& ur,
                             const State& shift, double bn, double gamma);

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

  EntropyModel(const State& ul, const State& ur, double bn, double gamma);
  Real margin(const State& flux) const;
  Real tolerance(const State& flux) const;
  LineResult repair(const State& candidate) const;
  LineResult cap(const State& baseline, const State& direction) const;

 private:
  State ul_{};
  State ur_{};
};

}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE
