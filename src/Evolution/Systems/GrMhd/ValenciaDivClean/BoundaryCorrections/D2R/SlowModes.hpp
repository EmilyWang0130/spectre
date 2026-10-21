// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <string>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/D2R/Types.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE {

struct SlowAction {
  bool supported = false;
  bool zero_operator = false;
  bool normalization_bound_active = false;
  // A literal, or an exception's message in this thread's buffer (valid until
  // the next one).
  const char* status = "not_evaluated";
  // Order f-,a-,s-,c,s+,a+,f+. Exactly Bn=0 exits before a spectrum is needed,
  // so these diagnostic speeds remain zero on that analytical zero branch.
  std::array<double, 7> lambdas{};
  // Slow-family order s-,s+. No fitted activation or conditioning multiplier.
  std::array<double, 2> weights{};
  State slow_jump{};
  State absolute_slow_jump{};
  // Preserve all MP contraction arithmetic for independent audit/future
  // callers. Current prototype semantics round each complete G,Q action to
  // binary64.
  MpState slow_jump_mp{};
  MpState absolute_slow_jump_mp{};
};

// Compute G=Ws(UR-UL), Q=|A|Ws(UR-UL) using true magnetosonic scalar roots and
// Saul Teukolsky's two analytic conserved slow pairs. No characteristic-matrix
// eigensolve/inverse, EOS floor, fitted gap/conditioning gate or Python
// runtime. The family polynomial, acoustic normalization and maximum bound are
// the same explicit choices as the audited scalar SR construction prototype.
// Positive gamma-law flat-SR primitives and one fixed Bn are the supported
// system.
SlowAction slow_action(const State& primitive, double bn, double gamma,
                       const State& left_tau, const State& right_tau);

}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE
