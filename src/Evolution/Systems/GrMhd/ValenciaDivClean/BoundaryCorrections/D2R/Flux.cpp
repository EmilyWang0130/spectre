// Distributed under the MIT License.
// See LICENSE.txt for details.
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/D2R/Flux.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <mutex>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/D2R/SlowModes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/D2R/State.hpp"

extern "C" {
#include "spectre_d2r_core.h"
}

namespace grmhd::ValenciaDivClean::BoundaryCorrections::d2r_detail {
namespace {
struct CoreResult {
  bool api_ok = false;
  State flux{}, ul{}, ur{}, fl{}, fr{};
  std::array<double, 2> bounds{};
  std::array<int, D2R_CORE_NI> integers{};
  std::array<double, D2R_CORE_ND> doubles{};
  int status() const { return api_ok ? integers[0] : -100; }
};

CoreResult core(const int mode, const State& left, const State& right,
                const double bn, const double gamma,
                const std::array<double, 2>& bounds = {}) {
  std::array<double, 16> input{};
  input[0] = gamma;
  input[1] = bn;
  std::copy(left.begin(), left.end(), input.begin() + 2);
  std::copy(right.begin(), right.end(), input.begin() + 9);
  std::array<double, 14> u{}, f{};
  CoreResult result;
  result.api_ok = d2r_sr_core_batch(
                      1, mode, input.data(), bounds.data(), result.flux.data(),
                      u.data(), f.data(), result.bounds.data(),
                      result.integers.data(), result.doubles.data()) != 0;
  std::copy_n(u.begin(), 7, result.ul.begin());
  std::copy_n(u.begin() + 7, 7, result.ur.begin());
  std::copy_n(f.begin(), 7, result.fl.begin());
  std::copy_n(f.begin() + 7, 7, result.fr.begin());
  return result;
}

bool finite(const State& a) {
  return std::all_of(a.begin(), a.end(),
                     [](const double x) { return std::isfinite(x); });
}
}  // namespace

FaceResult solve_face(const State& left, const State& right, const double bn,
                      const double gamma) {
  // Serialize this first correctness embedding, including lazy PLUTO scratch
  // allocation. Legacy PLUTO methods retain their existing threading contract.
  static std::mutex kernel_mutex;
  const std::lock_guard<std::mutex> guard(kernel_mutex);
  FaceResult result;
  result.status = "invalid_primitive";
  if (not physical_primitive(left, bn, gamma) or
      not physical_primitive(right, bn, gamma)) {
    return result;
  }
  auto base = core(0, left, right, bn, gamma);
  result.core_calls = 1;
  result.native_status = base.status();
  if (base.status() == -2) {
    // Fetch each first-side map separately: the C bridge prepares that map
    // before reporting a wave-speed failure. No failed root is reused.
    const auto lmap = core(2, left, left, bn, gamma);
    const auto rmap = core(2, right, right, bn, gamma);
    result.core_calls += 2;
    const auto map_ok = [](const CoreResult& c) {
      return (c.status() == 1 or c.status() == -2) and finite(c.ul) and
             finite(c.fl);
    };
    if (not map_ok(lmap) or not map_ok(rmap) or
        not causal_hll(&base.flux, lmap.ul, rmap.ul, lmap.fl, rmap.fl)) {
      result.status = "causal_map_unavailable";
      return result;
    }
    base.ul = lmap.ul;
    base.ur = rmap.ul;
    base.fl = lmap.fl;
    base.fr = rmap.fl;
    base.bounds = {-1., 1.};
    base.integers[0] = 1;
    result.causal_fallback = true;
  }
  if (base.status() != 1) {
    result.status = "native_unusable";
    return result;
  }
  result.left_tau = base.ul;
  result.right_tau = base.ur;
  result.bounds = base.bounds;
  State candidate = base.flux;
  const bool equal = left == right;
  if (equal) {
    candidate = base.fl;
    // Identical primitives: the flux is this state's physical flux and the
    // slow-mode response is identically zero (G = W_s (U_R - U_L) = 0). The
    // original path still built the two-state entropy model -- two 160-digit
    // recoveries -- and ran the (identity) repair before returning this same
    // flux, so the uniform transverse faces of a 1-D tube on SpECTRE's 3-D
    // slab cost as much as real faces (hours per step). Return it directly
    // when it is finite; a non-finite map falls through to the original path.
    // (2026-10-15, task 2's integration of the kernel.)
    if (finite(candidate)) {
      result.supported = true;
      result.flux = result.baseline = candidate;
      result.status = "equal_state_physical_flux";
      return result;
    }
  } else if (not result.causal_fallback and base.bounds[0] < 0. and
             base.bounds[1] > 0.) {
    const auto refined = core(1, left, right, bn, gamma, base.bounds);
    ++result.core_calls;
    result.baseline_outcome = refined.integers[5];
    if (refined.status() != 1) {
      result.status = "refined_core_unusable";
      return result;
    }
    candidate = refined.flux;
  }
  const EntropyModel model(base.ul, base.ur, bn, gamma);
  if (not model.supported) {
    result.status = model.reason;
    return result;
  }
  const auto repair = model.repair(candidate);
  if (not repair.supported) {
    result.status = repair.reason;
    return result;
  }
  result.supported = true;
  result.flux = result.baseline = repair.flux;
  const auto keep = [&result](const char* reason) {
    result.status = reason;
    return result;
  };
  if (equal) {
    return keep("equal_state_physical_flux");
  }
  if (result.causal_fallback) {
    return keep("causal_hll_baseline");
  }
  if (base.bounds[0] >= 0. or base.bounds[1] <= 0.) {
    return keep("whole_fan_upwind");
  }
  if (base.ul == base.ur) {
    return keep("zero_stored_jump");
  }
  const auto& pl = model.left.recovery.primitive;
  const auto& pr = model.right.recovery.primitive;
  const auto reference =
      reference_from_tau(base.ul, base.ur, pl, pr, bn, gamma);
  if (not reference.success) {
    return keep("reference_unsupported");
  }
  const auto slow =
      slow_action(reference.primitive, bn, gamma, base.ul, base.ur);
  if (not slow.supported) {
    return keep("analytic_modes_unsupported");
  }
  if (not finite(slow.slow_jump) or not finite(slow.absolute_slow_jump)) {
    return keep("nonfinite_slow_action");
  }
  if (std::all_of(slow.slow_jump.begin(), slow.slow_jump.end(),
                  [](double x) { return x == 0.; })) {
    return keep("zero_slow_action");
  }
  State shift = slow.slow_jump;
  for (auto& x : shift) {
    x /= 2.;
  }
  const auto input_cap = physical_ray_cap(base.ul, base.ur, shift, bn, gamma);
  result.beta = input_cap.beta;
  if (not input_cap.success or input_cap.beta == 0.) {
    return keep("input_ray_baseline");
  }
  if (input_cap.left == base.ul and input_cap.right == base.ur) {
    return keep("zero_represented_input_change");
  }
  const auto cl = recover(input_cap.left, bn, gamma);
  const auto cr = recover(input_cap.right, bn, gamma);
  if (not cl.success or not cr.success) {
    return keep("modified_recovery_unsupported");
  }
  if (cl.primitive == pl and cr.primitive == pr) {
    return keep("zero_represented_primitive_change");
  }
  const auto k0 = core(1, pl, pr, bn, gamma, base.bounds);
  const auto kc = core(1, cl.primitive, cr.primitive, bn, gamma, base.bounds);
  result.core_calls += 2;
  if (k0.status() != 1 or kc.status() != 1) {
    return keep("fixed_core_unusable");
  }
  for (const auto* encoded : {&k0.ul, &k0.ur, &kc.ul, &kc.ur}) {
    if (not admissibility(*encoded, bn, gamma).admissible) {
      return keep("helper_encoded_input_unsupported");
    }
  }
  for (size_t j = 0; j < 7; ++j) {
    const long double fc0 =
        (static_cast<long double>(k0.fl[j]) + k0.fr[j]) / 2.L;
    const long double fcc =
        (static_cast<long double>(kc.fl[j]) + kc.fr[j]) / 2.L;
    const long double v0 = static_cast<long double>(k0.flux[j]) - fc0;
    const long double vc = static_cast<long double>(kc.flux[j]) - fcc;
    result.direction[j] =
        static_cast<double>(vc - v0 -
                            static_cast<long double>(input_cap.beta) *
                                slow.absolute_slow_jump[j] / 2.L);
  }
  const auto output = model.cap(result.baseline, result.direction);
  if (not output.supported) {
    return keep("output_line_unsupported");
  }
  result.flux = output.flux;
  result.alpha = output.coefficient.convert_to<double>();
  result.active = result.flux != result.baseline;
  result.status = output.reason;
  return result;
}
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::d2r_detail

namespace {
thread_local PlutoD2RDiagnostics diagnostics{};
}

extern "C" void pluto_d2r_reset_diagnostics() { diagnostics = {}; }
extern "C" PlutoD2RDiagnostics pluto_d2r_diagnostics() { return diagnostics; }

extern "C" int pluto_d2r_flux(const int npts, const double* left,
                              const double* right, const double gamma,
                              double* flux, double* pressure) {
  using namespace grmhd::ValenciaDivClean::BoundaryCorrections::d2r_detail;
  if (npts < 0 or left == nullptr or right == nullptr or flux == nullptr or
      pressure == nullptr) {
    return 0;
  }
  int supported = 0;
  // Faces with bit-identical inputs inside one batch share a single solve: on
  // SpECTRE's 3-D slab a 1-D tube presents every x-face 25 times (the 5 x 5
  // transverse points of a 3-point element), and the kernel is deterministic,
  // so the copies are exact. Keyed on the bit patterns of gamma and the two
  // 8-component states; NaN inputs never reach the map. (2026-10-15)
  struct Seen {
    int index;
    bool supported;
    bool active;
    bool causal_fallback;
  };
  std::map<std::array<std::uint64_t, 17>, Seen> seen;
  for (int k = 0; k < npts; ++k) {
    ++diagnostics.faces;
    auto* out = flux + 8 * k;
    const auto* l = left + 8 * k;
    const auto* r = right + 8 * k;
    std::fill_n(out, 8, std::numeric_limits<double>::quiet_NaN());
    pressure[k] = std::numeric_limits<double>::quiet_NaN();
    if (not std::isfinite(l[4]) or l[4] != r[4]) {
      ++diagnostics.unsupported;
      continue;
    }
    std::array<std::uint64_t, 17> key{};
    std::memcpy(key.data(), &gamma, sizeof(double));
    std::memcpy(key.data() + 1, l, 8 * sizeof(double));
    std::memcpy(key.data() + 9, r, 8 * sizeof(double));
    if (const auto it = seen.find(key); it != seen.end()) {
      const Seen& first = it->second;
      std::copy_n(flux + 8 * first.index, 8, out);
      pressure[k] = pressure[first.index];
      diagnostics.causal_fallback += first.causal_fallback;
      if (first.supported) {
        ++supported;
        ++diagnostics.supported;
        diagnostics.active += first.active;
      } else {
        ++diagnostics.unsupported;
      }
      continue;
    }
    try {
      const State pl{l[0], l[1], l[2], l[3], l[7], l[5], l[6]};
      const State pr{r[0], r[1], r[2], r[3], r[7], r[5], r[6]};
      const auto face = solve_face(pl, pr, l[4], gamma);
      diagnostics.causal_fallback += face.causal_fallback;
      if (not face.supported or not finite(face.flux)) {
        ++diagnostics.unsupported;
        seen.emplace(key, Seen{k, false, false, face.causal_fallback});
        continue;
      }
      seen.emplace(key, Seen{k, true, face.active, face.causal_fallback});
      out[0] = face.flux[0];
      out[1] = face.flux[1];
      out[2] = face.flux[2];
      out[3] = face.flux[3];
      out[4] = 0.;
      out[5] = face.flux[5];
      out[6] = face.flux[6];
      out[7] = face.flux[4];
      pressure[k] = 0.;
      ++supported;
      ++diagnostics.supported;
      diagnostics.active += face.active;
    } catch (const std::exception&) {
      // A numerical-domain failure must select the outer HLL fallback rather
      // than escape through the C function-pointer interface.
      ++diagnostics.unsupported;
    }
  }
  return supported;
}
