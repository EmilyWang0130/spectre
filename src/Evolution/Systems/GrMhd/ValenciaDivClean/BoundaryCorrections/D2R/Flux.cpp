// Distributed under the MIT License.
// See LICENSE.txt for details.
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/D2R/Flux.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <vector>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/D2R/SlowModes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/D2R/State.hpp"

extern "C" {
#include "spectre_d2r_core.h"
}

namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE {
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
double state_norm(const State& a) {
  double s = 0.;
  for (const double x : a) {
    s += x * x;
  }
  return std::sqrt(s);
}

// SPECTRE_D2R_CONSISTENCY (unset = 2, 0 = off): the consistency bound. A
// Riemann flux must return to the physical flux as the two states converge,
// and no part of a slow-mode correction can legitimately exceed the largest
// signal speed times the jump, |F - F_native| <= C s_max |U_R - U_L|. Two
// pieces of the kernel violated this on nearly uniform pairs (2026-10-19,
// run07_pluto_d2r/sensitivity): the entropy repair, whose coefficient divides
// by the pair's entropy jump and moved the baseline by 1e-5..1e-4 of the flux
// for jumps of 2e-10 (the source of the transverse noise of the 1-D cold tube
// on the 3-D slab); and the response, whose slow projection amplifies the
// jump up to 50x where the slow and entropy speeds nearly coincide (beta ~
// 1e-3), reaching 23 s_max |dU|. On the relativistic tube, where the method
// gains, the response stays below 0.93 s_max |dU| and the repair below 4e-7
// of the flux, so C = 2 leaves that unchanged. A repair beyond the bound
// leaves the face unsupported (the shim keeps its HLL flux, which is
// consistent); a response beyond it is scaled down to it, which keeps the
// entropy condition because the margin is affine along the ray.
const double consistency_factor = [] {
  const char* e = std::getenv("SPECTRE_D2R_CONSISTENCY");
  return e == nullptr ? 2.0 : std::atof(e);
}();

// SPECTRE_D2R_JUMP_FLOOR (unset or 0 = off): respond only to resolved jumps.
// At low beta the slow-mode amplitude is read from the pressure jump divided
// by c_s^2, and the response is a small residual of two O(Q) terms, so the
// response of a face moves by ~1e7 times a relative change of its inputs
// (relative to the jump's flux) where c_s^2 ~ 1e-4: faces whose states differ
// by round-off respond differently at the 1e-3 level (2026-10-19,
// run07_pluto_d2r/sensitivity, harness14). The weight
// min(1, (|U_R - U_L| / (eta |U|))^2) switches the response off where the
// cell-to-cell jump is below eta of the state -- the regime in which that
// sensitivity acts on round-off -- and leaves shocks untouched.
const double jump_floor = [] {
  const char* e = std::getenv("SPECTRE_D2R_JUMP_FLOOR");
  return e == nullptr ? 0.0 : std::atof(e);
}();
}  // namespace

FaceResult solve_face(const State& left, const State& right, const double bn,
                      const double gamma) {
#ifdef D2R_MULTIPRECISION
  // The 160-digit oracle stays serialized, as the first correctness embedding
  // was: it is a test backend, and Boost's lazily initialized multiprecision
  // constants have not been audited for concurrent first use.
  static std::mutex kernel_mutex;
  const std::lock_guard<std::mutex> guard(kernel_mutex);
#endif
  // The double kernel runs unlocked on every worker (2026-10-20). Its own
  // state is per call (the memo is per batch, stats and diagnostics are
  // thread_local, the switches are constants set before main). The C core it
  // calls keeps its grid, contexts and every PLUTO scratch array in
  // _Thread_local storage and initializes them on each thread's first call
  // (spectre_d2r_core.c core_setup, hlld.c, d2r_hlld_fixed.c, arrays.c,
  // hll_speed.c, mappers.c); the process globals that path still writes
  // (g_maxMach, g_usedMemory, g_maxRootIter) are diagnostics nothing reads.
  // That is the threading contract of the legacy PLUTO solvers, which call
  // the same HLL_Speed/Magnetosonic/HLLD_Solver from every worker unlocked.
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
  State jump_state{};
  for (size_t j = 0; j < 7; ++j) {
    jump_state[j] = base.ur[j] - base.ul[j];
  }
  const double consistency_scale =
      consistency_factor *
      std::max(std::abs(base.bounds[0]), std::abs(base.bounds[1])) *
      state_norm(jump_state);
  if (consistency_factor > 0.) {
    State moved{};
    for (size_t j = 0; j < 7; ++j) {
      moved[j] = repair.flux[j] - candidate[j];
    }
    if (state_norm(moved) > consistency_scale) {
      result.status = "repair_exceeds_consistency_bound";
      return result;
    }
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
  // The cap recovered exactly these states to accept them (2026-10-20: the
  // same two recoveries were repeated here).
  const auto& cl = input_cap.left_recovery;
  const auto& cr = input_cap.right_recovery;
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
  // The response T = V_c - V_0 compares two fixed-bound solves that must come
  // from the same branch of the helper (an HLLD fan, or its HLL fallback): if
  // the original and the modified inputs land on different branches, T carries
  // the O(dU) HLLD-HLL difference instead of an O(beta G) slow response and
  // only the entropy cap would bound it. Keep the baseline then. (Review of
  // 2026-10-15; integers[5] is the helper's outcome code, as for
  // baseline_outcome above.)
  if (k0.integers[5] != kc.integers[5] or
      (result.baseline_outcome >= 0 and
       k0.integers[5] != result.baseline_outcome)) {
    return keep("baseline_outcome_mismatch");
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
  if (jump_floor > 0.) {
    State mean_state{};
    for (size_t j = 0; j < 7; ++j) {
      mean_state[j] = 0.5 * (base.ul[j] + base.ur[j]);
    }
    const double resolved =
        state_norm(jump_state) / (jump_floor * state_norm(mean_state));
    if (resolved < 1.) {
      result.jump_weight = resolved * resolved;
      for (auto& x : result.direction) {
        x *= result.jump_weight;
      }
    }
  }
  if (consistency_factor > 0.) {
    const double response_norm = state_norm(result.direction);
    if (response_norm > consistency_scale) {
      result.consistency = consistency_scale / response_norm;
      for (auto& x : result.direction) {
        x *= result.consistency;
      }
    }
  }
  const auto output = model.cap(result.baseline, result.direction);
  if (not output.supported) {
    return keep("output_line_unsupported");
  }
  result.flux = output.flux;
  result.alpha = to_double(output.coefficient);
  result.active = result.flux != result.baseline;
  result.status = output.reason;
  return result;
}
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE

namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE {
namespace {
thread_local PlutoD2RDiagnostics diagnostics{};

// Cost statistics, printed to stderr every SPECTRE_D2R_STATS calls of
// pluto_d2r_flux (environment variable; unset or 0 = silent). Diagnostic only;
// no flux depends on it. (2026-10-15, task 2's integration tests.)
struct Stats {
  long calls = 0, points = 0, solves = 0, memo_hits = 0, equal = 0, active = 0,
       unsupported = 0, near_equal = 0, near_memo = 0, consistency_capped = 0,
       repair_bound = 0, jump_damped = 0;
  double kernel_seconds = 0., max_face_seconds = 0.;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
};
thread_local Stats stats{};
const long stats_every = [] {
  const char* e = std::getenv("SPECTRE_D2R_STATS");
  return e == nullptr ? 0L : std::atol(e);
}();

// SPECTRE_D2R_ROUNDOFF_TOL (unset or 0 = off): a face whose two states differ
// by no more than this relative amount in every component gets the native
// HLLD flux of the pair (one C call) and no slow-mode response. SpECTRE's FD
// slab is transversely uniform only to ~1e-12 (HLLD and HLLx never notice),
// so without this every y/z face of a "1-D" tube is a distinct 0.15 s solve:
// ~60x the x-faces. A response to a 1e-12 jump is below round-off anyway.
// Scales: rho and p by themselves, velocities by c = 1, the field by |B|.
// (2026-10-15, task 2's integration tests.)
const double roundoff_tol = [] {
  const char* e = std::getenv("SPECTRE_D2R_ROUNDOFF_TOL");
  return e == nullptr ? 0.0 : std::atof(e);
}();

// Two faces (lA, rA) and (lB, rB) whose inputs agree component by component
// to within tol (same scales as near_equal, taken from face A) are the same
// face at the round-off level: the second reuses the first's flux. Used by
// the per-batch memo when SPECTRE_D2R_ROUNDOFF_TOL is set, so the 25
// transverse copies of a 1-D tube's x-face, which SpECTRE keeps equal only to
// ~1e-12, are one solve instead of 25.
bool near_inputs(const double* lA, const double* rA, const double* lB,
                 const double* rB, const double tol) {
  const double b_scale =
      std::max(std::sqrt(lA[4] * lA[4] + lA[5] * lA[5] + lA[6] * lA[6]),
               std::sqrt(rA[4] * rA[4] + rA[5] * rA[5] + rA[6] * rA[6]));
  const std::array<double, 8> scale{
      std::max(std::abs(lA[0]), std::abs(rA[0])), 1., 1., 1., b_scale,
      b_scale, b_scale, std::max(std::abs(lA[7]), std::abs(rA[7]))};
  for (int i = 0; i < 8; ++i) {
    if (std::abs(lA[i] - lB[i]) > tol * scale[i] or
        std::abs(rA[i] - rB[i]) > tol * scale[i]) {
      return false;
    }
  }
  return true;
}

bool near_equal(const double* l, const double* r, const double tol) {
  // P = (rho, vn, vt, vb, Bn, Bt, Bb, p)
  const double b_scale =
      std::max(std::sqrt(l[4] * l[4] + l[5] * l[5] + l[6] * l[6]),
               std::sqrt(r[4] * r[4] + r[5] * r[5] + r[6] * r[6]));
  const std::array<double, 8> scale{std::max(std::abs(l[0]), std::abs(r[0])),
                                    1., 1., 1., b_scale, b_scale, b_scale,
                                    std::max(std::abs(l[7]), std::abs(r[7]))};
  for (int i = 0; i < 8; ++i) {
    if (std::abs(l[i] - r[i]) > tol * scale[i]) {
      return false;
    }
  }
  return true;
}
}  // namespace

void reset_diagnostics_impl() { diagnostics = {}; }
PlutoD2RDiagnostics diagnostics_impl() { return diagnostics; }

int pluto_d2r_flux_impl(const int npts, const double* left,
                        const double* right, const double gamma, double* flux,
                        double* pressure) {
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
  // Solved points of this batch, for the round-off-tolerance memo.
  std::vector<Seen> representatives;
  ++stats.calls;
  stats.points += npts;
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
    const auto reuse = [&](const Seen& first) {
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
    };
    if (const auto it = seen.find(key); it != seen.end()) {
      ++stats.memo_hits;
      reuse(it->second);
      continue;
    }
    if (roundoff_tol > 0.) {
      bool reused = false;
      for (const Seen& rep : representatives) {
        if (near_inputs(left + 8 * rep.index, right + 8 * rep.index, l, r,
                        roundoff_tol)) {
          ++stats.near_memo;
          reuse(rep);
          reused = true;
          break;
        }
      }
      if (reused) {
        continue;
      }
    }
    try {
      const State pl{l[0], l[1], l[2], l[3], l[7], l[5], l[6]};
      const State pr{r[0], r[1], r[2], r[3], r[7], r[5], r[6]};
      if (roundoff_tol > 0. and pl != pr and near_equal(l, r, roundoff_tol)) {
        // Round-off-level jump: the baseline of the pair, no response. The
        // baseline is the SAME solve as solve_face's candidate -- the
        // fixed-bound solve with the pair's native bounds, whose root is
        // refined to adjacent doubles -- not native hlld.c, whose pressure
        // iteration stops at 1e-5 of p: with the native flux here, a face
        // crossing the tolerance jumped by 1e-5 of the flux, which broke the
        // transverse symmetry of the 1-D cold tube on the 3-D slab at t = 1.7
        // (2026-10-19, run07_pluto_d2r/sensitivity, iteration 3). Falls
        // through to the full path if the core calls fail.
        const auto base = core(0, pl, pr, l[4], gamma);
        State flux = base.flux;
        if (base.status() == 1 and base.bounds[0] < 0. and
            base.bounds[1] > 0.) {
          const auto refined = core(1, pl, pr, l[4], gamma, base.bounds);
          if (refined.status() == 1 and finite(refined.flux)) {
            flux = refined.flux;
          }
        }
        if (base.status() == 1 and finite(flux)) {
          out[0] = flux[0];
          out[1] = flux[1];
          out[2] = flux[2];
          out[3] = flux[3];
          out[4] = 0.;
          out[5] = flux[5];
          out[6] = flux[6];
          out[7] = flux[4];
          pressure[k] = 0.;
          ++supported;
          ++diagnostics.supported;
          ++stats.near_equal;
          seen.emplace(key, Seen{k, true, false, false});
          representatives.push_back(Seen{k, true, false, false});
          continue;
        }
      }
      const auto t0 = std::chrono::steady_clock::now();
      const auto face = solve_face(pl, pr, l[4], gamma);
      const double dt = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      ++stats.solves;
      stats.kernel_seconds += dt;
      stats.max_face_seconds = std::max(stats.max_face_seconds, dt);
      stats.equal += (pl == pr);
      stats.active += face.active;
      stats.unsupported += not face.supported;
      stats.consistency_capped += face.consistency < 1.;
      stats.repair_bound += face.status == "repair_exceeds_consistency_bound";
      stats.jump_damped += face.jump_weight < 1.;
      diagnostics.causal_fallback += face.causal_fallback;
      if (not face.supported or not finite(face.flux)) {
        ++diagnostics.unsupported;
        seen.emplace(key, Seen{k, false, false, face.causal_fallback});
        representatives.push_back(Seen{k, false, false, face.causal_fallback});
        continue;
      }
      seen.emplace(key, Seen{k, true, face.active, face.causal_fallback});
      representatives.push_back(
          Seen{k, true, face.active, face.causal_fallback});
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
    } catch (...) {
      ++diagnostics.unsupported;
    }
  }
  if (stats_every > 0 and stats.calls % stats_every == 0) {
    const double wall = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - stats.start)
                            .count();
    std::fprintf(
        stderr,
        "D2R stats [%s]: %ld calls, %ld points, %ld solves (%ld equal, "
        "%ld active, %ld unsupported), %ld near-equal (tol %g), "
        "%ld memo hits + %ld near-memo, %ld consistency-capped + "
        "%ld repair-bound (C %g), %ld jump-damped (eta %g); kernel "
        "%.1f s of %.1f s wall, max face %.3f s\n",
        multiprecision_build ? "160-digit" : "double", stats.calls,
        stats.points, stats.solves, stats.equal, stats.active,
        stats.unsupported, stats.near_equal, roundoff_tol, stats.memo_hits,
        stats.near_memo, stats.consistency_capped, stats.repair_bound,
        consistency_factor, stats.jump_damped, jump_floor, stats.kernel_seconds,
        wall, stats.max_face_seconds);
    std::fflush(stderr);
  }
  return supported;
}
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE

#ifndef D2R_MULTIPRECISION
// The extern "C" interface the SpECTRE adapter and the tests call, defined once
// (in the double build) and dispatched by SPECTRE_D2R_PRECISION: unset or
// "double" selects the production kernel, "mp" the 160-digit oracle that the
// *Mp.cpp wrappers compile from the same sources. (2026-10-19)
namespace {
const bool use_multiprecision = [] {
  const char* e = std::getenv("SPECTRE_D2R_PRECISION");
  return e != nullptr and (std::strcmp(e, "mp") == 0 or
                           std::strcmp(e, "160") == 0 or
                           std::strcmp(e, "multiprecision") == 0);
}();
}  // namespace

extern "C" void pluto_d2r_reset_diagnostics() {
  if (use_multiprecision) {
    grmhd::ValenciaDivClean::BoundaryCorrections::d2r_detail_mp::
        reset_diagnostics_impl();
  } else {
    grmhd::ValenciaDivClean::BoundaryCorrections::d2r_detail::
        reset_diagnostics_impl();
  }
}
extern "C" PlutoD2RDiagnostics pluto_d2r_diagnostics() {
  return use_multiprecision
             ? grmhd::ValenciaDivClean::BoundaryCorrections::d2r_detail_mp::
                   diagnostics_impl()
             : grmhd::ValenciaDivClean::BoundaryCorrections::d2r_detail::
                   diagnostics_impl();
}
extern "C" int pluto_d2r_flux(const int npts, const double* left,
                              const double* right, const double gamma,
                              double* flux, double* pressure) {
  return use_multiprecision
             ? grmhd::ValenciaDivClean::BoundaryCorrections::d2r_detail_mp::
                   pluto_d2r_flux_impl(npts, left, right, gamma, flux,
                                       pressure)
             : grmhd::ValenciaDivClean::BoundaryCorrections::d2r_detail::
                   pluto_d2r_flux_impl(npts, left, right, gamma, flux,
                                       pressure);
}
#endif
