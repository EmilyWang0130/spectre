// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/D2R/SlowModes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>


namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE {
namespace {

constexpr size_t FastMinus = 0;
constexpr size_t AlfvenMinus = 1;
constexpr size_t SlowMinus = 2;
constexpr size_t SlowPlus = 4;
constexpr size_t AlfvenPlus = 5;
constexpr size_t FastPlus = 6;

Real square(const Real& value) { return value * value; }
Real cube(const Real& value) { return value * value * value; }
Real magnitude(const Real& value) { return abs_real(value); }
bool finite(const Real& value) { return finite_real(value); }

Real dot(const MpState& a, const MpState& b) {
  Real result = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

struct PhysicalState {
  Real rho;
  std::array<Real, 3> velocity;
  Real pressure;
  std::array<Real, 3> magnetic;
  Real lorentz;
  Real enthalpy;
  Real sound_squared;
  Real magnetic_squared;
  Real magnetic_dot_velocity;
  Real comoving_magnetic_squared;
  Real gamma;
};

PhysicalState physical_state(const State& primitive, const double bn,
                             const double gamma) {
  PhysicalState state{};
  state.rho = exact_double(primitive[0]);
  state.pressure = exact_double(primitive[4]);
  state.gamma = exact_double(gamma);
  Real velocity_squared = 0;
  state.magnetic_squared = 0;
  state.magnetic_dot_velocity = 0;
  for (size_t i = 0; i < 3; ++i) {
    state.velocity[i] = exact_double(primitive[i + 1]);
    state.magnetic[i] = exact_double(i == 0 ? bn : primitive[i + 4]);
    velocity_squared += square(state.velocity[i]);
    state.magnetic_squared += square(state.magnetic[i]);
    state.magnetic_dot_velocity += state.magnetic[i] * state.velocity[i];
  }
  if (not(state.rho > 0 and state.pressure > 0 and state.gamma > 1 and
          state.gamma <= 2 and velocity_squared < 1)) {
    throw std::domain_error("positive_causal_Gamma_law_primitive_required");
  }
  state.lorentz = 1 / sqrt(1 - velocity_squared);
  state.enthalpy =
      1 + state.gamma / (state.gamma - 1) * state.pressure / state.rho;
  state.sound_squared =
      state.gamma * state.pressure / (state.rho * state.enthalpy);
  if (not(state.sound_squared > 0 and state.sound_squared < 1)) {
    throw std::domain_error("strictly_causal_positive_sound_required");
  }
  state.comoving_magnetic_squared =
      state.magnetic_squared / square(state.lorentz) +
      square(state.magnetic_dot_velocity);
  return state;
}

// Bisection to the declared160-digit relative bracket precision. cpp_dec_float
// has extra guard limbs, so equality of repeatedly computed midpoints alone is
// not a reliable termination rule. No physical gap threshold is introduced.
// The resource ceiling only identifies unsupported numerical arithmetic.
//
// An unresolved family bracket (both ends of one sign) is an expected outcome
// on evolved faces (a few per cent of a jet's), so it is returned, not thrown
// (2026-10-20): false, the root untouched. Every other failure still throws.
//
// Design D2 (2026-10-21): Newton's method on the quartic with its analytic
// slope, safeguarded by the bracket (a bisection step whenever the Newton step
// would leave it or not halve the step before), to the same relative precision
// as the bisection it replaces: about 6-10 evaluations instead of ~52
// (binary64) or ~530 (160 digits). The root differs from the bisection's in the
// last bits.
template <typename Function, typename Slope>
bool bracketed_root(const Function& function, const Slope& slope, Real left,
                    Real right, Real* const root) {
  Real fleft = function(left);
  Real fright = function(right);
  if (not(finite(fleft) and finite(fright))) {
    throw std::runtime_error("nonfinite_quartic_endpoint");
  }
  if (fleft == 0) {
    *root = left;
    return true;
  }
  if (fright == 0) {
    *root = right;
    return true;
  }
  if ((fleft < 0) == (fright < 0)) {
    return false;
  }
  // The bracket oriented by sign: function(lower) < 0 < function(upper).
  Real lower = fleft < 0 ? left : right;
  Real upper = fleft < 0 ? right : left;
  Real x = half(left + right);
  Real step_before = magnitude(right - left);
  Real step = step_before;
  Real f = function(x);
  for (size_t iteration = 0; iteration < 4096; ++iteration) {
    if (not finite(f)) {
      throw std::runtime_error("nonfinite_quartic_evaluation");
    }
    if (f == 0) {
      *root = x;
      return true;
    }
    if (f < 0) {
      lower = x;
    } else {
      upper = x;
    }
    const Real df = slope(x);
    const bool newton = finite(df) and df != 0 and
                        ((x - upper) * df - f) * ((x - lower) * df - f) < 0 and
                        magnitude(2 * f) <= magnitude(step_before * df);
    step_before = step;
    if (newton) {
      step = f / df;
      x -= step;
    } else {
      step = half(upper - lower);
      x = lower + step;
    }
    if (magnitude(step) <=
        std::numeric_limits<Real>::epsilon() * magnitude(x)) {
      *root = x;
      return true;
    }
    f = function(x);
  }
  throw std::runtime_error("quartic_bisection_resource_limit");
}

struct Spectrum {
  std::array<Real, 7> lambda{};
  std::array<Real, 7> gap{};
  bool exact_slow_alfven_collision = false;
  // The expected unsupported outcomes, returned rather than thrown: null, or
  // the status slow_action reports.
  const char* failure = nullptr;
};

Spectrum physical_spectrum(const PhysicalState& state) {
  const Real& vn = state.velocity[0];
  const Real& bn = state.magnetic[0];
  const Real H = state.rho * state.enthalpy;
  Spectrum result{};
  if (state.magnetic[1] == 0 and state.magnetic[2] == 0 and
      state.velocity[1] == 0 and state.velocity[2] == 0) {
    // Exactly parallel normal flow factorizes into acoustic and Alfven roots.
    // Do not infer a false separation by subtracting nearly equal quartic
    // terms.
    const Real acoustic = sqrt(state.sound_squared);
    const Real alfven = magnitude(bn) / sqrt(H + square(bn));
    const Real slow = std::min(acoustic, alfven);
    const Real fast = std::max(acoustic, alfven);
    const auto boost = [&vn](const Real& speed) -> Real {
      return (vn + speed) / (1 + vn * speed);
    };
    result.lambda = {boost(-fast), boost(-alfven), boost(-slow), vn,
                     boost(slow),  boost(alfven),  boost(fast)};
    for (size_t k = 0; k < 7; ++k) {
      result.gap[k] = result.lambda[k] - vn;
    }
    result.exact_slow_alfven_collision = acoustic >= alfven;
    return result;
  }
  const Real rootH = sqrt(H + state.comoving_magnetic_squared);
  std::array<Real, 2> alfven_gap{
      bn / (square(state.lorentz) * (state.magnetic_dot_velocity + rootH)),
      bn / (square(state.lorentz) * (state.magnetic_dot_velocity - rootH))};
  std::sort(alfven_gap.begin(), alfven_gap.end());
  const auto quartic = [&state, &H, &vn, &bn](const Real& gap) -> Real {
    const Real aa = square(state.lorentz * gap);
    const Real G = 1 - square(vn + gap);
    const Real Bq =
        bn / state.lorentz - state.lorentz * state.magnetic_dot_velocity * gap;
    return aa * G * state.comoving_magnetic_squared - H * square(aa) +
           state.sound_squared * (H * aa * (aa + G) - square(Bq) * G);
  };
  // d quartic / d gap, for the Newton steps of the root search.
  const auto quartic_slope = [&state, &H, &vn, &bn](const Real& gap) -> Real {
    const Real aa = square(state.lorentz * gap);
    const Real G = 1 - square(vn + gap);
    const Real Bq =
        bn / state.lorentz - state.lorentz * state.magnetic_dot_velocity * gap;
    const Real daa = 2 * square(state.lorentz) * gap;
    const Real dG = -2 * (vn + gap);
    const Real dBq = -state.lorentz * state.magnetic_dot_velocity;
    return state.comoving_magnetic_squared * (daa * G + aa * dG) -
           2 * H * aa * daa +
           state.sound_squared * (H * (daa * (aa + G) + aa * (daa + dG)) -
                                  2 * Bq * dBq * G - square(Bq) * dG);
  };
  // A quartic root at an Alfven bracket endpoint need not be a DOUBLE
  // magnetosonic root. Returning it from both neighboring intervals would
  // silently mislabel a fast/slow family. The exactly parallel factorization
  // above handles its known collisions; other zero endpoint evaluations are
  // an explicit numerical support gap, without a fitted near-gap threshold.
  if (quartic(alfven_gap[0]) == 0 or quartic(alfven_gap[1]) == 0) {
    result.failure = "nonparallel_quartic_boundary_root_unsupported";
    return result;
  }
  const Real zero = 0;
  // The four families in the order of the original braced list, stopping at
  // the first unresolved bracket as the exception did.
  const std::array<std::array<Real, 2>, 4> brackets{{{-1 - vn, alfven_gap[0]},
                                                     {alfven_gap[0], zero},
                                                     {zero, alfven_gap[1]},
                                                     {alfven_gap[1], 1 - vn}}};
  std::array<Real, 4> roots{};
  for (size_t j = 0; j < 4; ++j) {
    if (not bracketed_root(quartic, quartic_slope, brackets[j][0],
                           brackets[j][1], &roots[j])) {
      result.failure = "quartic_family_bracket_unresolved";
      return result;
    }
  }
  result.gap = {roots[0], alfven_gap[0], roots[1], zero,
                roots[2], alfven_gap[1], roots[3]};
  for (size_t k = 0; k < 7; ++k) {
    result.lambda[k] = vn + result.gap[k];
    if (not(result.lambda[k] > -1 and result.lambda[k] < 1 and
            finite(result.lambda[k]))) {
      throw std::runtime_error("causal_family_speed_unresolved");
    }
    if (k > 0 and result.lambda[k] < result.lambda[k - 1]) {
      throw std::runtime_error("physical_family_order_unresolved");
    }
  }
  return result;
}

struct FilterWeights {
  std::array<Real, 2> weights{};
  bool bound_active = false;
};

// The family-filter weight. Default since 2026-10-19 (the user's decision on
// the exact-reference tubes at 80 and 160 cells): "roe-balsara",
//   w = (b_A/f^2) alpha_s^2,  alpha_s^2 = (f^2 - c_s^2)/(f^2 - s^2),
// the standard slow-magnetosonic normalization factor (1/2 at the umbilic by
// continuity). SPECTRE_D2R_WEIGHT selects the alternatives:
//   "polynomial" (the audited weight of the Newtonian design):
//                  (b_A/f^2) (1 - c_s^2/f^2)^2 (1 - s^2/f^2). On the warm
//                  relativistic large-B_x tube (c_s^2/f^2 = 0.17) it is 0.54
//                  where the cold tubes give 0.998, and D2-R recovers only half
//                  of the seven-wave gain there (task 2, 2026-10-15).
//   "linear":      (b_A/f^2) (1 - c_s^2/f^2) = (mu_s^2/c_s^2)(1 - c_s^2/mu_f^2):
//                  the same zeros as the audited weight (B_n -> 0, slow = Alfven
//                  at c_s^2 = f^2, the umbilic), each simple; ties HLLx.
// All three use s^2 f^2 = c_s^2 b_A (the rest-frame dispersion relation) to
// write b_A/f^2 as mu_s^2/c_s^2. Per side: mu_s and mu_f of that side.
// Relativistic large-B_x tube against its exact solution (L1 of rho, % of the
// exact range): 80 cells HLLD 9.81, polynomial 8.22, HLLx 6.72, linear 7.17,
// roe-balsara 6.43; 160 cells 7.02 / 5.95 / 4.91 / 5.26 / 4.73.
enum class WeightLaw { Polynomial, Linear, RoeBalsara };
const WeightLaw weight_law = [] {
  const char* e = std::getenv("SPECTRE_D2R_WEIGHT");
  if (e == nullptr) {
    return WeightLaw::RoeBalsara;
  }
  const std::string law(e);
  if (law == "polynomial" or law == "audited") {
    return WeightLaw::Polynomial;
  }
  if (law == "linear") {
    return WeightLaw::Linear;
  }
  return WeightLaw::RoeBalsara;
}();

FilterWeights family_weights(const PhysicalState& state,
                             const Spectrum& spectrum) {
  std::array<Real, 7> mu{};
  for (size_t k = 0; k < 7; ++k) {
    const Real rest_frequency = state.lorentz * spectrum.gap[k];
    mu[k] = rest_frequency /
            sqrt(1 - square(spectrum.lambda[k]) + square(rest_frequency));
  }
  if (weight_law != WeightLaw::Polynomial) {
    FilterWeights result{};
    const Real& cs2 = state.sound_squared;
    const std::array<std::pair<size_t, size_t>, 2> sides{
        std::pair<size_t, size_t>{SlowMinus, FastMinus},
        std::pair<size_t, size_t>{SlowPlus, FastPlus}};
    for (size_t j = 0; j < 2; ++j) {
      const Real s2 = square(mu[sides[j].first]);
      const Real f2 = square(mu[sides[j].second]);
      Real w = (s2 / cs2) * (weight_law == WeightLaw::Linear
                                 ? (1 - cs2 / f2)
                                 : (f2 - cs2) / (f2 - s2));
      if (not finite(w)) {
        throw std::runtime_error("filter_weight_arithmetic_unsupported");
      }
      // The bounds hold analytically; round-off may overshoot them by an ulp.
      result.weights[j] = std::min(std::max(w, Real(0)), Real(1));
    }
    return result;
  }
  const auto polynomial = [&mu](const Real& z) -> Real {
    return square(z) * square(z - mu[AlfvenMinus]) *
           square(z - mu[AlfvenPlus]) * (z - mu[FastMinus]) *
           (mu[FastPlus] - z);
  };
  const Real N0 = state.sound_squared *
                  square(-mu[AlfvenMinus] * mu[AlfvenPlus]) *
                  (-mu[FastMinus] * mu[FastPlus]);
  const std::array<Real, 2> raw{polynomial(mu[SlowMinus]),
                                polynomial(mu[SlowPlus])};
  const Real normalizer = std::max(N0, std::max(raw[0], raw[1]));
  if (not(normalizer > 0 and finite(normalizer))) {
    throw std::runtime_error("filter_normalizer_unresolved");
  }
  FilterWeights result{};
  result.bound_active = normalizer > N0;
  for (size_t j = 0; j < 2; ++j) {
    result.weights[j] = raw[j] / normalizer;
    if (not(result.weights[j] >= 0 and result.weights[j] <= 1 and
            finite(result.weights[j]))) {
      throw std::runtime_error("filter_weight_arithmetic_unsupported");
    }
  }
  return result;
}

// Flat-metric fixed-Bn restriction of characteristic_eigenvectors_mhd's
// analytic magnetosonic formulas (Teukolsky Paper II conserved L/R vectors).
// No individual Euclidean norm is treated as the biorthogonal normalization.
std::pair<MpState, MpState> analytic_pair(const PhysicalState& state,
                                          const Real& y, const Real& gap) {
  const Real& rho = state.rho;
  const Real& W = state.lorentz;
  const Real& h = state.enthalpy;
  const Real& cs2 = state.sound_squared;
  const auto& v = state.velocity;
  const auto& B = state.magnetic;
  const Real& Bv = state.magnetic_dot_velocity;
  const Real& bn = B[0];
  const Real& vn = v[0];
  const Real H = rho * h;
  const Real a = -W * gap;
  const Real G = 1 - square(y);
  const Real Bq = bn / W + Bv * a;
  const Real scriptG = H * square(a) - G * state.comoving_magnetic_squared;
  if (a == 0 or G == 0 or cs2 == 0) {
    throw std::runtime_error("exact_analytic_denominator_zero");
  }
  const Real hstar = h + state.comoving_magnetic_squared / rho;
  const Real Z = H * square(W);
  const Real kappa = (state.gamma - 1) * rho;
  const Real m1s =
      H * a * W * (Bq * Bv - (H + state.comoving_magnetic_squared) * a);
  const Real m1v = H * (bn * Bv * (y * a + 2 * G * W) - 2 * a * square(bn) -
                        a * W *
                            (y * a * (state.magnetic_squared / square(W) + H) +
                             (1 - 1 / cs2) * scriptG * W));
  const Real m1B = H * (Bq * (y * a - G * W) + 2 * bn * (square(a) + G)) / W;
  const Real m4 = rho / a *
                  (square(a * Bq) * h - 2 * square(bn) * h * (square(a) + G) +
                   square(Bq) * W * (2 * y * a * h - G) +
                   bn * Bq * (G + 2 * h * W * G - 2 * y * a * h) +
                   scriptG * W * square(a) * (h * W * (1 - cs2) - 1) / cs2 +
                   H * cube(a) * (y - 2 * y * hstar * W + a * (W - hstar)));
  MpState right{};
  right[0] = -rho * Bq * G * bn / a - square(rho) * a * h * (y * a - G * W);
  right[4] = m4;
  for (size_t i = 0; i < 3; ++i) {
    const Real n = i == 0 ? Real{1} : Real{0};
    right[i + 1] = m1s * n + m1v * v[i] + m1B * B[i];
    if (i > 0) {
      right[i + 4] = H * a * (B[i] * (1 - y * vn) - bn * (n - y * v[i]));
    }
  }
  const Real f1v = W * (-G + Bq * G * bn * W / (Z * square(a)) +
                        scriptG * square(W) * (kappa + rho) / (Z * rho * cs2));
  const Real g1B = scriptG * kappa * W / (rho * Z * cs2) - (square(a) + G) / W;
  const Real g1v =
      Bv * square(W) * g1B + W * (a * Bq + scriptG * Bq * square(W) / (Z * a));
  const Real h1 = -(f1v + y * a);
  MpState left{};
  left[0] = h1 + scriptG * (kappa - rho * cs2) / (square(rho) * cs2);
  left[4] = h1;
  for (size_t i = 0; i < 3; ++i) {
    const Real n = i == 0 ? Real{1} : Real{0};
    left[i + 1] = a * n - Bq * G * W * B[i] / (Z * a) + f1v * v[i];
    if (i > 0) {
      left[i + 4] =
          Bq * n + g1B * B[i] + g1v * v[i] -
          (scriptG / (H * cs2) * (kappa + rho * cs2) * Bq / (G * rho)) * n;
    }
  }
  Real right_scale = 0;
  Real left_scale = 0;
  for (size_t i = 0; i < 7; ++i) {
    right_scale = std::max(right_scale, magnitude(right[i]));
    left_scale = std::max(left_scale, magnitude(left[i]));
  }
  if (not(right_scale > 0 and left_scale > 0 and finite(right_scale) and
          finite(left_scale))) {
    throw std::runtime_error("analytic_vector_zero_or_nonfinite");
  }
  for (size_t i = 0; i < 7; ++i) {
    right[i] /= right_scale;
    left[i] /= left_scale;
  }
  const Real overlap = dot(left, right);
  if (overlap == 0 or not finite(overlap)) {
    throw std::runtime_error("analytic_LdotR_zero_or_nonfinite");
  }
  for (auto& value : left) {
    value /= overlap;
  }
  return {std::move(right), std::move(left)};
}

}  // namespace

SlowAction slow_action(const State& primitive, const double bn,
                       const double gamma, const State& left_tau,
                       const State& right_tau) {
  SlowAction result{};
  try {
    const PhysicalState state = physical_state(primitive, bn, gamma);
    // This analytical Type-I exit precedes roots, eigenvectors and all their
    // degeneracy-prone normalizations, exactly as in the scalar construction.
    if (state.magnetic[0] == 0) {
      result.supported = true;
      result.zero_operator = true;
      result.status = "exact_Bn_zero_filter";
      return result;
    }
    const Spectrum spectrum = physical_spectrum(state);
    if (spectrum.failure != nullptr) {
      result.supported = false;
      result.status = spectrum.failure;
      return result;
    }
    const FilterWeights filter = family_weights(state, spectrum);
    result.normalization_bound_active = filter.bound_active;
    for (size_t k = 0; k < 7; ++k) {
      result.lambdas[k] = to_double(spectrum.lambda[k]);
    }
    for (size_t j = 0; j < 2; ++j) {
      result.weights[j] = to_double(filter.weights[j]);
    }
    if (spectrum.exact_slow_alfven_collision) {
      result.supported = true;
      result.zero_operator = true;
      result.status = "exact_slow_alfven_zero_filter";
      return result;
    }
    MpState jump{};
    for (size_t i = 0; i < 7; ++i) {
      jump[i] = exact_double(right_tau[i]) - exact_double(left_tau[i]);
    }
    constexpr std::array<size_t, 2> slow_indices{SlowMinus, SlowPlus};
    for (size_t j = 0; j < 2; ++j) {
      if (filter.weights[j] == 0) {
        continue;
      }
      const size_t family = slow_indices[j];
      const auto pair =
          analytic_pair(state, spectrum.lambda[family], spectrum.gap[family]);
      const Real coefficient = filter.weights[j] * dot(pair.second, jump);
      const Real absolute_coefficient =
          magnitude(spectrum.lambda[family]) * coefficient;
      for (size_t i = 0; i < 7; ++i) {
        result.slow_jump_mp[i] += pair.first[i] * coefficient;
        result.absolute_slow_jump_mp[i] += pair.first[i] * absolute_coefficient;
      }
    }
    for (size_t i = 0; i < 7; ++i) {
      if (not(finite(result.slow_jump_mp[i]) and
              finite(result.absolute_slow_jump_mp[i]))) {
        throw std::runtime_error("nonfinite_high_precision_slow_action");
      }
      result.slow_jump[i] = to_double(result.slow_jump_mp[i]);
      result.absolute_slow_jump[i] =
          to_double(result.absolute_slow_jump_mp[i]);
      if (not(std::isfinite(result.slow_jump[i]) and
              std::isfinite(result.absolute_slow_jump[i]))) {
        throw std::runtime_error("nonfinite_rounded_slow_action");
      }
    }
    result.supported = true;
    result.status = "analytic_high_precision_pairs";
  } catch (const std::exception& error) {
    thread_local std::string message;
    message = error.what();
    result.supported = false;
    result.status = message.c_str();
  }
  return result;
}

}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE
