// Distributed under the MIT License.
// See LICENSE.txt for details.
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/D2R/State.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#ifdef D2R_MULTIPRECISION
#include <boost/multiprecision/cpp_int.hpp>
#endif

namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE {
namespace {
#ifdef D2R_MULTIPRECISION
using Integer = boost::multiprecision::cpp_int;
using Rational = boost::multiprecision::cpp_rational;
#else
// In the double build the "exact rational" bookkeeping is ordinary arithmetic
// carried in long double and rounded once to double at the end.
using Rational = long double;
#endif
using Triple = std::array<Real, 3>;

bool finite(const State& x) {
  return std::all_of(x.begin(), x.end(),
                     [](double a) { return std::isfinite(a); });
}
bool finite(const Real& x) { return finite_real(x); }
Real absolute(const Real& x) { return abs_real(x); }
Real maximum(const Real& a, const Real& b) { return a > b ? a : b; }
Real dot(const Triple& a, const Triple& b) {
  Real out = 0;
  for (std::size_t i = 0; i < 3; ++i)
    out += a[i] * b[i];
  return out;
}
Real dot(const MpState& a, const MpState& b) {
  Real out = 0;
  for (std::size_t i = 0; i < 7; ++i)
    out += a[i] * b[i];
  return out;
}
Triple cross(const Triple& a, const Triple& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
MpState imported(const State& x) {
  MpState out{};
  for (std::size_t i = 0; i < 7; ++i)
    out[i] = exact_double(x[i]);
  return out;
}
#ifdef D2R_MULTIPRECISION
Rational dyadic(double x) {
  if (x == 0.)
    return Rational(0);
  int exponent = 0;
  const double mantissa = std::frexp(x, &exponent);
  const auto integer = static_cast<std::int64_t>(std::ldexp(mantissa, 53));
  Rational out(integer);
  exponent -= 53;
  if (exponent >= 0)
    out *= Integer(1) << exponent;
  else
    out /= Integer(1) << (-exponent);
  return out;
}
Integer power10(unsigned long exponent) {
  Integer out = 1, factor = 10;
  while (exponent != 0) {
    if ((exponent & 1UL) != 0)
      out *= factor;
    exponent >>= 1U;
    if (exponent != 0)
      factor *= factor;
  }
  return out;
}
Rational represented_real(const Real& value) {
  // Include every cpp_dec_float guard limb, not merely its public160 digits.
  // Consequently this is the exact finite decimal represented by the backend.
  const std::string text =
      value.str(Real::backend_type::cpp_dec_float_max_digits10,
                std::ios_base::scientific);
  const auto e = text.find_first_of("eE");
  std::string mantissa = text.substr(0, e);
  long exponent = e == std::string::npos ? 0 : std::stol(text.substr(e + 1));
  bool negative = false;
  if (!mantissa.empty() &&
      (mantissa.front() == '-' || mantissa.front() == '+')) {
    negative = mantissa.front() == '-';
    mantissa.erase(mantissa.begin());
  }
  const auto point = mantissa.find('.');
  if (point != std::string::npos) {
    exponent -= static_cast<long>(mantissa.size() - point - 1);
    mantissa.erase(point, 1);
  }
  const auto first = mantissa.find_first_not_of('0');
  if (first == std::string::npos)
    return Rational(0);
  mantissa.erase(0, first);
  // cpp_int's string constructor otherwise treats a leading zero as octal.
  Integer numerator(mantissa);
  if (negative)
    numerator = -numerator;
  Rational out(numerator);
  if (exponent >= 0)
    out *= power10(static_cast<unsigned long>(exponent));
  else
    out /= power10(static_cast<unsigned long>(-exponent));
  return out;
}
bool rounded_rational(double* out, const Rational& value) {
  Integer n = numerator(value), d = denominator(value);
  if (n == 0) {
    *out = 0.;
    return true;
  }
  const bool negative = n < 0;
  if (negative)
    n = -n;
  long exponent = static_cast<long>(boost::multiprecision::msb(n)) -
                  static_cast<long>(boost::multiprecision::msb(d));
  if (exponent >= 0) {
    if (n < (d << exponent))
      --exponent;
  } else if ((n << (-exponent)) < d)
    --exponent;
  if (exponent > 1023)
    return false;
  if (exponent < -1075) {
    *out = negative ? -0. : 0.;
    return true;
  }
  const long quantum = std::max(exponent - 52, -1074L);
  Integer scaled_n = n, scaled_d = d;
  if (quantum >= 0)
    scaled_d <<= quantum;
  else
    scaled_n <<= (-quantum);
  Integer q = scaled_n / scaled_d;
  const Integer remainder = scaled_n % scaled_d;
  if (2 * remainder > scaled_d || (2 * remainder == scaled_d && (q & 1) != 0))
    ++q;
  const double magnitude =
      std::ldexp(q.convert_to<double>(), static_cast<int>(quantum));
  *out = negative ? -magnitude : magnitude;
  return std::isfinite(*out);
}
bool rounded_real(double* out, const Real& value) {
  if (!finite(value))
    return false;
  return rounded_rational(out, represented_real(value));
}
bool rounded_state(State* out, const MpState& value) {
  for (std::size_t i = 0; i < 7; ++i)
    if (!rounded_real(&(*out)[i], value[i]))
      return false;
  return true;
}

#else
Rational dyadic(const double x) { return x; }
Rational represented_real(const Real& value) { return value; }
bool rounded_rational(double* out, const Rational& value) {
  *out = static_cast<double>(value);
  return std::isfinite(*out);
}
bool rounded_real(double* out, const Real& value) {
  if (!std::isfinite(value))
    return false;
  *out = value;
  return true;
}
bool rounded_state(State* out, const MpState& value) {
  for (std::size_t i = 0; i < 7; ++i)
    if (!rounded_real(&(*out)[i], value[i]))
      return false;
  return true;
}
#endif

struct Parts {
  Real rho, p, v2, b2, vb, w, wm1;
  Triple v, b;
};
Parts parts(const MpState& p, double bn) {
  Parts x{};
  x.rho = p[0];
  x.p = p[4];
  x.v = {p[1], p[2], p[3]};
  x.b = {exact_double(bn), p[5], p[6]};
  x.v2 = dot(x.v, x.v);
  x.b2 = dot(x.b, x.b);
  x.vb = dot(x.v, x.b);
  x.w = 1 / sqrt(1 - x.v2);
  x.wm1 = x.w * x.w * x.v2 / (x.w + 1);
  return x;
}
MpState conserved_mp(const MpState& p, double bn, double gamma) {
  const auto x = parts(p, bn);
  const Real k = exact_double(gamma) / (exact_double(gamma) - 1);
  const Real z = (x.rho + k * x.p) * x.w * x.w;
  const auto magnetic_momentum = cross(x.b, cross(x.v, x.b));
  const auto vx_b = cross(x.v, x.b);
  return {x.rho * x.w,
          z * x.v[0] + magnetic_momentum[0],
          z * x.v[1] + magnetic_momentum[1],
          z * x.v[2] + magnetic_momentum[2],
          x.rho * x.w * x.wm1 + x.p * (k * x.w * x.w - 1) +
              (x.b2 + dot(vx_b, vx_b)) / 2,
          x.b[1],
          x.b[2]};
}
MpState flux_mp(const MpState& p, double bn, double gamma) {
  const auto x = parts(p, bn);
  const auto u = conserved_mp(p, bn, gamma);
  Triple fm{};
  for (std::size_t i = 0; i < 3; ++i)
    fm[i] = u[i + 1] * x.v[0] -
            exact_double(bn) * ((1 - x.v2) * x.b[i] + x.vb * x.v[i]);
  fm[0] += x.p + (x.b2 * (1 - x.v2) + x.vb * x.vb) / 2;
  const auto magnetic_momentum = cross(x.b, cross(x.v, x.b));
  const Real k = exact_double(gamma) / (exact_double(gamma) - 1);
  const Real ftau = (x.rho * x.w * x.wm1 + k * x.p * x.w * x.w) * x.v[0] +
                    magnetic_momentum[0];
  return {u[0] * x.v[0],
          fm[0],
          fm[1],
          fm[2],
          ftau,
          x.b[1] * x.v[0] - x.b[0] * x.v[1],
          x.b[2] * x.v[0] - x.b[0] * x.v[2]};
}
bool reconstruction_budget(MpState* budget, const State& p, double bn,
                           double gamma) {
  const auto x = parts(imported(p), bn);
  const Real eps = pow(Real(2), -53);
  const Real g5 = 5 * eps / (1 - 5 * eps), g96 = 96 * eps / (1 - 96 * eps);
  const Real c = 1 - x.v2, dc = g5 * x.v2 + eps * (1 + x.v2);
  if (!(c > dc))
    return false;
  const Real low = (1 - eps) * (1 - eps) / sqrt(c + dc);
  const Real high = (1 + eps) * (1 + eps) / sqrt(c - dc);
  const Real dw = maximum(absolute(low - x.w), absolute(high - x.w));
  const Real k = exact_double(gamma) / (exact_double(gamma) - 1);
  const Real rh = x.rho + k * x.p, z = rh * x.w * x.w;
  Triple av{}, ab{};
  for (std::size_t i = 0; i < 3; ++i) {
    av[i] = absolute(x.v[i]);
    ab[i] = absolute(x.b[i]);
  }
  const Triple cross_abs{av[1] * ab[2] + av[2] * ab[1],
                         av[2] * ab[0] + av[0] * ab[2],
                         av[0] * ab[1] + av[1] * ab[0]};
  MpState terms{absolute(x.rho * x.w),
                0,
                0,
                0,
                absolute(x.rho * x.w * x.wm1) +
                    absolute(x.p) * (k * x.w * x.w + 1) +
                    (x.b2 + dot(cross_abs, cross_abs)) / 2,
                ab[1],
                ab[2]};
  const Real abs_vb = dot(av, ab);
  for (std::size_t i = 0; i < 3; ++i)
    terms[i + 1] = absolute(z) * av[i] + x.b2 * av[i] + ab[i] * abs_vb;
  const Real dz = absolute(rh) * (2 * x.w * dw + dw * dw);
  const Real dtau = absolute(x.rho) * absolute(2 * x.w - 1) * dw +
                    absolute(x.rho) * dw * dw +
                    absolute(k * x.p) * (2 * x.w * dw + dw * dw);
  const MpState propagated{
      absolute(x.rho) * dw, dz * av[0], dz * av[1], dz * av[2], dtau, 0, 0};
  for (std::size_t i = 0; i < 7; ++i) {
    (*budget)[i] = g96 * terms[i] + propagated[i];
    if (!finite((*budget)[i]))
      return false;
  }
  return true;
}
LineResult line(bool supported, const std::string& reason,
                const State& flux = {}, const Real& coefficient = 0,
                const Real& margin = 0) {
  return {supported, reason, flux, coefficient, margin};
}
}  // namespace

bool valid_gamma(double gamma) {
  // Relative 1e-12 above 2: the index arrives as 1 + p/(rho eps) and Gamma = 2
  // can round an ulp high (see PlutoD2R::dg_package_data).
  return std::isfinite(gamma) && 1 < gamma && gamma <= 2 * (1 + 1e-12);
}
bool physical_primitive(const State& p, double bn, double gamma) {
  if (!finite(p) || !std::isfinite(bn) || !valid_gamma(gamma) || !(p[0] > 0) ||
      !(p[4] > 0))
    return false;
  const Triple v{exact_double(p[1]), exact_double(p[2]), exact_double(p[3])};
  const Real v2 = dot(v, v);
  return v2 < 1;
}
State conserved_tau(const State& p, double bn, double gamma) {
  State out{};
  if (!physical_primitive(p, bn, gamma) ||
      !rounded_state(&out, conserved_mp(imported(p), bn, gamma)))
    out.fill(std::numeric_limits<double>::quiet_NaN());
  return out;
}
State flux_tau(const State& p, double bn, double gamma) {
  State out{};
  if (!physical_primitive(p, bn, gamma) ||
      !rounded_state(&out, flux_mp(imported(p), bn, gamma)))
    out.fill(std::numeric_limits<double>::quiet_NaN());
  return out;
}
Admissibility admissibility(const State& u, double bn, double gamma) {
  Admissibility out{};
  if (!finite(u) || !std::isfinite(bn)) {
    out.reason = "nonfinite_input";
    return out;
  }
  if (!valid_gamma(gamma)) {
    out.reason = "gamma_outside_WT17_scope";
    return out;
  }
  const auto v = imported(u);
  const Real d = v[0], tau = v[4], e = d + tau;
  const Triple m{v[1], v[2], v[3]}, b{exact_double(bn), v[5], v[6]};
  if (!(d > 0)) {
    out.reason = "D_nonpositive";
    return out;
  }
  const Real b2 = dot(b, b), m2 = dot(m, m), mb = dot(m, b);
  const Real r = tau * (2 * d + tau) - m2, norm = sqrt(d * d + m2);
  out.q = e + norm > 0 ? r / (e + norm) : e - norm;
  if (!(e > 0 && out.q > 0)) {
    out.reason = "q_nonpositive_or_unresolved";
    return out;
  }
  const Real a = b2 - e, rad = a * a + 3 * r;
  if (rad < 0) {
    out.reason = "negative_Phi_radicand";
    return out;
  }
  const Real phi = sqrt(rad);
  const Real inner = a < 0 ? 3 * r / (phi - a) : phi + a;
  const Real first = a > 0 ? 3 * (r - a * a) / (phi + 2 * a) : phi - 2 * a;
  if (!(inner > 0 && first > 0)) {
    out.reason = "Psi_nonpositive_or_unresolved";
    return out;
  }
  out.psi = first * first * inner - Real(13.5) * (d * d * b2 + mb * mb);
  out.admissible = out.psi > 0;
  out.reason = out.admissible ? "admissible"
               : out.psi == 0 ? "Psi_zero_or_precision_unresolved"
                              : "Psi_nonpositive";
  return out;
}

Recovery recover(const State& original, double bn, double gamma) {
  Recovery out{};
  out.admissibility = admissibility(original, bn, gamma);
  if (!out.admissibility.admissible) {
    out.reason = "WT17_" + out.admissibility.reason;
    return out;
  }
  const auto u = imported(original);
  const Real d = u[0], tau = u[4], gm = exact_double(gamma), a = (gm - 1) / gm;
  const Triple m{u[1], u[2], u[3]}, b{exact_double(bn), u[5], u[6]};
  const Real b2 = dot(b, b), mb = dot(m, b);
  // cpp_dec_float retains guard limbs beyond its advertised precision.
  // Requiring equality of every guard limb is not a valid public-precision
  // stopping rule and can permit non-interior/cycling midpoints. Resolve a
  // bracket to its declared160-digit relative spacing, while preserving both
  // endpoints and all later physical/forward checks. This is an arithmetic
  // root criterion, never a physical activation band.
  const auto resolved = [](const Real& lower, const Real& upper) {
    const Real scale = maximum(absolute(lower), absolute(upper));
    return upper - lower <= std::numeric_limits<Real>::epsilon() * scale;
  };
  // Final numerical resource safeguard, not a convergence substitute. The
  // binary64 exponent span plus160 decimal digits needs far fewer bisections
  // on ordinary finite inputs. Exhaustion is explicitly unsupported.
  constexpr std::size_t maximum_evaluations = 16384;
  struct RootState {
    Triple v{};
    Real v2 = 0, t = 0, p = 0, f = 0;
    bool causal = false;
  };
  const auto state = [&](const Real& x) {
    ++out.iterations;
    RootState s{};
    const Real z = d + x;
    for (std::size_t i = 0; i < 3; ++i)
      s.v[i] = (m[i] + (mb / z) * b[i]) / (z + b2);
    s.v2 = dot(s.v, s.v);
    s.causal = s.v2 < 1;
    if (!s.causal)
      return s;
    const Real t = sqrt(1 - s.v2);
    s.t = t;
    s.p = a * (x * (1 - s.v2) - d * t * s.v2 / (1 + t));
    const auto vx_b = cross(s.v, b);
    s.f = x - s.p + (b2 + dot(vx_b, vx_b)) / 2 - tau;
    return s;
  };
  // Bisection with foregone decisions skipped (2026-10-20). Both bisections
  // below run to adjacent binary64 values, about 55 evaluations each, and
  // almost all of their decisions lie far from the crossing, where the sign of
  // the predicate is that of the exact function: the pressure is negative from
  // the causal limit up to the pressure-zero boundary and positive beyond it,
  // and the energy residual increases through its root. In the double build a
  // safeguarded Newton iteration locates the crossing, and probes on either
  // side of it whose value exceeds four times an estimate of its rounding
  // error bound a window: a midpoint at or below the window's low end takes
  // the lower decision and one at or above its high end the upper decision,
  // as the exact function and so the evaluation would give them; the
  // midpoints inside are evaluated as before. The bisection keeps its
  // midpoints, its termination and its final bracket, so its result. The
  // 160-digit build evaluates every midpoint (its window is the bracket).
  struct Window {
    Real low, high;
  };
#ifdef D2R_MULTIPRECISION
  const auto trusted_window = [](const Real& lower, const Real& upper,
                                 const RootState& /*at_upper*/,
                                 const bool /*energy*/) {
    return Window{lower, upper};
  };
#else
  const double eps = std::numeric_limits<double>::epsilon();
  const double b_norm = std::sqrt(b2), m_norm = std::sqrt(dot(m, m));
  struct Local {
    bool upper = false, safe = false, smooth = false;
    double g = 0., dg = 0., margin = 0.;
  };
  // The bisection's predicate at x (from its own state(x), so the decision a
  // midpoint at x would get), the value whose sign it tests, the derivative
  // for a Newton step and the rounding allowance: four times an estimate of
  // the error of that value, from the magnitudes of the terms it is made of
  // and the error of v^2 propagated through d/dv^2.
  const auto local = [&](const double x, const RootState& rs,
                         const bool energy) {
    Local r{};
    r.upper = energy ? rs.f >= 0 : (rs.causal && rs.p >= 0);
    if (!rs.causal) {
      // Below the causal limit the pressure predicate is false: p -> -a d t
      // < 0 as t -> 0. (The energy bracket starts above the boundary.)
      r.safe = !energy;
      return r;
    }
    const double z = d + x, zb = z + b2, v2 = rs.v2, t = rs.t;
    const double c = mb / (z * z * zb);
    Triple dv{};
    for (std::size_t i = 0; i < 3; ++i)
      dv[i] = -c * b[i] - rs.v[i] / zb;
    const double dv2 = 2 * dot(rs.v, dv), dt = -dv2 / (2 * t);
    const double t1 = x * (1 - v2), t2 = d * t * v2 / (1 + t);
    const double dp = a * ((1 - v2) - x * dv2 - d * (1 - 2 * t) * dt);
    const double v_terms = (m_norm + std::abs(mb) * b_norm / z) / zb;
    const double v2_error = eps * (8 * std::sqrt(v2) * v_terms + 2 * v2);
    const double inner_error =
        4 * eps * (std::abs(t1) + std::abs(t2) + std::abs(t1 - t2)) +
        v2_error * (std::abs(x) + d * std::abs(1 - 2 * t) / (2 * t));
    if (energy) {
      const auto vx_b = cross(rs.v, b), dvx_b = cross(dv, b);
      const double vxb2 = dot(vx_b, vx_b);
      r.g = rs.f;
      r.dg = 1 - dp + dot(vx_b, dvx_b);
      r.margin =
          4 * (4 * eps *
                   (std::abs(x) + std::abs(rs.p) + b2 / 2 + vxb2 / 2 +
                    std::abs(tau)) +
               a * inner_error + 8 * eps * std::sqrt(vxb2) * b_norm * v_terms);
    } else {
      r.g = rs.p;
      r.dg = dp;
      r.margin = 4 * a * inner_error;
    }
    r.smooth = std::isfinite(r.g) && std::isfinite(r.dg) && r.dg > 0 &&
               std::isfinite(r.margin);
    r.safe = r.smooth && (r.upper ? r.g > r.margin : r.g < -r.margin);
    return r;
  };
  // The window for a bracket [lower, upper] whose upper end's state is known.
  // Only probes whose decision is safe narrow it, so a poor Newton iterate can
  // cost evaluations but never a decision.
  const auto trusted_window = [&](const double lower, const double upper,
                                  const RootState& at_upper,
                                  const bool energy) {
    Window w{lower, upper};
    double below = lower, above = upper, x = upper, slope = 0., margin = 0.;
    bool located = false;
    Local c = local(x, at_upper, energy);
    for (int k = 0; k < 64; ++k) {
      if (c.upper) {
        above = x;
        if (c.safe)
          w.high = std::min(w.high, x);
      } else {
        below = x;
        if (c.safe)
          w.low = std::max(w.low, x);
      }
      if (c.smooth) {
        slope = c.dg;
        margin = c.margin;
      }
      if (!c.safe || !(std::nextafter(below, above) < above)) {
        located = slope > 0;
        break;
      }
      double next = c.smooth ? x - c.g / c.dg : below;
      if (!(next > below && next < above))
        next = below + (above - below) / 2;
      x = next;
      c = local(x, state(x), energy);
    }
    if (!located)
      return w;
    const double ulp =
        std::nextafter(x, std::numeric_limits<double>::infinity()) - x;
    const double first = std::max(4 * margin / slope, 4 * ulp);
    double delta = first;
    for (int k = 0; k < 4 && x - delta > w.low; ++k, delta *= 8) {
      const auto probe = local(x - delta, state(x - delta), energy);
      if (probe.upper)
        break;
      if (probe.safe) {
        w.low = x - delta;
        break;
      }
    }
    delta = first;
    for (int k = 0; k < 4 && x + delta < w.high; ++k, delta *= 8) {
      const auto probe = local(x + delta, state(x + delta), energy);
      if (!probe.upper)
        break;
      if (probe.safe) {
        w.high = x + delta;
        break;
      }
    }
    return w;
  };
#endif
  Real low = 0, boundary;
  auto s = state(low);
  if (s.causal && s.p >= 0) {
    boundary = low;
    out.boundary_termination = "nonnegative_pressure_at_zero";
  } else {
    Real high =
        maximum(maximum(d, sqrt(dot(m, m))), maximum(b2, absolute(tau)));
    while (true) {
      if (out.iterations >= maximum_evaluations) {
        out.reason = "boundary_bracket_evaluation_limit";
        return out;
      }
      s = state(high);
      if (s.causal && s.p >= 0)
        break;
      high *= 2;
      if (!finite(high)) {
        out.reason = "boundary_bracket_nonfinite";
        return out;
      }
    }
    const Window window = trusted_window(low, high, s, false);
    while (true) {
      if (resolved(low, high)) {
        out.boundary_termination = "declared_decimal_precision";
        break;
      }
      if (out.iterations >= maximum_evaluations) {
        out.reason = "boundary_bisection_evaluation_limit";
        return out;
      }
      const Real mid = (low + high) / 2;
      if (!(low < mid && mid < high)) {
        out.reason = "boundary_noninterior_midpoint_before_declared_precision";
        return out;
      }
      bool upper = mid >= window.high;
      if (window.low < mid && mid < window.high) {
        s = state(mid);
        upper = s.causal && s.p >= 0;
      }
      ++out.boundary_bisections;
      // Selects, not branches: the foregone decisions alternate unpredictably.
      high = upper ? mid : high;
      low = upper ? low : mid;
    }
    boundary = high;
  }
  s = state(boundary);
  if (!s.causal || s.f >= 0) {
    out.reason = "pressure_zero_energy_margin_unresolved";
    return out;
  }
  Real left = boundary,
       right = maximum(maximum(2 * boundary, d), maximum(b2, absolute(tau)));
  while (true) {
    if (out.iterations >= maximum_evaluations) {
      out.reason = "energy_bracket_evaluation_limit";
      return out;
    }
    s = state(right);
    if (s.causal && s.p >= 0 && s.f >= 0)
      break;
    right *= 2;
    if (!finite(right)) {
      out.reason = "energy_bracket_nonfinite";
      return out;
    }
  }
  const Window window = trusted_window(left, right, s, true);
  while (true) {
    if (resolved(left, right)) {
      out.energy_termination = "declared_decimal_precision";
      break;
    }
    if (out.iterations >= maximum_evaluations) {
      out.reason = "energy_bisection_evaluation_limit";
      return out;
    }
    const Real mid = (left + right) / 2;
    if (!(left < mid && mid < right)) {
      out.reason = "energy_noninterior_midpoint_before_declared_precision";
      return out;
    }
    bool upper = mid >= window.high;
    if (window.low < mid && mid < window.high) {
      s = state(mid);
      upper = s.f >= 0;
    }
    ++out.energy_bisections;
    right = upper ? mid : right;
    left = upper ? left : mid;
  }
  bool found = false;
  Real best_error = 0;
  for (const Real& x : std::array<Real, 2>{left, right}) {
    s = state(x);
    if (!s.causal || !(s.p > 0))
      continue;
    const MpState mp{
        d * sqrt(1 - s.v2), s.v[0], s.v[1], s.v[2], s.p, u[5], u[6]};
    State p{};
    if (!rounded_state(&p, mp) || !physical_primitive(p, bn, gamma))
      continue;
    MpState budget{};
    const State rebuilt = conserved_tau(p, bn, gamma);
    if (!finite(rebuilt))
      continue;
    bool ok = finite(rebuilt) && reconstruction_budget(&budget, p, bn, gamma);
    Real scale = exact_double(std::numeric_limits<double>::min()), error = 0;
    for (std::size_t i = 0; i < 7; ++i) {
      const Real delta = absolute(exact_double(rebuilt[i]) - u[i]);
      scale = maximum(scale, absolute(u[i]));
      error = maximum(error, delta);
      // The double build's root is resolved to binary64 precision only, so its
      // round trip carries that error on top of the rounding of p; allow it.
      ok = ok &&
           (original[i] == 0 ||
            std::abs(original[i]) >= std::numeric_limits<double>::min()) &&
           delta <= (multiprecision_build ? budget[i] : 8 * budget[i]);
    }
    error /= scale;
    if (!found || (ok && !out.roundtrip_checked) ||
        (ok == out.roundtrip_checked && error < best_error)) {
      found = true;
      out.primitive = p;
      out.high_precision_primitive = mp;
      out.roundtrip_checked = ok;
      best_error = error;
      out.reconstruction_error = to_double(error);
      out.thermal_relative_error =
          to_double(absolute(exact_double(p[4]) - s.p) / s.p);
    }
  }
  if (!found) {
    out.reason = "no_physical_rounded_primitive";
    return out;
  }
  out.success = out.roundtrip_checked;
  out.reason = out.success ? "recovered"
                           : "rounded_primitive_forward_support_unresolved";
  return out;
}

EntropyPair entropy_pair(const State& u, double bn, double gamma) {
  EntropyPair out{};
  out.recovery = recover(u, bn, gamma);
  if (!out.recovery.success) {
    out.reason = out.recovery.reason;
    return out;
  }
  const auto x = parts(out.recovery.high_precision_primitive, bn);
  const Real gm = exact_double(gamma), s = log(x.p) - gm * log(x.rho),
             beta = x.rho / x.p;
  const Real d = x.rho * x.w;
  out.eta = -d * s / (gm - 1);
  out.z = {(gm - s) / (gm - 1) - beta * x.wm1,
           beta * x.w * x.v[0],
           beta * x.w * x.v[1],
           beta * x.w * x.v[2],
           -beta * x.w,
           beta * (x.b[1] / x.w + x.w * x.vb * x.v[1]),
           beta * (x.b[2] / x.w + x.w * x.vb * x.v[2])};
  out.psi = d * x.v[0] + beta * x.w *
                             ((x.b2 * (1 - x.v2) + x.vb * x.vb) * x.v[0] / 2 -
                              x.b[0] * x.vb);
  out.success = true;
  out.reason = "numerical_entropy_pair";
  return out;
}

Reference reference_from_tau(const State& ul, const State& ur,
                             const State& left, const State& right, double bn,
                             double gamma) {
  Reference out{};
  if (!physical_primitive(left, bn, gamma) ||
      !physical_primitive(right, bn, gamma) || !finite(ul) || !finite(ur)) {
    out.reason = "physical_primitive_endpoints_required";
    return out;
  }
  State mid{};
  for (std::size_t i = 0; i < 7; ++i)
    if (!rounded_rational(&mid[i], (dyadic(ul[i]) + dyadic(ur[i])) / 2)) {
      out.reason = "midpoint_nonfinite";
      return out;
    }
  const auto rec = recover(mid, bn, gamma);
  if (!rec.success) {
    out.reason = "stored_tau_midpoint_" + rec.reason;
    return out;
  }
  const auto l = parts(imported(left), bn), r = parts(imported(right), bn);
  const Real wl = sqrt(l.rho), wr = sqrt(r.rho), sum = wl + wr;
  Triple difference{};
  for (std::size_t i = 0; i < 3; ++i)
    difference[i] = r.v[i] - l.v[i];
  const Real cross_term =
      ((1 - l.v2) + (1 - r.v2) + dot(difference, difference)) / 2;
  const Real norm2 =
      (wl * wl + wr * wr + 2 * wl * wr * l.w * r.w * cross_term) / (sum * sum);
  if (!(norm2 > 0)) {
    out.reason = "timelike_norm_unresolved";
    return out;
  }
  out.four_velocity[0] = (wl * l.w + wr * r.w) / sum / sqrt(norm2);
  out.primitive = rec.primitive;
  for (std::size_t i = 0; i < 3; ++i) {
    out.four_velocity[i + 1] =
        (wl * l.w * l.v[i] + wr * r.w * r.v[i]) / sum / sqrt(norm2);
    if (!rounded_real(&out.primitive[i + 1],
                      out.four_velocity[i + 1] / out.four_velocity[0])) {
      out.reason = "rounded_reference_nonfinite";
      return out;
    }
  }
  out.primitive[5] = mid[5];
  out.primitive[6] = mid[6];
  out.success = physical_primitive(out.primitive, bn, gamma);
  out.reason =
      out.success ? "physical_reference" : "rounded_reference_unphysical";
  return out;
}

PhysicalCap physical_ray_cap(const State& ul, const State& ur,
                             const State& shift, double bn, double gamma) {
  PhysicalCap out{};
  out.left = ul;
  out.right = ur;
  if (!finite(ul) || !finite(ur) || !finite(shift)) {
    out.reason = "invalid_pair_or_shift";
    return out;
  }
  Recovery left_recovery{}, right_recovery{};
  const auto at = [&](double beta, State* left, State* right) {
    for (std::size_t i = 0; i < 7; ++i) {
      // Match the prototype's stored long-double multiplication followed by
      // addition, not an accidental compiler-contracted FMA.
      volatile long double product = static_cast<long double>(beta) * shift[i];
      (*left)[i] =
          static_cast<double>(static_cast<long double>(ul[i]) + product);
      (*right)[i] =
          static_cast<double>(static_cast<long double>(ur[i]) - product);
    }
    ++out.evaluations;
    left_recovery = recover(*left, bn, gamma);
    if (!left_recovery.success)
      return false;
    right_recovery = recover(*right, bn, gamma);
    return right_recovery.success;
  };
  State l{}, r{};
  if (at(1., &l, &r)) {
    out.success = out.full_supported = true;
    out.beta = 1.;
    out.left = l;
    out.right = r;
    out.left_recovery = std::move(left_recovery);
    out.right_recovery = std::move(right_recovery);
    out.reason = "full_supported";
    return out;
  }
  if (!at(0., &l, &r)) {
    out.reason = "unsupported_original_pair";
    return out;
  }
  double low = 0., high = 1.;
  while (std::nextafter(low, high) < high) {
    const double mid = low + (high - low) / 2;
    if (at(mid, &l, &r))
      low = mid;
    else
      high = mid;
  }
  out.success = true;
  if (!at(low, &l, &r)) {
    out.reason = "baseline_only_final_support_gap";
    return out;
  }
  out.beta = low;
  out.left = l;
  out.right = r;
  out.left_recovery = std::move(left_recovery);
  out.right_recovery = std::move(right_recovery);
  out.reason = "verified_numerical_coefficient";
  return out;
}

bool rounded_ray(State* out, const State& base, const State& direction,
                 const Real& coefficient) {
  if (!finite(base) || !finite(direction) || !finite(coefficient))
    return false;
  const Rational c = represented_real(coefficient);
  for (std::size_t i = 0; i < 7; ++i)
    if (!rounded_rational(&(*out)[i],
                          dyadic(base[i]) + c * dyadic(direction[i])))
      return false;
  return true;
}

bool causal_hll(State* out, const State& ul, const State& ur, const State& fl,
                const State& fr) {
  if (!finite(ul) || !finite(ur) || !finite(fl) || !finite(fr)) {
    return false;
  }
  for (std::size_t i = 0; i < 7; ++i) {
    if (!rounded_rational(&(*out)[i], (dyadic(fl[i]) + dyadic(fr[i]) +
                                       dyadic(ul[i]) - dyadic(ur[i])) /
                                          2)) {
      return false;
    }
  }
  return true;
}

EntropyModel::EntropyModel(const State& ul, const State& ur, double bn,
                           double gamma)
    : left(entropy_pair(ul, bn, gamma)),
      right(entropy_pair(ur, bn, gamma)),
      ul_(ul),
      ur_(ur) {
  supported = left.success && right.success;
  reason = supported ? "supported"
                     : "entropy_inputs_" + left.reason + "_" + right.reason;
  if (!supported)
    return;
  dpsi = right.psi - left.psi;
  for (std::size_t i = 0; i < 7; ++i) {
    dz[i] = right.z[i] - left.z[i];
    jump[i] = exact_double(ur[i]) - exact_double(ul[i]);
  }
  h = dot(dz, jump);
}
Real EntropyModel::margin(const State& flux) const {
#ifdef D2R_MULTIPRECISION
  return dpsi - dot(dz, imported(flux));
#else
  long double acc = dpsi;
  for (std::size_t i = 0; i < 7; ++i)
    acc -= static_cast<long double>(dz[i]) * flux[i];
  return static_cast<Real>(acc);
#endif
}
Real EntropyModel::tolerance(const State& flux) const {
  // The 160-digit build decides entropy signs exactly; in double the margin is
  // a difference of terms of size |dpsi| + sum |dz_i F_i| and carries their
  // round-off, so a margin within that round-off of zero counts as zero.
  if (multiprecision_build)
    return 0;
  long double scale = std::fabs(static_cast<long double>(dpsi));
  for (std::size_t i = 0; i < 7; ++i)
    scale += std::fabs(static_cast<long double>(dz[i]) * flux[i]);
  return static_cast<Real>(32 * std::numeric_limits<double>::epsilon() * scale);
}
LineResult EntropyModel::repair(const State& candidate) const {
  if (!supported)
    return line(false, reason);
  if (!finite(candidate))
    return line(false, "nonfinite_native");
  const Real m0 = margin(candidate);
  if (m0 >= -tolerance(candidate))
    return line(true, "native_entropy_supported", candidate, 0, m0);
  if (!(h > 0))
    return line(false, "nonpositive_entropy_jump");
  const Real u = pow(Real(2), -53), sigma = pow(Real(2), -1075);
  Real a = 0, b = 0, c = 0;
  for (std::size_t i = 0; i < 7; ++i) {
    const Real z = absolute(dz[i]);
    a += z * absolute(exact_double(candidate[i]));
    b += z * absolute(jump[i]);
    c += z;
  }
  const Real denominator = h - u * b;
  if (!(denominator > 0))
    return line(false, "repair_rounding_budget_unresolved");
  Real nu = (-m0 + u * a + sigma * c) / denominator;
  nu = next_up(nu);
  State flux{};
  const Rational coefficient = represented_real(nu);
  for (std::size_t i = 0; i < 7; ++i)
    if (!rounded_rational(&flux[i],
                          dyadic(candidate[i]) -
                              coefficient * (dyadic(ur_[i]) - dyadic(ul_[i]))))
      return line(false, "repair_overflow");
  const Real m = margin(flux);
  if (m < -tolerance(flux))
    return line(false, "stored_repair_postcheck_failed", flux, nu, m);
  return line(true, "scalar_repaired", flux, nu, m);
}
LineResult EntropyModel::cap(const State& baseline,
                             const State& direction) const {
  if (!supported)
    return line(false, reason);
  if (!finite(baseline))
    return line(false, "nonfinite_baseline");
  if (!finite(direction))
    return line(false, "nonfinite_direction");
  const Real mb = margin(baseline);
  if (mb < -tolerance(baseline))
    return line(false, "unsupported_baseline_entropy");
  if (std::all_of(direction.begin(), direction.end(),
                  [](double x) { return x == 0.; }))
    return line(true, "zero_direction", baseline, 0, mb);
  State full{};
  if (rounded_ray(&full, baseline, direction, Real(1))) {
    const Real m = margin(full);
    if (m >= -tolerance(full))
      return line(true, "full_response", full, 1, m);
  }
  const Real u = pow(Real(2), -53), sigma = pow(Real(2), -1075);
  Real a = 0, b = 0, c = 0, slope = 0;
  for (std::size_t i = 0; i < 7; ++i) {
    const Real z = absolute(dz[i]), t = exact_double(direction[i]);
    a += z * absolute(exact_double(baseline[i]));
    b += z * absolute(t);
    c += z;
    slope += dz[i] * t;
  }
  const Real available = mb - u * a - sigma * c, safe_slope = slope + u * b;
  if (!(available > 0 && safe_slope > 0))
    return line(true, "no_supported_positive_affine_budget", baseline, 0, mb);
  const Real bound = std::min(Real(1), Real(available / safe_slope));
  double alpha = 0.;
  if (!rounded_real(&alpha, bound))
    return line(true, "affine_coefficient_unrepresentable", baseline, 0, mb);
  alpha = std::nextafter(alpha, 0.);
  if (!(alpha > 0))
    return line(true, "affine_coefficient_unrepresentable", baseline, 0, mb);
  State flux{};
  const Real coefficient = exact_double(alpha);
  if (!rounded_ray(&flux, baseline, direction, coefficient) ||
      margin(flux) < -tolerance(flux))
    return line(true, "stored_response_postcheck_failed", baseline, 0, mb);
  return line(true, "fractional_response", flux, coefficient, margin(flux));
}

}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE
