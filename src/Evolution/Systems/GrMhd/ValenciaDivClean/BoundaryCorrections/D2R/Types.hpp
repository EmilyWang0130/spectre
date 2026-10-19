// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

// The D2-R kernel is compiled twice from the same sources (2026-10-19):
//  - without D2R_MULTIPRECISION, in namespace d2r_detail, with Real = double:
//    the production kernel, selected by default;
//  - with D2R_MULTIPRECISION (the *Mp.cpp wrappers), in namespace d2r_detail_mp,
//    with Real a 160-decimal-digit Boost number: task 3's original correctness
//    backend, kept as the oracle and selected by SPECTRE_D2R_PRECISION=mp.
// The two differ only in the scalar type and in the rounding helpers below;
// the algebra is the same.
#ifdef D2R_MULTIPRECISION
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#define D2R_DETAIL_NAMESPACE d2r_detail_mp
#else
#define D2R_DETAIL_NAMESPACE d2r_detail
#endif

namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE {

#ifdef D2R_MULTIPRECISION
// Fixed precision in the original compiled correctness implementation.
// Disabling expression templates avoids storing references to temporary MP
// expressions.
using Real =
    boost::multiprecision::number<boost::multiprecision::cpp_dec_float<160>,
                                  boost::multiprecision::et_off>;
// Import the actual dyadic binary64 value before the single MP rounding.
// Decimal expansions exceeding the fixed precision are not represented exactly;
// this operation is not an outward interval enclosure or an exact-decimal
// claim.
inline Real exact_double(const double value) {
  if (not std::isfinite(value)) {
    throw std::invalid_argument("D2-R requires finite binary64 inputs");
  }
  if (value == 0.0) {
    return Real{0};
  }
  int exponent = 0;
  const double fraction = std::frexp(value, &exponent);
  const auto significand = static_cast<std::int64_t>(std::ldexp(fraction, 53));
  return boost::multiprecision::ldexp(Real{significand}, exponent - 53);
}
inline double to_double(const Real& x) { return x.convert_to<double>(); }
inline Real next_up(const Real& x) { return boost::math::float_next(x); }
inline bool finite_real(const Real& x) { return boost::math::isfinite(x); }
inline Real abs_real(const Real& x) { return boost::multiprecision::abs(x); }
constexpr bool multiprecision_build = true;
#else
using Real = double;
inline Real exact_double(const double value) {
  if (not std::isfinite(value)) {
    throw std::invalid_argument("D2-R requires finite binary64 inputs");
  }
  return value;
}
inline double to_double(const double x) { return x; }
inline Real next_up(const double x) {
  return std::nextafter(x, std::numeric_limits<double>::infinity());
}
inline bool finite_real(const double x) { return std::isfinite(x); }
inline Real abs_real(const double x) { return std::fabs(x); }
constexpr bool multiprecision_build = false;
using std::log;
using std::pow;
using std::sqrt;
#endif

using State = std::array<double, 7>;
using MpState = std::array<Real, 7>;

// Both state and flux use [D,mn,mt,mb,tau,Bt,Bb], with tau=E-D. Primitive
// arrays use [rho,vn,vt,vb,p,Bt,Bb]; Bn is always a common separate parameter.

}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::D2R_DETAIL_NAMESPACE
