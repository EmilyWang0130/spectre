// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <boost/multiprecision/cpp_dec_float.hpp>

namespace grmhd::ValenciaDivClean::BoundaryCorrections::d2r_detail {

// Fixed precision in the initial compiled correctness implementation. Disabling
// expression templates avoids storing references to temporary MP expressions.
using Real =
    boost::multiprecision::number<boost::multiprecision::cpp_dec_float<160>,
                                  boost::multiprecision::et_off>;
using State = std::array<double, 7>;
using MpState = std::array<Real, 7>;

// Both state and flux use [D,mn,mt,mb,tau,Bt,Bb], with tau=E-D. Primitive
// arrays use [rho,vn,vt,vb,p,Bt,Bb]; Bn is always a common separate parameter.
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

}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::d2r_detail
