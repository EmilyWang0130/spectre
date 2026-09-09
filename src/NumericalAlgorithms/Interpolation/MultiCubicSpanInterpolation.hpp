// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "NumericalAlgorithms/Interpolation/LagrangePolynomial.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace intrp {

/// The number of points along one dimension used by MultiCubicSpanInterpolation
/// (4-point Lagrange).
inline constexpr size_t multi_cubic_stencil_width = 4;

/// The number of table nodes contributing to a Dim-dimensional
/// MultiCubicSpanInterpolation query. For Dim=3 this is 64.
template <size_t Dim>
inline constexpr size_t multi_cubic_stencil_size =
    pow<static_cast<int>(Dim)>(multi_cubic_stencil_width);

/*!
 * \brief Performs tensor-product 4-point Lagrange (cubic) interpolation
 * in arbitrary dimensions on a rectilinear grid.
 *
 * Mirrors the public API of intrp::MultiLinearSpanInterpolation so it can
 * serve as a higher-order drop-in replacement. The class is non-owning
 * and expects a C-ordered array `(n, x, y, z)` in which the variable
 * index `n` varies fastest in memory.
 *
 * Interior stencils are centered on the interval [x[i], x[i+1]] that
 * brackets the target, i.e. they use table nodes {i-1, i, i+1, i+2}.
 * At table boundaries the stencil is shifted so it does not extend
 * outside the table, giving a one-sided 4-point Lagrange approximation
 * (still 4th-order accurate in the interior; slightly biased near
 * boundaries). This is a strict improvement over trilinear (2-point
 * per direction) because the reconstruction is continuous with
 * continuous first derivative within each 4-point cell instead of
 * only C0 across cell edges.
 *
 * \tparam Dim dimensionality of the table
 * \tparam NumberOfVariables number of variables stored in the table
 * \tparam UniformSpacing whether the table has uniform spacing along
 * each axis; enables a faster index lookup.
 */
template <size_t Dim, size_t NumberOfVariables, bool UniformSpacing>
class MultiCubicSpanInterpolation {
 public:
  /// The interpolation weights (and corresponding flat indices into the
  /// underlying table) for a Dim-dimensional stencil.
  template <size_t ThisDimension>
  struct Weight {
    std::array<double, multi_cubic_stencil_size<ThisDimension>> weights;
    Index<multi_cubic_stencil_size<ThisDimension>> index;
  };

  size_t extents(const size_t which_dimension) const {
    return number_of_points_[which_dimension];
  }

  void extrapolate_above_data(const size_t which_dimension, const bool value) {
    allow_extrapolation_above_data_[which_dimension] = value;
  }

  void extrapolate_below_data(const size_t which_dimension, const bool value) {
    allow_extrapolation_below_data_[which_dimension] = value;
  }

  /// Compute interpolation weights for 1D tables
  Weight<1> get_weights(const double x1) const;

  /// Compute interpolation weights for 2D tables
  Weight<2> get_weights(const double x1, const double x2) const;

  /// Compute interpolation weights for 3D tables
  Weight<3> get_weights(const double x1, const double x2,
                        const double x3) const;

  double interpolate(const Weight<Dim>& weights,
                     const size_t which_variable = 0) const {
    double result = 0.0;
    for (size_t nn = 0; nn < weights.weights.size(); ++nn) {
      result += weights.weights[nn] *
                y_[which_variable + NumberOfVariables * weights.index[nn]];
    }
    return result;
  }

  template <size_t... variables_to_interpolate>
  std::array<double, sizeof...(variables_to_interpolate)> interpolate(
      const Weight<Dim>& weights) const {
    static_assert(sizeof...(variables_to_interpolate) <= NumberOfVariables,
                  "You are trying to interpolate more variables than this "
                  "container holds.");
    return std::array<double, sizeof...(variables_to_interpolate)>{
        interpolate(weights, variables_to_interpolate)...};
  }

  template <size_t NumberOfVariablesToInterpolate, std::floating_point... T>
  std::array<double, NumberOfVariablesToInterpolate> interpolate(
      std::array<size_t, NumberOfVariablesToInterpolate>&
          variables_to_interpolate,
      const T&... target_points) const {
    static_assert(sizeof...(T) == Dim,
                  "You need to provide the correct number of target points");
    static_assert(NumberOfVariablesToInterpolate <= NumberOfVariables,
                  "You are trying to interpolate more variables than this "
                  "container holds.");
    auto weights = get_weights(target_points...);
    std::array<double, NumberOfVariablesToInterpolate> interpolated_values;
    for (size_t nn = 0; nn < NumberOfVariablesToInterpolate; ++nn) {
      interpolated_values[nn] =
          interpolate(weights, variables_to_interpolate[nn]);
    }
    return interpolated_values;
  }

  MultiCubicSpanInterpolation() = default;

  MultiCubicSpanInterpolation(std::array<gsl::span<const double>, Dim> x,
                              gsl::span<const double> y,
                              Index<Dim> number_of_points);

  double lower_bound(const size_t which_dimension) const {
    return x_[which_dimension][0];
  }

  double upper_bound(const size_t which_dimension) const {
    return x_[which_dimension][number_of_points_[which_dimension] - 1];
  }

 private:
  std::array<bool, Dim> allow_extrapolation_below_data_{};
  std::array<bool, Dim> allow_extrapolation_above_data_{};
  std::array<double, Dim> inverse_spacing_{};
  std::array<double, Dim> spacing_{};
  Index<Dim> number_of_points_{};

  using DataPointer = gsl::span<const double>;
  std::array<DataPointer, Dim> x_{};
  DataPointer y_{};

  /// Left index of the bracketing interval [i, i+1] that contains the
  /// target point (or the extrapolation-clamped boundary interval).
  size_t find_index_general(const size_t which_dimension,
                            const double& target_point) const;

  size_t find_index_uniform(const size_t which_dimension,
                            const double& target_point) const;

  size_t find_index(const size_t which_dimension,
                    const double& target_point) const {
    if constexpr (UniformSpacing) {
      return find_index_uniform(which_dimension, target_point);
    } else {
      return find_index_general(which_dimension, target_point);
    }
  }

  /// Given the left index i of the bracketing interval [i, i+1], returns
  /// the index of the first stencil point so that the 4-point stencil is
  /// {stencil_start, ..., stencil_start + 3}. Shifted at boundaries so
  /// the stencil stays inside the table.
  size_t stencil_start(const size_t which_dimension,
                       const size_t bracket_left) const {
    const size_t n = number_of_points_[which_dimension];
    ASSERT(n >= multi_cubic_stencil_width,
           "MultiCubicSpanInterpolation requires at least "
               << multi_cubic_stencil_width
               << " table points along every dimension. Dimension "
               << which_dimension << " has only " << n << " points.");
    // Prefer a centered stencil {i-1, i, i+1, i+2}. Clamp to keep the
    // stencil inside [0, n-1].
    if (bracket_left == 0) {
      return 0;
    }
    if (bracket_left >= n - 2) {
      return n - multi_cubic_stencil_width;
    }
    return bracket_left - 1;
  }

  /// Compute the four Lagrange basis weights L_0..L_3 for the target
  /// point given the stencil node coordinates. Wraps
  /// intrp::lagrange_polynomial for each basis index so we reuse the
  /// existing 1D Lagrange evaluator.
  static std::array<double, multi_cubic_stencil_width> lagrange_weights_1d(
      const std::array<double, multi_cubic_stencil_width>& t_nodes,
      const double t) {
    return {
        lagrange_polynomial(0, t, t_nodes.begin(), t_nodes.end()),
        lagrange_polynomial(1, t, t_nodes.begin(), t_nodes.end()),
        lagrange_polynomial(2, t, t_nodes.begin(), t_nodes.end()),
        lagrange_polynomial(3, t, t_nodes.begin(), t_nodes.end()),
    };
  }
};

// ---------------------------------------------------------------------------
// Implementations
// ---------------------------------------------------------------------------

template <size_t Dim, size_t NumberOfVariables, bool UniformSpacing>
MultiCubicSpanInterpolation<Dim, NumberOfVariables, UniformSpacing>::
    MultiCubicSpanInterpolation(std::array<gsl::span<const double>, Dim> x,
                                gsl::span<const double> y,
                                Index<Dim> number_of_points)
    : number_of_points_(number_of_points), x_(x), y_(y) {
  for (size_t d = 0; d < Dim; ++d) {
    ASSERT(number_of_points_[d] >= multi_cubic_stencil_width,
           "MultiCubicSpanInterpolation requires at least "
               << multi_cubic_stencil_width
               << " points per dimension; dimension " << d << " has "
               << number_of_points_[d] << ".");
    allow_extrapolation_below_data_[d] = false;
    allow_extrapolation_above_data_[d] = false;
    if constexpr (UniformSpacing) {
      spacing_[d] = (x_[d][number_of_points_[d] - 1] - x_[d][0]) /
                    static_cast<double>(number_of_points_[d] - 1);
      inverse_spacing_[d] = 1.0 / spacing_[d];
    }
  }
}

template <size_t Dim, size_t NumberOfVariables, bool UniformSpacing>
size_t MultiCubicSpanInterpolation<Dim, NumberOfVariables, UniformSpacing>::
    find_index_general(const size_t which_dimension,
                       const double& target_point) const {
  const size_t n = number_of_points_[which_dimension];
  ASSERT((target_point > x_[which_dimension][0]) or
             allow_extrapolation_below_data_[which_dimension],
         "Interpolation exceeds lower table bounds. dim="
             << which_dimension << " target=" << target_point);
  ASSERT((target_point < x_[which_dimension][n - 1]) or
             allow_extrapolation_above_data_[which_dimension],
         "Interpolation exceeds upper table bounds. dim="
             << which_dimension << " target=" << target_point);
  // Bracket via bisection on the non-uniform grid.
  size_t lo = 0;
  size_t hi = n - 1;
  while (hi - lo > 1) {
    const size_t mid = lo + (hi - lo) / 2;
    if (target_point < x_[which_dimension][mid]) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  return lo;
}

template <size_t Dim, size_t NumberOfVariables, bool UniformSpacing>
size_t MultiCubicSpanInterpolation<Dim, NumberOfVariables, UniformSpacing>::
    find_index_uniform(const size_t which_dimension,
                       const double& target_point) const {
  const size_t n = number_of_points_[which_dimension];
  const double frac = (target_point - x_[which_dimension][0]) *
                      inverse_spacing_[which_dimension];
  if (frac <= 0.0) {
    return 0;
  }
  const size_t i = static_cast<size_t>(std::floor(frac));
  if (i >= n - 1) {
    return n - 2;
  }
  return i;
}

template <size_t Dim, size_t NumberOfVariables, bool UniformSpacing>
typename MultiCubicSpanInterpolation<Dim, NumberOfVariables,
                                     UniformSpacing>::template Weight<1>
MultiCubicSpanInterpolation<Dim, NumberOfVariables,
                            UniformSpacing>::get_weights(const double x1)
    const {
  static_assert(Dim >= 1, "Requesting 1D weights on a lower-dim table");
  const size_t bracket = find_index(0, x1);
  const size_t start = stencil_start(0, bracket);
  const std::array<double, multi_cubic_stencil_width> t_nodes{
      x_[0][start], x_[0][start + 1], x_[0][start + 2], x_[0][start + 3]};
  const auto w = lagrange_weights_1d(t_nodes, x1);
  Weight<1> result;
  for (size_t i = 0; i < multi_cubic_stencil_width; ++i) {
    result.weights[i] = w[i];
    result.index[i] = start + i;
  }
  return result;
}

template <size_t Dim, size_t NumberOfVariables, bool UniformSpacing>
typename MultiCubicSpanInterpolation<Dim, NumberOfVariables,
                                     UniformSpacing>::template Weight<2>
MultiCubicSpanInterpolation<Dim, NumberOfVariables,
                            UniformSpacing>::get_weights(const double x1,
                                                         const double x2)
    const {
  static_assert(Dim >= 2, "Requesting 2D weights on a lower-dim table");
  const std::array<size_t, 2> bracket{find_index(0, x1), find_index(1, x2)};
  const std::array<size_t, 2> start{stencil_start(0, bracket[0]),
                                    stencil_start(1, bracket[1])};
  const std::array<double, multi_cubic_stencil_width> t0_nodes{
      x_[0][start[0]], x_[0][start[0] + 1], x_[0][start[0] + 2],
      x_[0][start[0] + 3]};
  const std::array<double, multi_cubic_stencil_width> t1_nodes{
      x_[1][start[1]], x_[1][start[1] + 1], x_[1][start[1] + 2],
      x_[1][start[1] + 3]};
  const auto w0 = lagrange_weights_1d(t0_nodes, x1);
  const auto w1 = lagrange_weights_1d(t1_nodes, x2);
  Weight<2> result;
  const size_t n0 = number_of_points_[0];
  size_t nn = 0;
  for (size_t j = 0; j < multi_cubic_stencil_width; ++j) {
    for (size_t i = 0; i < multi_cubic_stencil_width; ++i) {
      result.weights[nn] = w0[i] * w1[j];
      result.index[nn] = (start[0] + i) + n0 * (start[1] + j);
      ++nn;
    }
  }
  return result;
}

template <size_t Dim, size_t NumberOfVariables, bool UniformSpacing>
typename MultiCubicSpanInterpolation<Dim, NumberOfVariables,
                                     UniformSpacing>::template Weight<3>
MultiCubicSpanInterpolation<Dim, NumberOfVariables, UniformSpacing>::
    get_weights(const double x1, const double x2, const double x3) const {
  static_assert(Dim >= 3, "Requesting 3D weights on a lower-dim table");
  const std::array<size_t, 3> bracket{find_index(0, x1), find_index(1, x2),
                                      find_index(2, x3)};
  const std::array<size_t, 3> start{stencil_start(0, bracket[0]),
                                    stencil_start(1, bracket[1]),
                                    stencil_start(2, bracket[2])};
  const std::array<double, multi_cubic_stencil_width> t0_nodes{
      x_[0][start[0]], x_[0][start[0] + 1], x_[0][start[0] + 2],
      x_[0][start[0] + 3]};
  const std::array<double, multi_cubic_stencil_width> t1_nodes{
      x_[1][start[1]], x_[1][start[1] + 1], x_[1][start[1] + 2],
      x_[1][start[1] + 3]};
  const std::array<double, multi_cubic_stencil_width> t2_nodes{
      x_[2][start[2]], x_[2][start[2] + 1], x_[2][start[2] + 2],
      x_[2][start[2] + 3]};
  const auto w0 = lagrange_weights_1d(t0_nodes, x1);
  const auto w1 = lagrange_weights_1d(t1_nodes, x2);
  const auto w2 = lagrange_weights_1d(t2_nodes, x3);
  Weight<3> result;
  const size_t n0 = number_of_points_[0];
  const size_t n1 = number_of_points_[1];
  size_t nn = 0;
  for (size_t k = 0; k < multi_cubic_stencil_width; ++k) {
    for (size_t j = 0; j < multi_cubic_stencil_width; ++j) {
      for (size_t i = 0; i < multi_cubic_stencil_width; ++i) {
        result.weights[nn] = w0[i] * w1[j] * w2[k];
        result.index[nn] =
            (start[0] + i) + n0 * ((start[1] + j) + n1 * (start[2] + k));
        ++nn;
      }
    }
  }
  return result;
}

/// Multi-cubic tensor-product span interpolation with uniform grid spacing
template <size_t Dim, size_t NumberOfVariables>
using UniformMultiCubicSpanInterpolation =
    MultiCubicSpanInterpolation<Dim, NumberOfVariables, true>;

/// Multi-cubic tensor-product span interpolation with non-uniform grid spacing
template <size_t Dim, size_t NumberOfVariables>
using GeneralMultiCubicSpanInterpolation =
    MultiCubicSpanInterpolation<Dim, NumberOfVariables, false>;

}  // namespace intrp
