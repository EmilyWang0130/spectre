// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "NumericalAlgorithms/FiniteDifference/Reconstruct.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Direction;
template <size_t Dim>
class Index;
/// \endcond

namespace fd::reconstruction {
namespace detail {
struct FirstOrderReconstructor {
  SPECTRE_ALWAYS_INLINE static std::array<double, 2> pointwise(
      const double* const q, const int /*stride*/) {
    // First-order (piecewise-constant / Godunov) reconstruction: the value on
    // both faces of the cell is just the cell-centered value, with no slope.
    return {{q[0], q[0]}};
  }

  SPECTRE_ALWAYS_INLINE static constexpr size_t stencil_width() { return 3; }
};
}  // namespace detail

/*!
 * \ingroup FiniteDifferenceGroup
 * \brief Performs first-order (piecewise-constant) reconstruction on the
 * `volume_vars` in each direction.
 *
 * On a 1d mesh the solution at the \f$j\f$th point is denoted by \f$u_j\f$. The
 * first-order reconstruction assumes the solution is constant in each cell, so
 * the reconstructed solution \f$u_j(x)\f$ in the \f$j\f$th cell is simply
 *
 * \f{align}
 * u_j(x) = u_j,
 * \f}
 *
 * i.e. both faces of the cell take the cell-centered value. This is the most
 * diffusive (and most robust) reconstruction; it is useful for reproducing
 * first-order finite-volume schemes such as those used in standard shock-tube
 * benchmarks.
 */
template <size_t Dim>
void first_order(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_upper_side_of_face_vars,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_lower_side_of_face_vars,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Index<Dim>& volume_extents, const size_t number_of_variables) {
  detail::reconstruct<detail::FirstOrderReconstructor>(
      reconstructed_upper_side_of_face_vars,
      reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
      volume_extents, number_of_variables);
}
}  // namespace fd::reconstruction
