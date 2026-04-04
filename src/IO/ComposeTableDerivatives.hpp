// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"

namespace io {

/// @{
/*!
 * \brief Compute \f$\zeta\f$ from a tabulated CompOSE 3D EoS.
 *
 * Computes
 * \f[
 * \zeta = \left(\frac{\partial p}{\partial Y_e}\right)_{\rho,T}
 *        - \left(\frac{\partial p}{\partial T}\right)_{\rho,Y_e}
 *          \frac{\left(\frac{\partial \epsilon}{\partial Y_e}\right)_{\rho,T}}
 *               {\left(\frac{\partial \epsilon}{\partial T}\right)_{\rho,Y_e}}
 * \f]
 * using 3-point finite differences in \f$T\f$ and \f$Y_e\f$, with the stencil
 * weights computed from the tabulated coordinates so nonuniform grids are
 * handled correctly. If a direction has only 2 points, a 2-point one-sided
 * difference is used.
 *
 * Assumes the CompOSE table is flattened in file order, with \f$Y_e\f$
 * varying fastest, then \f$n_b\f$, then \f$T\f$:
 * idx = (iT * nN + in) * nYe + iYe.
 */
DataVector compute_zeta_from_pressure_and_eps(
    const DataVector& pressure, const DataVector& specific_internal_energy,
    const std::vector<double>& temperature_grid,
    const std::vector<double>& electron_fraction_grid,
    size_t number_density_points, size_t temperature_points,
    size_t electron_fraction_points);
/// @}

}  // namespace io
