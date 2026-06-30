// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"

namespace io {

/*!
 * \brief Compute \f$\zeta\f$ analytically from the tabulated free-energy
 * derivatives of a CompOSE 3D EoS.
 *
 * Computes
 * \f[
 * \zeta = \left.\frac{\partial p}{\partial Y_e}\right|_{\rho,\epsilon}
 *       = \left.\frac{\partial(p,\epsilon) / \partial(T,Y_e)}
 *                    {\partial(Y_e,\epsilon) / \partial(T,Y_e)}\right|_{n_b}
 *       = p_{Y_e} - p_T\,\frac{\epsilon_{Y_e}}{\epsilon_T}
 * \f]
 * at fixed baryon density (\f$\rho \propto n_b\f$), where every partial
 * derivative is evaluated analytically from the free energy per baryon
 * \f$\mathcal{F}\f$. With \f$p = n_b^2 \mathcal{F}_{n_b}\f$ and
 * \f$\epsilon = \mathcal{F} - T\mathcal{F}_T\f$ this gives
 * \f{align}{
 * p_{Y_e}      &= n_b^2\,\mathcal{F}_{n_b Y_e}, &
 * p_T          &= n_b^2\,\mathcal{F}_{n_b T}, \\
 * \epsilon_{Y_e} &= \mathcal{F}_{Y_e} - T\,\mathcal{F}_{T Y_e}, &
 * \epsilon_T     &= -T\,\mathcal{F}_{T T}.
 * \f}
 * The neutron-mass scaling and rest-mass offset between CompOSE's stored
 * \f$\epsilon\f$ and \f$\mathcal{F} - T\mathcal{F}_T\f$ cancel because only the
 * ratio \f$\epsilon_{Y_e}/\epsilon_T\f$ enters, so the result is in units of
 * MeV/fm\f$^3\f$ per unit \f$Y_e\f$ (matching the tabulated pressure).
 *
 * The arguments `d2f_dt2`, `d2f_dt_dnb`, `d2f_dt_dye`, `d2f_dnb_dye`, and
 * `df_dye` are the CompOSE free-energy derivatives \f$\mathcal{F}_{TT}\f$,
 * \f$\mathcal{F}_{T n_b}\f$, \f$\mathcal{F}_{T Y_e}\f$,
 * \f$\mathcal{F}_{n_b Y_e}\f$, and \f$\mathcal{F}_{Y_e}\f$ (Table 7.3
 * derivative indices 3, 4, 5, 8, and 9). The `number_density_grid` and
 * `temperature_grid` hold the per-node \f$n_b\f$ (in fm\f$^{-3}\f$) and \f$T\f$
 * (in MeV).
 *
 * The CompOSE table is flattened in file order, with \f$Y_e\f$ varying
 * fastest, then \f$n_b\f$, then \f$T\f$: idx = (iT * nN + in) * nYe + iYe.
 */
DataVector compute_zeta_from_free_energy_derivatives(
    const DataVector& d2f_dt2, const DataVector& d2f_dt_dnb,
    const DataVector& d2f_dt_dye, const DataVector& d2f_dnb_dye,
    const DataVector& df_dye, const std::vector<double>& number_density_grid,
    const std::vector<double>& temperature_grid, size_t number_density_points,
    size_t temperature_points, size_t electron_fraction_points);

/*!
 * \brief Reconstruct the adiabatic sound speed squared from the tabulated
 * pressure, specific entropy, and specific internal energy of a CompOSE 3D
 * EoS.
 *
 * Computes
 * \f[
 * c_s^2 = \frac{1}{h}\left[\left(\frac{\partial p}{\partial n_b}\right)_T
 *  - \frac{(\partial s/\partial n_b)_T}{(\partial s/\partial T)_{n_b}}
 *    \left(\frac{\partial p}{\partial T}\right)_{n_b}\right]
 * \f]
 * at fixed \f$Y_e\f$, where \f$s = \mathcal{S}/n_b\f$ is the specific entropy
 * (CompOSE Q2) and \f$h = m_n(1+\epsilon) + p/n_b\f$ is the relativistic
 * enthalpy per baryon in MeV. The bracketed combination converts the
 * isothermal derivative \f$(\partial p/\partial n_b)_T\f$ into the adiabatic
 * one \f$(\partial p/\partial n_b)_{s,Y_e}\f$ via the Maxwell-style identity
 * \f$(\partial T/\partial n_b)_s = -(\partial s/\partial n_b)_T /
 * (\partial s/\partial T)_{n_b}\f$.
 *
 * The implementation follows the same finite-difference recipe as PyCompOSE
 * (https://github.com/computationalrelativity/PyCompOSE, GPL-3.0; this is a
 * clean C++ re-derivation, not a port of their source): derivatives are taken
 * in log-space on the CompOSE log-spaced \f$n_b\f$ and \f$T\f$ grids using a
 * 3-point centered stencil in the interior and 2-point one-sided stencils at
 * the boundaries, then converted back via
 * \f$\partial Q/\partial x = Q\,\partial(\ln Q)/\partial(\ln x)\,/\,x\f$.
 * The motivation is that CompOSE's own Q12 (sound speed squared, computed
 * internally from high-order interpolation of the free energy) can become
 * unphysical (negative or > 1) at table corners; recomputing \f$c_s^2\f$
 * directly from \f$p\f$ and \f$s\f$ removes that noise.
 *
 * Inputs `pressure` (MeV/fm\f$^3\f$, CompOSE Q1), `specific_entropy`
 * (dimensionless per baryon, Q2), and `specific_internal_energy`
 * (dimensionless, Q7 = \f$\epsilon\f$) are the table data flattened in
 * CompOSE order (\f$Y_e\f$ fastest, then \f$n_b\f$, then \f$T\f$). The grids
 * are \f$n_b\f$ in fm\f$^{-3}\f$ and \f$T\f$ in MeV. `neutron_mass_mev` is
 * the baryon mass scale used in \f$h\f$. If `cs2_floor` is positive, the
 * returned \f$c_s^2\f$ is element-wise clamped from below by it; pass a
 * non-positive value to disable the floor.
 *
 * For tables with a single temperature slice (\f$n_T = 1\f$) the formula
 * collapses to the cold limit \f$c_s^2 = (\partial p/\partial n_b)/h\f$.
 */
DataVector compute_cs2_from_pressure_and_entropy(
    const DataVector& pressure, const DataVector& specific_entropy,
    const DataVector& specific_internal_energy,
    const std::vector<double>& number_density_grid,
    const std::vector<double>& temperature_grid, double neutron_mass_mev,
    size_t number_density_points, size_t temperature_points,
    size_t electron_fraction_points, double cs2_floor);

/*!
 * \brief Reconstruct CompOSE Q11 (\f$\kappa = (\partial p/\partial
 * E)_{n_b,Y_e}\f$, where \f$E = m_n(1+\epsilon)\f$ is the energy per baryon
 * in MeV) from the tabulated pressure and specific internal energy of a
 * CompOSE 3D EoS.
 *
 * At fixed \f$(n_b, Y_e)\f$ both pressure and the dimensionless
 * \f$\epsilon\f$ depend on \f$T\f$, so the chain rule gives
 * \f[
 * \left(\frac{\partial p}{\partial E}\right)_{n_b,Y_e}
 *   = \frac{1}{m_n}\,\frac{(\partial p/\partial T)_{n_b,Y_e}}
 *                         {(\partial \epsilon/\partial T)_{n_b,Y_e}} \,.
 * \f]
 * The derivatives are evaluated by finite differences on the CompOSE
 * \f$T\f$ grid using a 3-point centered stencil in the interior and 2-point
 * one-sided stencils at the boundaries. Pressure is differenced in log-space
 * (it is strictly positive and spans many orders of magnitude); the
 * dimensionless \f$\epsilon\f$ is differenced linearly (it crosses zero for
 * cold matter and would yield a numerically-vanishing denominator if logged
 * via \f$\ln(1+\epsilon)\f$ at low \f$T\f$). After the common \f$/T\f$
 * cancellation in \f$(\partial p/\partial T)/(\partial \epsilon/\partial T)\f$
 * the formula reduces to
 * \f[
 * \kappa = \frac{p}{m_n}\,\frac{\ln p(T_+) - \ln p(T_-)}
 *                              {\epsilon(T_+) - \epsilon(T_-)} \,,
 * \f]
 * independent of the actual \f$T\f$ spacing.
 *
 * Returned \f$\kappa\f$ is in units of fm\f$^{-3}\f$ when called on raw
 * CompOSE data, matching CompOSE Q11; downstream `Tabulated3D` converts that
 * to geometric units via the standard
 * \f$\kappa_\mathrm{geom} = (m_n/p_\mathrm{unit})\,\kappa_\mathrm{CompOSE}\f$
 * rescaling already applied to CompOSE Q11.
 *
 * Inputs are flattened in CompOSE order (\f$Y_e\f$ fastest, then \f$n_b\f$,
 * then \f$T\f$). For tables with a single temperature slice (\f$n_T = 1\f$)
 * the T-derivatives are undefined; the returned array is filled with
 * `kappa_floor` if it is positive, otherwise with zeros. If `kappa_floor` is
 * positive the result is also element-wise clamped from below by it; pass a
 * non-positive value to disable the floor.
 */
DataVector compute_kappa_from_pressure_and_energy(
    const DataVector& pressure, const DataVector& specific_internal_energy,
    const std::vector<double>& temperature_grid, double neutron_mass_mev,
    size_t number_density_points, size_t temperature_points,
    size_t electron_fraction_points, double kappa_floor);

}  // namespace io
