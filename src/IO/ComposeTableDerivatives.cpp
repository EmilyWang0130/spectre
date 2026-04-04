// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/ComposeTableDerivatives.hpp"

#include <cmath>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace io {
namespace {

size_t idx_of(const size_t in, const size_t iT, const size_t iYe,
              const size_t nN, const size_t nYe) {
  return (iT * nN + in) * nYe + iYe;
}

double first_derivative_3_point(const double f0, const double f1,
                                const double f2, const double x0,
                                const double x1, const double x2) {
  const double dx01 = x0 - x1;
  const double dx02 = x0 - x2;
  const double dx12 = x1 - x2;

  if (dx01 == 0.0 or dx02 == 0.0 or dx12 == 0.0) {
    ERROR("Finite-difference stencil points must be distinct, but got x = ("
          << x0 << ", " << x1 << ", " << x2 << ").");
  }

  const double w0 = (2.0 * x0 - x1 - x2) / (dx01 * dx02);
  const double w1 = (x0 - x2) / ((x1 - x0) * dx12);
  const double w2 = (x0 - x1) / ((x2 - x0) * (x2 - x1));
  return w0 * f0 + w1 * f1 + w2 * f2;
}

double first_derivative_1d(const DataVector& q, const std::vector<double>& grid,
                           const size_t idx, const size_t i, const size_t n,
                           const size_t stride,
                           const auto& make_stencil_index) {
  if (n < 2) {
    ERROR("Need at least 2 points to finite-difference.");
  }

  if (n == 2) {
    if (i == 0) {
      return (q[idx + stride] - q[idx]) / (grid[1] - grid[0]);
    }
    return (q[idx] - q[idx - stride]) / (grid[1] - grid[0]);
  }

  if (i == 0) {
    return first_derivative_3_point(q[idx], q[idx + stride],
                                    q[idx + 2 * stride], grid[0], grid[1],
                                    grid[2]);
  }
  if (i + 1 == n) {
    return first_derivative_3_point(q[idx], q[idx - stride],
                                    q[idx - 2 * stride], grid[n - 1],
                                    grid[n - 2], grid[n - 3]);
  }
  return first_derivative_3_point(q[idx], q[make_stencil_index(i - 1)],
                                  q[make_stencil_index(i + 1)], grid[i],
                                  grid[i - 1], grid[i + 1]);
}

double d_dYe(const DataVector& q, const std::vector<double>& Ye_grid,
             const size_t in, const size_t iT, const size_t iYe,
             const size_t nN, const size_t nYe) {
  const size_t idx = idx_of(in, iT, iYe, nN, nYe);
  constexpr size_t stride_Ye = 1;

  return first_derivative_1d(q, Ye_grid, idx, iYe, nYe, stride_Ye,
                             [in, iT, nN, nYe](const size_t stencil_iYe) {
                               return idx_of(in, iT, stencil_iYe, nN, nYe);
                             });
}

double d_dT(const DataVector& q, const std::vector<double>& T_grid,
            const size_t in, const size_t iT, const size_t iYe, const size_t nN,
            const size_t nT, const size_t nYe) {
  const size_t idx = idx_of(in, iT, iYe, nN, nYe);
  const size_t stride_T = nN * nYe;

  if (nT < 2) {
    ERROR("Need at least 2 T points to finite-difference in T.");
  }

  return first_derivative_1d(q, T_grid, idx, iT, nT, stride_T,
                             [in, iYe, nN, nYe](const size_t stencil_iT) {
                               return idx_of(in, stencil_iT, iYe, nN, nYe);
                             });
}

}  // namespace

DataVector compute_zeta_from_pressure_and_eps(
    const DataVector& pressure, const DataVector& specific_internal_energy,
    const std::vector<double>& temperature_grid,
    const std::vector<double>& electron_fraction_grid, const size_t nN,
    const size_t nT, const size_t nYe) {
  ASSERT(pressure.size() == nN * nT * nYe,
         "Pressure size does not match table dimensions.");
  ASSERT(specific_internal_energy.size() == pressure.size(),
         "Specific internal energy size does not match pressure size.");
  ASSERT(temperature_grid.size() == nT, "Temperature grid size mismatch.");
  ASSERT(electron_fraction_grid.size() == nYe, "Ye grid size mismatch.");

  DataVector zeta(pressure.size());
  constexpr double eps_T_floor = 1e-300;

  for (size_t in = 0; in < nN; ++in) {
    for (size_t iT = 0; iT < nT; ++iT) {
      for (size_t iYe = 0; iYe < nYe; ++iYe) {
        const size_t idx = idx_of(in, iT, iYe, nN, nYe);

        const double p_Ye =
            d_dYe(pressure, electron_fraction_grid, in, iT, iYe, nN, nYe);
        const double p_T =
            d_dT(pressure, temperature_grid, in, iT, iYe, nN, nT, nYe);
        const double eps_Ye =
            d_dYe(specific_internal_energy, electron_fraction_grid, in, iT, iYe,
                  nN, nYe);
        const double eps_T = d_dT(specific_internal_energy, temperature_grid,
                                  in, iT, iYe, nN, nT, nYe);

        if (std::abs(eps_T) < eps_T_floor) {
          ERROR("Encountered eps_T ~ 0 while computing zeta at idx=" << idx);
        }
        zeta[idx] = p_Ye - p_T * (eps_Ye / eps_T);
      }
    }
  }

  return zeta;
}

}  // namespace io
