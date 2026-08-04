// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/ComposeTableDerivatives.hpp"

#include <algorithm>
#include <cmath>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace io {

double log_log_derivative(const double q_minus, const double q_center,
                          const double q_plus, const double log_x_minus,
                          const double log_x_center, const double log_x_plus,
                          const bool at_low_boundary,
                          const bool at_high_boundary) {
  if (at_low_boundary) {
    if (q_center <= 0.0 or q_plus <= 0.0) {
      return 0.0;
    }
    return (std::log(q_plus) - std::log(q_center)) /
           (log_x_plus - log_x_center);
  }
  if (at_high_boundary) {
    if (q_center <= 0.0 or q_minus <= 0.0) {
      return 0.0;
    }
    return (std::log(q_center) - std::log(q_minus)) /
           (log_x_center - log_x_minus);
  }
  if (q_plus <= 0.0 or q_minus <= 0.0) {
    return 0.0;
  }
  return (std::log(q_plus) - std::log(q_minus)) / (log_x_plus - log_x_minus);
}

namespace {

size_t idx_of(const size_t in, const size_t iT, const size_t iYe,
              const size_t nN, const size_t nYe) {
  return (iT * nN + in) * nYe + iYe;
}

}  // namespace

DataVector compute_zeta_from_free_energy_derivatives(
    const DataVector& d2f_dt2, const DataVector& d2f_dt_dnb,
    const DataVector& d2f_dt_dye, const DataVector& d2f_dnb_dye,
    const DataVector& df_dye, const std::vector<double>& number_density_grid,
    const std::vector<double>& temperature_grid,
    const size_t number_density_points, const size_t temperature_points,
    const size_t electron_fraction_points) {
  // Local aliases for the densely-used grid sizes.
  const size_t nN = number_density_points;
  const size_t nT = temperature_points;
  const size_t nYe = electron_fraction_points;
  const size_t ntot = nN * nT * nYe;
  ASSERT(d2f_dt2.size() == ntot,
         "d2F/dT2 size " << d2f_dt2.size() << " does not match table size "
                         << ntot << ".");
  ASSERT(d2f_dt_dnb.size() == ntot,
         "d2F/dTdn_b size " << d2f_dt_dnb.size()
                            << " does not match table size " << ntot << ".");
  ASSERT(d2f_dt_dye.size() == ntot,
         "d2F/dTdY_e size " << d2f_dt_dye.size()
                            << " does not match table size " << ntot << ".");
  ASSERT(d2f_dnb_dye.size() == ntot,
         "d2F/dn_bdY_e size " << d2f_dnb_dye.size()
                              << " does not match table size " << ntot << ".");
  ASSERT(df_dye.size() == ntot, "dF/dY_e size " << df_dye.size()
                                                << " does not match table size "
                                                << ntot << ".");
  ASSERT(number_density_grid.size() == nN,
         "Number-density grid size " << number_density_grid.size()
                                     << " does not match nN " << nN << ".");
  ASSERT(temperature_grid.size() == nT,
         "Temperature grid size " << temperature_grid.size()
                                  << " does not match nT " << nT << ".");

  DataVector zeta(ntot);
  // eps_T = -T d2F/dT2
  // This floor only guards against unphysical/degenerate table entries (e.g. T
  // = 0).
  constexpr double eps_T_floor = 1e-300;

  for (size_t in = 0; in < nN; ++in) {
    const double nb = number_density_grid[in];
    const double nb_squared = nb * nb;
    for (size_t iT = 0; iT < nT; ++iT) {
      const double temperature = temperature_grid[iT];
      for (size_t iYe = 0; iYe < nYe; ++iYe) {
        const size_t idx = idx_of(in, iT, iYe, nN, nYe);

        // epsilon = F - T F_T, so its derivatives are
        //   eps_Ye = F_Ye - T F_{T,Ye},  eps_T = -T F_{T,T}.
        const double eps_Ye = df_dye[idx] - temperature * d2f_dt_dye[idx];
        const double eps_T = -temperature * d2f_dt2[idx];

        if (std::abs(eps_T) < eps_T_floor) {
          ERROR(
              "Encountered eps_T = -T d2F/dT2 ~ 0 while computing zeta at idx="
              << idx << "; cannot hold epsilon fixed when changing Y_e.");
        }

        // pressure p = n_b^2 F_{n_b}, so
        //   p_Ye = n_b^2 F_{n_b,Ye},  p_T = n_b^2 F_{n_b,T}.
        const double p_Ye = nb_squared * d2f_dnb_dye[idx];
        const double p_T = nb_squared * d2f_dt_dnb[idx];

        zeta[idx] = p_Ye - p_T * eps_Ye / eps_T;
      }
    }
  }

  return zeta;
}

DataVector compute_cs2_from_pressure_and_entropy(
    const DataVector& pressure, const DataVector& specific_entropy,
    const DataVector& specific_internal_energy,
    const std::vector<double>& number_density_grid,
    const std::vector<double>& temperature_grid, const double neutron_mass_mev,
    const size_t number_density_points, const size_t temperature_points,
    const size_t electron_fraction_points, const double cs2_floor) {
  const size_t nN = number_density_points;
  const size_t nT = temperature_points;
  const size_t nYe = electron_fraction_points;
  const size_t ntot = nN * nT * nYe;
  ASSERT(pressure.size() == ntot,
         "Pressure size " << pressure.size() << " does not match table size "
                          << ntot << ".");
  ASSERT(specific_entropy.size() == ntot, "Specific-entropy size "
                                              << specific_entropy.size()
                                              << " does not match table size "
                                              << ntot << ".");
  ASSERT(specific_internal_energy.size() == ntot,
         "Specific-internal-energy size " << specific_internal_energy.size()
                                          << " does not match table size "
                                          << ntot << ".");
  ASSERT(number_density_grid.size() == nN,
         "Number-density grid size " << number_density_grid.size()
                                     << " does not match nN " << nN << ".");
  ASSERT(temperature_grid.size() == nT,
         "Temperature grid size " << temperature_grid.size()
                                  << " does not match nT " << nT << ".");
  ASSERT(nN >= 2,
         "Need at least 2 number-density points to take a finite difference; "
         "got "
             << nN << ".");
  ASSERT(neutron_mass_mev > 0.0,
         "neutron_mass_mev must be positive, got " << neutron_mass_mev << ".");

  std::vector<double> log_nb(nN);
  for (size_t i = 0; i < nN; ++i) {
    ASSERT(number_density_grid[i] > 0.0,
           "Non-positive number density at grid index "
               << i << ": " << number_density_grid[i] << ".");
    log_nb[i] = std::log(number_density_grid[i]);
  }
  std::vector<double> log_t(nT);
  for (size_t i = 0; i < nT; ++i) {
    ASSERT(temperature_grid[i] > 0.0, "Non-positive temperature at grid index "
                                          << i << ": " << temperature_grid[i]
                                          << ".");
    log_t[i] = std::log(temperature_grid[i]);
  }

  DataVector cs2(ntot);

  // Smallest cs² safely sqrt'able by downstream code; only used as a
  // last-resort positive value when (dS/dT) is exactly zero or the FD on the
  // log of a non-positive quantity short-circuited to 0. cs2_floor (if
  // positive) is applied after this on top.
  constexpr double cs2_fallback_when_undefined = 1e-30;

  for (size_t iT = 0; iT < nT; ++iT) {
    const bool t_low = (iT == 0);
    const bool t_high = (iT + 1 == nT);
    for (size_t in = 0; in < nN; ++in) {
      const bool n_low = (in == 0);
      const bool n_high = (in + 1 == nN);
      const double nb = number_density_grid[in];
      for (size_t iYe = 0; iYe < nYe; ++iYe) {
        const size_t idx = idx_of(in, iT, iYe, nN, nYe);

        const double p_center = pressure[idx];
        const double s_center = specific_entropy[idx];

        // ∂(log P)/∂(log n_b), ∂(log S)/∂(log n_b) at fixed (iT, iYe).
        const double dlnp_dlnn = log_log_derivative(
            n_low ? 0.0 : pressure[idx_of(in - 1, iT, iYe, nN, nYe)], p_center,
            n_high ? 0.0 : pressure[idx_of(in + 1, iT, iYe, nN, nYe)],
            n_low ? 0.0 : log_nb[in - 1], log_nb[in],
            n_high ? 0.0 : log_nb[in + 1], n_low, n_high);
        const double dlns_dlnn = log_log_derivative(
            n_low ? 0.0 : specific_entropy[idx_of(in - 1, iT, iYe, nN, nYe)],
            s_center,
            n_high ? 0.0 : specific_entropy[idx_of(in + 1, iT, iYe, nN, nYe)],
            n_low ? 0.0 : log_nb[in - 1], log_nb[in],
            n_high ? 0.0 : log_nb[in + 1], n_low, n_high);

        const double dpdn = p_center * dlnp_dlnn / nb;
        const double dsdn = s_center * dlns_dlnn / nb;

        // h = m_n (1 + ε) + p / n_b, MeV per baryon.
        const double h =
            neutron_mass_mev * (1.0 + specific_internal_energy[idx]) +
            p_center / nb;

        double value = 0.0;
        if (nT > 1) {
          const double dlnp_dlnt = log_log_derivative(
              t_low ? 0.0 : pressure[idx_of(in, iT - 1, iYe, nN, nYe)],
              p_center,
              t_high ? 0.0 : pressure[idx_of(in, iT + 1, iYe, nN, nYe)],
              t_low ? 0.0 : log_t[iT - 1], log_t[iT],
              t_high ? 0.0 : log_t[iT + 1], t_low, t_high);
          const double dlns_dlnt = log_log_derivative(
              t_low ? 0.0 : specific_entropy[idx_of(in, iT - 1, iYe, nN, nYe)],
              s_center,
              t_high ? 0.0 : specific_entropy[idx_of(in, iT + 1, iYe, nN, nYe)],
              t_low ? 0.0 : log_t[iT - 1], log_t[iT],
              t_high ? 0.0 : log_t[iT + 1], t_low, t_high);

          const double temperature = temperature_grid[iT];
          const double dpdt = p_center * dlnp_dlnt / temperature;
          const double dsdt = s_center * dlns_dlnt / temperature;

          // dS/dT is the (per-baryon) heat capacity at constant volume; it is
          // strictly positive for stable matter. If the FD returned 0 (because
          // log_log_derivative short-circuited on a non-positive S, or the FD
          // is numerically zero), we cannot form (dS/dn)/(dS/dT). Fall back to
          // the cold formula for this point.
          if (dsdt == 0.0) {
            value = dpdn / h;
          } else {
            value = (dpdn - (dsdn / dsdt) * dpdt) / h;
          }
        } else {
          value = dpdn / h;
        }

        // Treat NaNs as undefined and use the safe fallback.
        if (not std::isfinite(value)) {
          value = cs2_fallback_when_undefined;
        }
        cs2[idx] = (cs2_floor > 0.0) ? std::max(value, cs2_floor) : value;
      }
    }
  }

  return cs2;
}

DataVector compute_kappa_from_pressure_and_energy(
    const DataVector& pressure, const DataVector& specific_internal_energy,
    const std::vector<double>& temperature_grid, const double neutron_mass_mev,
    const size_t number_density_points, const size_t temperature_points,
    const size_t electron_fraction_points, const double kappa_floor) {
  const size_t nN = number_density_points;
  const size_t nT = temperature_points;
  const size_t nYe = electron_fraction_points;
  const size_t ntot = nN * nT * nYe;
  ASSERT(pressure.size() == ntot,
         "Pressure size " << pressure.size() << " does not match table size "
                          << ntot << ".");
  ASSERT(specific_internal_energy.size() == ntot,
         "Specific-internal-energy size " << specific_internal_energy.size()
                                          << " does not match table size "
                                          << ntot << ".");
  ASSERT(temperature_grid.size() == nT,
         "Temperature grid size " << temperature_grid.size()
                                  << " does not match nT " << nT << ".");
  ASSERT(neutron_mass_mev > 0.0,
         "neutron_mass_mev must be positive, got " << neutron_mass_mev << ".");

  // Single-temperature table: kappa is undefined from a T-axis FD. Match the
  // existing "kappa missing" fallback in ConvertComposeTable and return zeros
  // (or the floor, whichever is larger).
  if (nT < 2) {
    const double fill = (kappa_floor > 0.0) ? kappa_floor : 0.0;
    DataVector result(ntot, fill);
    return result;
  }

  for (size_t i = 0; i < nT; ++i) {
    ASSERT(temperature_grid[i] > 0.0, "Non-positive temperature at grid index "
                                          << i << ": " << temperature_grid[i]
                                          << ".");
  }

  DataVector kappa(ntot);
  constexpr double kappa_fallback_when_undefined = 0.0;

  // For the T-axis derivatives we use log on p (which is strictly positive
  // and spans many orders of magnitude) and linear on ε (which crosses zero
  // for cold matter and is essentially constant at low T — log(1+ε) would
  // yield a numerically-vanishing denominator and a spurious blow-up).
  // After the common /T cancellation in (∂p/∂T)/(∂ε/∂T) and the conversion
  // ∂ε/∂E = 1/m_n, the formula reduces to
  //   κ = (p / m_n) · [log p(T_+) − log p(T_−)] / [ε(T_+) − ε(T_−)],
  // independent of the actual T spacing. The output matches CompOSE Q11
  // (dp/dE | n_b, in fm^-3 when called on raw CompOSE data).
  const double inv_m_n = 1.0 / neutron_mass_mev;

  for (size_t iT = 0; iT < nT; ++iT) {
    const bool t_low = (iT == 0);
    const bool t_high = (iT + 1 == nT);
    const size_t iT_lo = t_low ? iT : iT - 1;
    const size_t iT_hi = t_high ? iT : iT + 1;
    for (size_t in = 0; in < nN; ++in) {
      for (size_t iYe = 0; iYe < nYe; ++iYe) {
        const size_t idx = idx_of(in, iT, iYe, nN, nYe);
        const size_t idx_lo = idx_of(in, iT_lo, iYe, nN, nYe);
        const size_t idx_hi = idx_of(in, iT_hi, iYe, nN, nYe);

        const double p_center = pressure[idx];
        const double p_lo = pressure[idx_lo];
        const double p_hi = pressure[idx_hi];
        const double eps_lo = specific_internal_energy[idx_lo];
        const double eps_hi = specific_internal_energy[idx_hi];

        double value = 0.0;
        if (p_lo <= 0.0 or p_hi <= 0.0 or p_center <= 0.0) {
          value = kappa_fallback_when_undefined;
        } else {
          const double dlogp = std::log(p_hi) - std::log(p_lo);
          const double deps = eps_hi - eps_lo;
          // (∂ε/∂T) at fixed (n_b, Y_e) is proportional to the specific heat
          // at constant volume; strictly positive for any stable EoS. A
          // vanishing deps means the FD couldn't resolve it (degenerate
          // matter, or the table is constant in T at this point); treat as
          // undefined.
          if (deps == 0.0) {
            value = kappa_fallback_when_undefined;
          } else {
            value = (p_center * inv_m_n) * dlogp / deps;
          }
        }

        if (not std::isfinite(value)) {
          value = kappa_fallback_when_undefined;
        }
        kappa[idx] = (kappa_floor > 0.0) ? std::max(value, kappa_floor) : value;
      }
    }
  }

  return kappa;
}

}  // namespace io
