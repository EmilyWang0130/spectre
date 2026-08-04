// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated1D.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/EosTable.hpp"
#include "IO/H5/File.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic2D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace EquationsOfState {

template <bool IsRelativistic>
Tabulated1D<IsRelativistic>::Tabulated1D(const std::string& filename,
                                         const std::string& subfilename) {
  const h5::H5File<h5::AccessType::ReadOnly> eos_file{filename};
  const auto& spectre_eos = eos_file.get<h5::EosTable>("/" + subfilename);
  initialize(spectre_eos);
}

template <bool IsRelativistic>
Tabulated1D<IsRelativistic>::Tabulated1D(const h5::EosTable& spectre_eos) {
  initialize(spectre_eos);
}

template <bool IsRelativistic>
Tabulated1D<IsRelativistic>::Tabulated1D(const Tabulated1D& rhs)
    : EquationOfState<IsRelativistic, 1>(rhs),
      log_rho_grid_(rhs.log_rho_grid_),
      table_data_(rhs.table_data_),
      specific_enthalpy_(rhs.specific_enthalpy_),
      energy_shift_(rhs.energy_shift_) {
  initialize_interpolator();
}

template <bool IsRelativistic>
Tabulated1D<IsRelativistic>& Tabulated1D<IsRelativistic>::operator=(
    const Tabulated1D& rhs) {
  if (this != &rhs) {
    EquationOfState<IsRelativistic, 1>::operator=(rhs);
    log_rho_grid_ = rhs.log_rho_grid_;
    table_data_ = rhs.table_data_;
    specific_enthalpy_ = rhs.specific_enthalpy_;
    energy_shift_ = rhs.energy_shift_;
    initialize_interpolator();
  }
  return *this;
}

template <bool IsRelativistic>
void Tabulated1D<IsRelativistic>::initialize(const h5::EosTable& spectre_eos) {
  const auto& names = spectre_eos.independent_variable_names();
  if (names.size() != 1) {
    ERROR(
        "Tabulated1D expects an EosTable with exactly one independent "
        "variable (the rest mass number density in fm^-3), but the h5 "
        "subfile has "
        << names.size() << " independent variables.");
  }
  if (names[0] != "number density") {
    ERROR(
        "Tabulated1D expects the sole independent variable to be named "
        "\"number density\", but the h5 subfile has \""
        << names[0] << "\".");
  }

  const auto bounds = spectre_eos.independent_variable_bounds()[0];
  const bool grid_uses_log_spacing =
      spectre_eos.independent_variable_uses_log_spacing()[0];
  const size_t num_grid_points =
      spectre_eos.independent_variable_number_of_points()[0];
  if (num_grid_points < 2) {
    ERROR("Tabulated1D needs at least 2 grid points, got " << num_grid_points
                                                           << ".");
  }

  constexpr double nb_fm3_to_geom = hydro::units::nuclear::neutron_mass /
                                    hydro::units::nuclear::pressure_unit;
  const double log_nb_fm3_to_geom = std::log(nb_fm3_to_geom);
  constexpr double press_MeV_to_geom =
      1.0 / hydro::units::nuclear::pressure_unit;

  log_rho_grid_.resize(num_grid_points);
  if (grid_uses_log_spacing) {
    const double log_lo = std::log(bounds[0]);
    const double log_hi = std::log(bounds[1]);
    const double dlog =
        (log_hi - log_lo) / static_cast<double>(num_grid_points - 1);
    for (size_t i = 0; i < num_grid_points; ++i) {
      log_rho_grid_[i] =
          log_lo + static_cast<double>(i) * dlog + log_nb_fm3_to_geom;
    }
  } else {
    const double dnb =
        (bounds[1] - bounds[0]) / static_cast<double>(num_grid_points - 1);
    for (size_t i = 0; i < num_grid_points; ++i) {
      const double nb = bounds[0] + static_cast<double>(i) * dnb;
      log_rho_grid_[i] = std::log(nb_fm3_to_geom * nb);
    }
  }

  const auto pressure_data = spectre_eos.read_quantity("pressure");
  const auto eps_data = spectre_eos.read_quantity("specific internal energy");
  const auto chi_slope_data = spectre_eos.read_quantity("chi slope");
  if (pressure_data.size() != num_grid_points or
      eps_data.size() != num_grid_points or
      chi_slope_data.size() != num_grid_points) {
    ERROR("Tabulated1D h5 dataset length mismatch: expected "
          << num_grid_points << " points, got pressure=" << pressure_data.size()
          << ", specific internal energy=" << eps_data.size()
          << ", chi slope=" << chi_slope_data.size() << ".");
  }

  // Determine energy_shift so that (eps - energy_shift) is strictly
  // positive across the table. Matches Tabulated3D::initialize:
  // if eps_min < 0 the shift is 2 * eps_min (i.e., more negative), else 0.
  double eps_min = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < num_grid_points; ++i) {
    eps_min = std::min(eps_min, eps_data[i]);
  }
  energy_shift_ = (eps_min < 0.0) ? 2.0 * eps_min : 0.0;

  // Fill packed [log_pressure, log(eps - energy_shift), chi_slope]
  // per grid point.
  table_data_.assign(num_grid_points * NumberOfVars, 0.0);
  specific_enthalpy_.resize(num_grid_points);
  for (size_t i = 0; i < num_grid_points; ++i) {
    const double p_geom = press_MeV_to_geom * pressure_data[i];
    const double eps = eps_data[i];
    const double rho_geom = std::exp(log_rho_grid_[i]);
    table_data_[i * NumberOfVars + LogPressure] = std::log(p_geom);
    table_data_[i * NumberOfVars + LogShiftedEpsilon] =
        std::log(eps - energy_shift_);
    table_data_[i * NumberOfVars + ChiSlope] = chi_slope_data[i];
    if constexpr (IsRelativistic) {
      specific_enthalpy_[i] = 1.0 + eps + p_geom / rho_geom;
    } else {
      specific_enthalpy_[i] = eps + p_geom / rho_geom;
    }
  }

  // Strict-monotonicity checks. p and h must be strictly increasing in rho
  // for log-space pressure interpolation and h -> rho inversion to work.
  for (size_t i = 1; i < num_grid_points; ++i) {
    if (not(pressure_data[i] > pressure_data[i - 1])) {
      ERROR("Tabulated1D: loaded pressure is not strictly monotonic. p["
            << i - 1 << "] = " << pressure_data[i - 1] << ", p[" << i
            << "] = " << pressure_data[i] << " at log(rho)[" << i
            << "] = " << log_rho_grid_[i]
            << ". Log-space interpolation on p requires positive, "
               "strictly-monotonic p.");
    }
    if (not(specific_enthalpy_[i] > specific_enthalpy_[i - 1])) {
      ERROR(
          "Tabulated1D: computed specific enthalpy is not strictly "
          "monotonic. h["
          << i - 1 << "] = " << specific_enthalpy_[i - 1] << ", h[" << i
          << "] = " << specific_enthalpy_[i] << " at log(rho)[" << i
          << "] = " << log_rho_grid_[i]
          << ". The h(rho) -> rho(h) inversion requires monotonic h; "
             "check the input h5 for physical consistency.");
    }
  }

  initialize_interpolator();
}

template <bool IsRelativistic>
void Tabulated1D<IsRelativistic>::initialize_interpolator() {
  if (log_rho_grid_.size() < 2 or
      table_data_.size() != log_rho_grid_.size() * NumberOfVars) {
    return;
  }
  Index<1> num_x_points;
  num_x_points[0] = log_rho_grid_.size();
  const std::array<gsl::span<const double>, 1> independent_data_view{
      gsl::span<const double>{log_rho_grid_.data(), log_rho_grid_.size()}};
  interpolator_ = intrp::UniformMultiLinearSpanInterpolation<1, NumberOfVars>(
      independent_data_view, {table_data_.data(), table_data_.size()},
      num_x_points);
}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     Tabulated1D<IsRelativistic>, double, 1)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     Tabulated1D<IsRelativistic>, DataVector, 1)

template <bool IsRelativistic>
bool Tabulated1D<IsRelativistic>::operator==(
    const Tabulated1D<IsRelativistic>& rhs) const {
  return log_rho_grid_ == rhs.log_rho_grid_ and
         table_data_ == rhs.table_data_ and
         specific_enthalpy_ == rhs.specific_enthalpy_ and
         energy_shift_ == rhs.energy_shift_;
}

template <bool IsRelativistic>
bool Tabulated1D<IsRelativistic>::operator!=(
    const Tabulated1D<IsRelativistic>& rhs) const {
  return not(*this == rhs);
}

template <bool IsRelativistic>
bool Tabulated1D<IsRelativistic>::is_equal(
    const EquationOfState<IsRelativistic, 1>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const Tabulated1D<IsRelativistic>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <bool IsRelativistic>
std::unique_ptr<EquationOfState<IsRelativistic, 1>>
Tabulated1D<IsRelativistic>::get_clone() const {
  auto clone = std::make_unique<Tabulated1D<IsRelativistic>>(*this);
  return std::unique_ptr<EquationOfState<IsRelativistic, 1>>(std::move(clone));
}

template <bool IsRelativistic>
std::unique_ptr<EquationOfState<IsRelativistic, 3>>
Tabulated1D<IsRelativistic>::promote_to_3d_eos() const {
  return std::make_unique<Barotropic3D<Tabulated1D<IsRelativistic>>>(*this);
}

template <bool IsRelativistic>
std::unique_ptr<EquationOfState<IsRelativistic, 2>>
Tabulated1D<IsRelativistic>::promote_to_2d_eos() const {
  return std::make_unique<Barotropic2D<Tabulated1D<IsRelativistic>>>(*this);
}

template <bool IsRelativistic>
Tabulated1D<IsRelativistic>::Tabulated1D(CkMigrateMessage* msg)
    : EquationOfState<IsRelativistic, 1>(msg) {}

template <bool IsRelativistic>
void Tabulated1D<IsRelativistic>::pup(PUP::er& p) {
  EquationOfState<IsRelativistic, 1>::pup(p);
  p | log_rho_grid_;
  p | table_data_;
  p | specific_enthalpy_;
  p | energy_shift_;
  if (p.isUnpacking()) {
    initialize_interpolator();
  }
}

template <bool IsRelativistic>
double Tabulated1D<IsRelativistic>::rest_mass_density_lower_bound() const {
  return log_rho_grid_.empty() ? 0.0 : std::exp(log_rho_grid_.front());
}

template <bool IsRelativistic>
double Tabulated1D<IsRelativistic>::rest_mass_density_upper_bound() const {
  return log_rho_grid_.empty() ? std::numeric_limits<double>::max()
                               : std::exp(log_rho_grid_.back());
}

template <bool IsRelativistic>
double Tabulated1D<IsRelativistic>::specific_enthalpy_lower_bound() const {
  if (specific_enthalpy_.empty()) {
    return IsRelativistic ? 1.0 : 0.0;
  }
  return specific_enthalpy_.front();
}

template <bool IsRelativistic>
double Tabulated1D<IsRelativistic>::specific_internal_energy_lower_bound()
    const {
  if (table_data_.size() < NumberOfVars) {
    return 0.0;
  }
  return std::exp(table_data_[LogShiftedEpsilon]) + energy_shift_;
}

template <bool IsRelativistic>
double Tabulated1D<IsRelativistic>::specific_internal_energy_upper_bound()
    const {
  if (table_data_.size() < NumberOfVars) {
    return std::numeric_limits<double>::max();
  }
  const size_t last = log_rho_grid_.size() - 1;
  return std::exp(table_data_[last * NumberOfVars + LogShiftedEpsilon]) +
         energy_shift_;
}

namespace {
// Clamp rho to the table's density range and take log. Mirrors what
// Tabulated3D does in convert_to_table_quantities.
double clamped_log_rho(const double rho, const double log_rho_lo,
                       const double log_rho_hi) {
  return std::min(std::max(std::log(rho), log_rho_lo), log_rho_hi);
}
}  // namespace

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> Tabulated1D<IsRelativistic>::pressure_from_density_impl(
    const Scalar<DataType>& rest_mass_density) const {
  const double log_rho_lo = log_rho_grid_.front();
  const double log_rho_hi = log_rho_grid_.back();
  Scalar<DataType> result =
      make_with_value<Scalar<DataType>>(get(rest_mass_density), 0.0);
  if constexpr (std::is_same_v<DataType, double>) {
    const double log_rho =
        clamped_log_rho(get(rest_mass_density), log_rho_lo, log_rho_hi);
    const auto weights = interpolator_.get_weights(log_rho);
    get(result) =
        std::exp(interpolator_.template interpolate<LogPressure>(weights)[0]);
  } else {
    const auto& rho_arr = get(rest_mass_density);
    for (size_t i = 0; i < rho_arr.size(); ++i) {
      const double log_rho =
          clamped_log_rho(rho_arr[i], log_rho_lo, log_rho_hi);
      const auto weights = interpolator_.get_weights(log_rho);
      get(result)[i] =
          std::exp(interpolator_.template interpolate<LogPressure>(weights)[0]);
    }
  }
  return result;
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
Tabulated1D<IsRelativistic>::specific_internal_energy_from_density_impl(
    const Scalar<DataType>& rest_mass_density) const {
  const double log_rho_lo = log_rho_grid_.front();
  const double log_rho_hi = log_rho_grid_.back();
  Scalar<DataType> result =
      make_with_value<Scalar<DataType>>(get(rest_mass_density), 0.0);
  if constexpr (std::is_same_v<DataType, double>) {
    const double log_rho =
        clamped_log_rho(get(rest_mass_density), log_rho_lo, log_rho_hi);
    const auto weights = interpolator_.get_weights(log_rho);
    get(result) =
        std::exp(
            interpolator_.template interpolate<LogShiftedEpsilon>(weights)[0]) +
        energy_shift_;
  } else {
    const auto& rho_arr = get(rest_mass_density);
    for (size_t i = 0; i < rho_arr.size(); ++i) {
      const double log_rho =
          clamped_log_rho(rho_arr[i], log_rho_lo, log_rho_hi);
      const auto weights = interpolator_.get_weights(log_rho);
      get(result)[i] =
          std::exp(interpolator_.template interpolate<LogShiftedEpsilon>(
              weights)[0]) +
          energy_shift_;
    }
  }
  return result;
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
Tabulated1D<IsRelativistic>::rest_mass_density_from_enthalpy_impl(
    const Scalar<DataType>& /*specific_enthalpy*/) const {
  ERROR("Tabulated1D::rest_mass_density_from_enthalpy is not implemented yet.");
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> Tabulated1D<IsRelativistic>::chi_from_density_impl(
    const Scalar<DataType>& rest_mass_density) const {
  // chi = dp/drho at fixed epsilon. We stored the dimensionless log-log
  // slope chi_slope = d(ln p)/d(ln rho) in the packed table and
  // reconstruct chi = (p_geom / rho_geom) * chi_slope at query time.
  // The rho_geom used here is the CLAMPED value (matching the density
  // clamped in the interpolator lookup), not the raw query — otherwise
  // out-of-range queries would return chi with an inconsistent
  // pressure/density ratio.
  const double log_rho_lo = log_rho_grid_.front();
  const double log_rho_hi = log_rho_grid_.back();
  Scalar<DataType> result =
      make_with_value<Scalar<DataType>>(get(rest_mass_density), 0.0);
  if constexpr (std::is_same_v<DataType, double>) {
    const double log_rho =
        clamped_log_rho(get(rest_mass_density), log_rho_lo, log_rho_hi);
    const auto weights = interpolator_.get_weights(log_rho);
    const auto interpolated =
        interpolator_.template interpolate<LogPressure, ChiSlope>(weights);
    const double p_geom = std::exp(interpolated[0]);
    const double slope = interpolated[1];
    get(result) = (p_geom / std::exp(log_rho)) * slope;
  } else {
    const auto& rho_arr = get(rest_mass_density);
    for (size_t i = 0; i < rho_arr.size(); ++i) {
      const double log_rho =
          clamped_log_rho(rho_arr[i], log_rho_lo, log_rho_hi);
      const auto weights = interpolator_.get_weights(log_rho);
      const auto interpolated =
          interpolator_.template interpolate<LogPressure, ChiSlope>(weights);
      const double p_geom = std::exp(interpolated[0]);
      const double slope = interpolated[1];
      get(result)[i] = (p_geom / std::exp(log_rho)) * slope;
    }
  }
  return result;
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
Tabulated1D<IsRelativistic>::kappa_times_p_over_rho_squared_from_density_impl(
    const Scalar<DataType>& rest_mass_density) const {
  // Barotropic: p = p(rho), so kappa = dp/deps|_rho = 0.
  return make_with_value<Scalar<DataType>>(get(rest_mass_density), 0.0);
}
}  // namespace EquationsOfState

template class EquationsOfState::Tabulated1D<true>;
template class EquationsOfState::Tabulated1D<false>;
