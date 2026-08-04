// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated1D.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/EosTable.hpp"
#include "IO/H5/File.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic2D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

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

  // Build the internal log(rho_geom) grid. The h5 stores n_b in fm^-3
  // (linear or log-spaced per the subfile metadata); we convert to
  // geometric units by multiplying by nb_fm3_to_geom, then work in
  // log-space for uniform-in-log interpolation later.
  constexpr double nb_fm3_to_geom = hydro::units::nuclear::neutron_mass /
                                    hydro::units::nuclear::pressure_unit;
  const double log_nb_fm3_to_geom = std::log(nb_fm3_to_geom);

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
    // Non-log-spaced input grid: keep linear n_b spacing but store log(rho)
    // for consistency of the query-time transforms. This branch is
    // supported for completeness; log spacing is the recommended path.
    const double dnb =
        (bounds[1] - bounds[0]) / static_cast<double>(num_grid_points - 1);
    for (size_t i = 0; i < num_grid_points; ++i) {
      const double nb = bounds[0] + static_cast<double>(i) * dnb;
      log_rho_grid_[i] = std::log(nb_fm3_to_geom * nb);
    }
  }

  // Read datasets from the h5 subfile. Same names the offline converter
  // (Stage 2b) writes.
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

  constexpr double press_MeV_to_geom =
      1.0 / hydro::units::nuclear::pressure_unit;

  pressure_.resize(num_grid_points);
  specific_internal_energy_.resize(num_grid_points);
  specific_enthalpy_.resize(num_grid_points);
  chi_slope_.resize(num_grid_points);
  for (size_t i = 0; i < num_grid_points; ++i) {
    pressure_[i] = press_MeV_to_geom * pressure_data[i];
    specific_internal_energy_[i] = eps_data[i];
    chi_slope_[i] = chi_slope_data[i];
    const double rho_geom = std::exp(log_rho_grid_[i]);
    if constexpr (IsRelativistic) {
      specific_enthalpy_[i] =
          1.0 + specific_internal_energy_[i] + pressure_[i] / rho_geom;
    } else {
      specific_enthalpy_[i] =
          specific_internal_energy_[i] + pressure_[i] / rho_geom;
    }
  }

  // Strict-monotonicity checks. p and h must be strictly increasing in rho
  // for log-space pressure interpolation and h -> rho inversion to work.
  for (size_t i = 1; i < num_grid_points; ++i) {
    if (not(pressure_[i] > pressure_[i - 1])) {
      ERROR("Tabulated1D: loaded pressure is not strictly monotonic. p["
            << i - 1 << "] = " << pressure_[i - 1] << ", p[" << i
            << "] = " << pressure_[i] << " at log(rho)[" << i
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
}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     Tabulated1D<IsRelativistic>, double, 1)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     Tabulated1D<IsRelativistic>, DataVector, 1)

template <bool IsRelativistic>
bool Tabulated1D<IsRelativistic>::operator==(
    const Tabulated1D<IsRelativistic>& rhs) const {
  return log_rho_grid_ == rhs.log_rho_grid_ and pressure_ == rhs.pressure_ and
         specific_internal_energy_ == rhs.specific_internal_energy_ and
         specific_enthalpy_ == rhs.specific_enthalpy_ and
         chi_slope_ == rhs.chi_slope_;
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
  p | pressure_;
  p | specific_internal_energy_;
  p | specific_enthalpy_;
  p | chi_slope_;
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
  return specific_internal_energy_.empty() ? 0.0
                                           : specific_internal_energy_.front();
}

template <bool IsRelativistic>
double Tabulated1D<IsRelativistic>::specific_internal_energy_upper_bound()
    const {
  return specific_internal_energy_.empty() ? std::numeric_limits<double>::max()
                                           : specific_internal_energy_.back();
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> Tabulated1D<IsRelativistic>::pressure_from_density_impl(
    const Scalar<DataType>& /*rest_mass_density*/) const {
  ERROR("Tabulated1D::pressure_from_density is not implemented yet.");
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
Scalar<DataType>
Tabulated1D<IsRelativistic>::specific_internal_energy_from_density_impl(
    const Scalar<DataType>& /*rest_mass_density*/) const {
  ERROR(
      "Tabulated1D::specific_internal_energy_from_density is not implemented "
      "yet.");
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> Tabulated1D<IsRelativistic>::chi_from_density_impl(
    const Scalar<DataType>& /*rest_mass_density*/) const {
  ERROR("Tabulated1D::chi_from_density is not implemented yet.");
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
Tabulated1D<IsRelativistic>::kappa_times_p_over_rho_squared_from_density_impl(
    const Scalar<DataType>& /*rest_mass_density*/) const {
  ERROR(
      "Tabulated1D::kappa_times_p_over_rho_squared_from_density is not "
      "implemented yet.");
}
}  // namespace EquationsOfState

template class EquationsOfState::Tabulated1D<true>;
template class EquationsOfState::Tabulated1D<false>;
