// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated1D.hpp"

#include <memory>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic2D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace EquationsOfState {

template <bool IsRelativistic>
Tabulated1D<IsRelativistic>::Tabulated1D(const size_t number_of_grid_points,
                                         const bool log_spacing)
    : number_of_grid_points_(number_of_grid_points),
      log_spacing_(log_spacing) {}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     Tabulated1D<IsRelativistic>, double, 1)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     Tabulated1D<IsRelativistic>, DataVector, 1)

template <bool IsRelativistic>
bool Tabulated1D<IsRelativistic>::operator==(
    const Tabulated1D<IsRelativistic>& rhs) const {
  return (number_of_grid_points_ == rhs.number_of_grid_points_) and
         (log_spacing_ == rhs.log_spacing_);
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
  p | number_of_grid_points_;
  p | log_spacing_;
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
