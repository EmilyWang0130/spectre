// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/Tabulated3DShockTube.hpp"

#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace {
template <typename DataType>
Scalar<DataType> compute_piecewise(const tnsr::I<DataType, 3>& x,
                                   const double discontinuity_position,
                                   const double left_value,
                                   const double right_value) {
  return Scalar<DataType>(
      left_value - (left_value - right_value) *
                       step_function(get<0>(x) - discontinuity_position));
}

template <typename DataType>
tnsr::I<DataType, 3> compute_piecewise_vector(
    const tnsr::I<DataType, 3>& x, const double discontinuity_position,
    const std::array<double, 3>& left_value,
    const std::array<double, 3>& right_value) {
  return tnsr::I<DataType, 3>{
      {{left_value[0] - (left_value[0] - right_value[0]) *
                            step_function(get<0>(x) - discontinuity_position),
        left_value[1] - (left_value[1] - right_value[1]) *
                            step_function(get<0>(x) - discontinuity_position),
        left_value[2] -
            (left_value[2] - right_value[2]) *
                step_function(get<0>(x) - discontinuity_position)}}};
}

template <typename DataType>
DataType compute_ye_hat(const tnsr::I<DataType, 3>& x,
                        const double ye_perturbation_center,
                        const double ye_perturbation_half_width_left,
                        const double ye_perturbation_half_width_right) {
  const auto& x_coord = get<0>(x);
  return step_function(x_coord - (ye_perturbation_center -
                                  ye_perturbation_half_width_left)) -
         step_function(x_coord - (ye_perturbation_center +
                                  ye_perturbation_half_width_right));
}
}  // namespace

namespace grmhd::AnalyticData {

Tabulated3DShockTube::Tabulated3DShockTube(
    const std::string& table_filename, const std::string& table_subfilename,
    const double left_rest_mass_density, const double right_rest_mass_density,
    const double left_temperature, const double right_temperature,
    const double left_electron_fraction, const double right_electron_fraction,
    const std::array<double, 3>& left_spatial_velocity,
    const std::array<double, 3>& right_spatial_velocity,
    const std::array<double, 3>& left_magnetic_field,
    const std::array<double, 3>& right_magnetic_field,
    const double discontinuity_position, const double ye_perturbation_value,
    const double ye_perturbation_center,
    const double ye_perturbation_half_width_left,
    const double ye_perturbation_half_width_right,
    std::optional<double> ye_perturbation_left_temperature,
    std::optional<double> ye_perturbation_right_temperature)
    : equation_of_state_(table_filename, table_subfilename),
      left_rest_mass_density_(left_rest_mass_density),
      right_rest_mass_density_(right_rest_mass_density),
      left_temperature_(left_temperature),
      right_temperature_(right_temperature),
      left_electron_fraction_(left_electron_fraction),
      right_electron_fraction_(right_electron_fraction),
      left_spatial_velocity_(left_spatial_velocity),
      right_spatial_velocity_(right_spatial_velocity),
      left_magnetic_field_(left_magnetic_field),
      right_magnetic_field_(right_magnetic_field),
      discontinuity_position_(discontinuity_position),
      ye_perturbation_value_(ye_perturbation_value),
      ye_perturbation_center_(ye_perturbation_center),
      ye_perturbation_half_width_left_(ye_perturbation_half_width_left),
      ye_perturbation_half_width_right_(ye_perturbation_half_width_right),
      ye_perturbation_left_temperature_(ye_perturbation_left_temperature),
      ye_perturbation_right_temperature_(ye_perturbation_right_temperature) {}

std::unique_ptr<evolution::initial_data::InitialData>
Tabulated3DShockTube::get_clone() const {
  return std::make_unique<Tabulated3DShockTube>(*this);
}

Tabulated3DShockTube::Tabulated3DShockTube(CkMigrateMessage* msg)
    : InitialData(msg) {}

void Tabulated3DShockTube::pup(PUP::er& p) {
  InitialData::pup(p);
  p | equation_of_state_;
  p | background_spacetime_;
  p | left_rest_mass_density_;
  p | right_rest_mass_density_;
  p | left_temperature_;
  p | right_temperature_;
  p | left_electron_fraction_;
  p | right_electron_fraction_;
  p | left_spatial_velocity_;
  p | right_spatial_velocity_;
  p | left_magnetic_field_;
  p | right_magnetic_field_;
  p | discontinuity_position_;
  p | ye_perturbation_value_;
  p | ye_perturbation_center_;
  p | ye_perturbation_half_width_left_;
  p | ye_perturbation_half_width_right_;
  p | ye_perturbation_left_temperature_;
  p | ye_perturbation_right_temperature_;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
Tabulated3DShockTube::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const {
  return compute_piecewise(x, discontinuity_position_, left_rest_mass_density_,
                           right_rest_mass_density_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>
Tabulated3DShockTube::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/) const {
  auto electron_fraction =
      compute_piecewise(x, discontinuity_position_, left_electron_fraction_,
                        right_electron_fraction_);
  // Override Y_e inside the (asymmetric) top-hat
  // [center - half_width_left, center + half_width_right] with the configured
  // constant ye_perturbation_value_, regardless of which side of the
  // discontinuity the cell sits on. Setting either half-width to 0 disables
  // that side of the hat (useful for isolating individual hat edges from the
  // contact discontinuity). The hat indicator is the difference of two
  // Heaviside steps (step_function(y) is 1 for y >= 0).
  const auto hat = compute_ye_hat(x, ye_perturbation_center_,
                                  ye_perturbation_half_width_left_,
                                  ye_perturbation_half_width_right_);
  get(electron_fraction) =
      ye_perturbation_value_ * hat + get(electron_fraction) * (1.0 - hat);
  return electron_fraction;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Temperature<DataType>>
Tabulated3DShockTube::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/) const {
  auto temperature = compute_piecewise(x, discontinuity_position_,
                                       left_temperature_, right_temperature_);
  if (ye_perturbation_left_temperature_.has_value() or
      ye_perturbation_right_temperature_.has_value()) {
    const auto hat = compute_ye_hat(x, ye_perturbation_center_,
                                    ye_perturbation_half_width_left_,
                                    ye_perturbation_half_width_right_);
    const auto right_side = step_function(get<0>(x) - discontinuity_position_);
    const auto left_hat = hat * (1.0 - right_side);
    const auto right_hat = hat * right_side;
    if (ye_perturbation_left_temperature_.has_value()) {
      get(temperature) = *ye_perturbation_left_temperature_ * left_hat +
                         get(temperature) * (1.0 - left_hat);
    }
    if (ye_perturbation_right_temperature_.has_value()) {
      get(temperature) = *ye_perturbation_right_temperature_ * right_hat +
                         get(temperature) * (1.0 - right_hat);
    }
  }
  return temperature;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
Tabulated3DShockTube::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const {
  return equation_of_state_
      .specific_internal_energy_from_density_and_temperature(
          get<hydro::Tags::RestMassDensity<DataType>>(variables(
              x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
          get<hydro::Tags::Temperature<DataType>>(
              variables(x, tmpl::list<hydro::Tags::Temperature<DataType>>{})),
          get<hydro::Tags::ElectronFraction<DataType>>(variables(
              x, tmpl::list<hydro::Tags::ElectronFraction<DataType>>{})));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>
Tabulated3DShockTube::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const {
  return equation_of_state_.pressure_from_density_and_temperature(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::Temperature<DataType>>(
          variables(x, tmpl::list<hydro::Tags::Temperature<DataType>>{})),
      get<hydro::Tags::ElectronFraction<DataType>>(
          variables(x, tmpl::list<hydro::Tags::ElectronFraction<DataType>>{})));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
Tabulated3DShockTube::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const {
  return compute_piecewise_vector(x, discontinuity_position_,
                                  left_spatial_velocity_,
                                  right_spatial_velocity_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
Tabulated3DShockTube::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const {
  return compute_piecewise_vector(x, discontinuity_position_,
                                  left_magnetic_field_, right_magnetic_field_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
Tabulated3DShockTube::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.0)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
Tabulated3DShockTube::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const {
  const auto spatial_velocity = get<hydro::Tags::SpatialVelocity<DataType, 3>>(
      variables(x, tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>>{}));
  return {
      hydro::lorentz_factor(dot_product(spatial_velocity, spatial_velocity))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
Tabulated3DShockTube::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const {
  using density_tag = hydro::Tags::RestMassDensity<DataType>;
  using energy_tag = hydro::Tags::SpecificInternalEnergy<DataType>;
  using pressure_tag = hydro::Tags::Pressure<DataType>;
  const auto data =
      variables(x, tmpl::list<density_tag, energy_tag, pressure_tag>{});
  return hydro::relativistic_specific_enthalpy(
      get<density_tag>(data), get<energy_tag>(data), get<pressure_tag>(data));
}

PUP::able::PUP_ID Tabulated3DShockTube::my_PUP_ID = 0;

bool operator==(const Tabulated3DShockTube& lhs,
                const Tabulated3DShockTube& rhs) {
  return lhs.left_rest_mass_density_ == rhs.left_rest_mass_density_ and
         lhs.right_rest_mass_density_ == rhs.right_rest_mass_density_ and
         lhs.left_temperature_ == rhs.left_temperature_ and
         lhs.right_temperature_ == rhs.right_temperature_ and
         lhs.left_electron_fraction_ == rhs.left_electron_fraction_ and
         lhs.right_electron_fraction_ == rhs.right_electron_fraction_ and
         lhs.left_spatial_velocity_ == rhs.left_spatial_velocity_ and
         lhs.right_spatial_velocity_ == rhs.right_spatial_velocity_ and
         lhs.left_magnetic_field_ == rhs.left_magnetic_field_ and
         lhs.right_magnetic_field_ == rhs.right_magnetic_field_ and
         lhs.discontinuity_position_ == rhs.discontinuity_position_ and
         lhs.ye_perturbation_value_ == rhs.ye_perturbation_value_ and
         lhs.ye_perturbation_center_ == rhs.ye_perturbation_center_ and
         lhs.ye_perturbation_half_width_left_ ==
             rhs.ye_perturbation_half_width_left_ and
         lhs.ye_perturbation_half_width_right_ ==
             rhs.ye_perturbation_half_width_right_ and
         lhs.ye_perturbation_left_temperature_ ==
             rhs.ye_perturbation_left_temperature_ and
         lhs.ye_perturbation_right_temperature_ ==
             rhs.ye_perturbation_right_temperature_ and
         lhs.equation_of_state_ == rhs.equation_of_state_;
}

bool operator!=(const Tabulated3DShockTube& lhs,
                const Tabulated3DShockTube& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                                          \
  template tuples::TaggedTuple < TAG(data) < DTYPE(data) >>                   \
      Tabulated3DShockTube::variables(const tnsr::I<DTYPE(data), 3>& x,       \
                                      tmpl::list < TAG(data) < DTYPE(data) >> \
                                      /*meta*/) const;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::ElectronFraction,
     hydro::Tags::Temperature, hydro::Tags::SpecificInternalEnergy,
     hydro::Tags::Pressure, hydro::Tags::DivergenceCleaningField,
     hydro::Tags::LorentzFactor, hydro::Tags::SpecificEnthalpy))

#define INSTANTIATE_VECTORS(_, data)                      \
  template tuples::TaggedTuple < TAG(data) < DTYPE(data), \
      3 >> Tabulated3DShockTube::variables(               \
               const tnsr::I<DTYPE(data), 3>& x,          \
               tmpl::list < TAG(data) < DTYPE(data), 3 >> \
               /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (double, DataVector),
                        (hydro::Tags::SpatialVelocity,
                         hydro::Tags::MagneticField))

#undef DTYPE
#undef TAG
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_VECTORS

}  // namespace grmhd::AnalyticData
