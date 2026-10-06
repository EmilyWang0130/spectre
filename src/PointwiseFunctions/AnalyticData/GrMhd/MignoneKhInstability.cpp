// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/MignoneKhInstability.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace grmhd::AnalyticData {

MignoneKhInstability::MignoneKhInstability(
    const double adiabatic_index, const double density, const double pressure,
    const double electron_fraction, const double shear_velocity,
    const double shear_thickness, const double perturbation_amplitude,
    const double perturbation_wavelength, const double perturbation_width,
    const std::array<double, 3>& magnetic_field)
    : adiabatic_index_(adiabatic_index),
      density_(density),
      pressure_(pressure),
      electron_fraction_(electron_fraction),
      shear_velocity_(shear_velocity),
      shear_thickness_(shear_thickness),
      perturbation_amplitude_(perturbation_amplitude),
      perturbation_wavelength_(perturbation_wavelength),
      perturbation_width_(perturbation_width),
      magnetic_field_(magnetic_field),
      equation_of_state_(adiabatic_index) {
  ASSERT(density_ > 0.0, "The density must be positive. The value given was "
                             << density_ << ".");
  ASSERT(pressure_ > 0.0, "The pressure must be positive. The value given was "
                              << pressure_ << ".");
  ASSERT(electron_fraction_ >= 0.0 and electron_fraction_ <= 1.0,
         "The electron fraction must be in [0, 1]. The value given was "
             << electron_fraction_ << ".");
  ASSERT(shear_thickness_ > 0.0,
         "The shear thickness must be positive. The value given was "
             << shear_thickness_ << ".");
  ASSERT(perturbation_wavelength_ > 0.0,
         "The perturbation wavelength must be positive. The value given was "
             << perturbation_wavelength_ << ".");
  ASSERT(perturbation_width_ > 0.0,
         "The perturbation width must be positive. The value given was "
             << perturbation_width_ << ".");
}

std::unique_ptr<evolution::initial_data::InitialData>
MignoneKhInstability::get_clone() const {
  return std::make_unique<MignoneKhInstability>(*this);
}

MignoneKhInstability::MignoneKhInstability(CkMigrateMessage* msg)
    : InitialData(msg) {}

void MignoneKhInstability::pup(PUP::er& p) {
  InitialData::pup(p);
  p | adiabatic_index_;
  p | density_;
  p | pressure_;
  p | electron_fraction_;
  p | shear_velocity_;
  p | shear_thickness_;
  p | perturbation_amplitude_;
  p | perturbation_wavelength_;
  p | perturbation_width_;
  p | magnetic_field_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
MignoneKhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const {
  auto result = make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.0);
  const size_t n_pts = get_size(get<0>(x));
  for (size_t s = 0; s < n_pts; ++s) {
    const double xx = get_element(get<0>(x), s);
    const double y = get_element(get<1>(x), s);
    // v_x = -v_0 tanh(y / a)
    get_element(get<0>(result), s) =
        -shear_velocity_ * tanh(y / shear_thickness_);
    // v_y = A sin(2 pi x / lambda) exp(-y^2 / beta^2)
    get_element(get<1>(result), s) =
        perturbation_amplitude_ *
        sin(2.0 * M_PI * xx / perturbation_wavelength_) *
        exp(-square(y / perturbation_width_));
  }
  return {std::move(result)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
MignoneKhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const {
  return make_with_value<Scalar<DataType>>(x, density_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>
MignoneKhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/) const {
  return make_with_value<Scalar<DataType>>(x, electron_fraction_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
MignoneKhInstability::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const {
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::Pressure<DataType>>(
          variables(x, tmpl::list<hydro::Tags::Pressure<DataType>>{})));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>
MignoneKhInstability::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const {
  return make_with_value<Scalar<DataType>>(x, pressure_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
MignoneKhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const {
  auto mag_field = make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(
      get<0>(x), magnetic_field_[0]);
  get<1>(mag_field) = magnetic_field_[1];
  get<2>(mag_field) = magnetic_field_[2];
  return mag_field;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
MignoneKhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return make_with_value<Scalar<DataType>>(x, 0.0);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
MignoneKhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const {
  using velocity_tag = hydro::Tags::SpatialVelocity<DataType, 3>;
  const auto velocity =
      get<velocity_tag>(variables(x, tmpl::list<velocity_tag>{}));
  return {hydro::lorentz_factor(dot_product(velocity, velocity))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
MignoneKhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const {
  return hydro::relativistic_specific_enthalpy(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::SpecificInternalEnergy<DataType>>(variables(
          x, tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})),
      get<hydro::Tags::Pressure<DataType>>(
          variables(x, tmpl::list<hydro::Tags::Pressure<DataType>>{})));
}

PUP::able::PUP_ID MignoneKhInstability::my_PUP_ID = 0;

bool operator==(const MignoneKhInstability& lhs,
                const MignoneKhInstability& rhs) {
  // No comparison for equation_of_state_. Comparing adiabatic_index_ should
  // suffice.
  return lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.density_ == rhs.density_ and lhs.pressure_ == rhs.pressure_ and
         lhs.electron_fraction_ == rhs.electron_fraction_ and
         lhs.shear_velocity_ == rhs.shear_velocity_ and
         lhs.shear_thickness_ == rhs.shear_thickness_ and
         lhs.perturbation_amplitude_ == rhs.perturbation_amplitude_ and
         lhs.perturbation_wavelength_ == rhs.perturbation_wavelength_ and
         lhs.perturbation_width_ == rhs.perturbation_width_ and
         lhs.magnetic_field_ == rhs.magnetic_field_;
}

bool operator!=(const MignoneKhInstability& lhs,
                const MignoneKhInstability& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                         \
  template tuples::TaggedTuple < TAG(data) < DTYPE(data) >>  \
      MignoneKhInstability::variables(                       \
          const tnsr::I<DTYPE(data), 3, Frame::Inertial>& x, \
          tmpl::list < TAG(data) < DTYPE(data) >>) const;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::ElectronFraction,
     hydro::Tags::SpecificInternalEnergy, hydro::Tags::Pressure,
     hydro::Tags::DivergenceCleaningField, hydro::Tags::SpecificEnthalpy,
     hydro::Tags::LorentzFactor))

#define INSTANTIATE_VECTORS(_, data)                                       \
  template tuples::TaggedTuple < TAG(data) < DTYPE(data), 3,               \
      Frame::Inertial >>                                                   \
          MignoneKhInstability::variables(                                 \
              const tnsr::I<DTYPE(data), 3, Frame::Inertial>& x,           \
              tmpl::list < TAG(data) < DTYPE(data), 3, Frame::Inertial >>) \
              const;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (double, DataVector),
                        (hydro::Tags::SpatialVelocity,
                         hydro::Tags::MagneticField))

#undef INSTANTIATE_VECTORS
#undef INSTANTIATE_SCALARS
#undef TAG
#undef DTYPE
}  // namespace grmhd::AnalyticData
