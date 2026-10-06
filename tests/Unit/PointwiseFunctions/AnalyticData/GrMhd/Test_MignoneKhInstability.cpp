// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <memory>

#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MignoneKhInstability.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
struct MignoneKhInstabilityProxy
    : public ::grmhd::AnalyticData::MignoneKhInstability {
  using grmhd::AnalyticData::MignoneKhInstability::
      MignoneKhInstability;

  template <typename DataType>
  using variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::ElectronFraction<DataType>,
                 hydro::Tags::SpatialVelocity<DataType, 3>,
                 hydro::Tags::SpecificInternalEnergy<DataType>,
                 hydro::Tags::Pressure<DataType>,
                 hydro::Tags::LorentzFactor<DataType>,
                 hydro::Tags::SpecificEnthalpy<DataType>,
                 hydro::Tags::MagneticField<DataType, 3>,
                 hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags<DataType>>
  primitive_variables(const tnsr::I<DataType, 3, Frame::Inertial>& x) const {
    return this->variables(x, variables_tags<DataType>{});
  }
};

template <typename DataType>
void test(const DataType& used_for_size) {
  const double adiabatic_index = 4.0 / 3.0;
  const double density = 1.0;
  const double pressure = 20.0;
  const double electron_fraction = 0.1;
  const double shear_velocity = 0.25;
  const double shear_thickness = 0.01;
  const double perturbation_amplitude = 0.0025;
  const double perturbation_wavelength = 1.0;
  const double perturbation_width = 0.1;
  const std::array<double, 3> magnetic_field{{0.6324555320336759, 0.0,
                                              6.324555320336759}};
  const auto members = std::make_tuple(
      adiabatic_index, density, pressure, electron_fraction, shear_velocity,
      shear_thickness, perturbation_amplitude, perturbation_wavelength,
      perturbation_width, magnetic_field);

  register_classes_with_charm<grmhd::AnalyticData::MignoneKhInstability>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          grmhd::AnalyticData::MignoneKhInstability>(
          "MignoneKhInstability:\n"
          "  AdiabaticIndex: 1.3333333333333333\n"
          "  Density: 1.0\n"
          "  Pressure: 20.0\n"
          "  ElectronFraction: 0.1\n"
          "  ShearVelocity: 0.25\n"
          "  ShearThickness: 0.01\n"
          "  PerturbAmplitude: 0.0025\n"
          "  PerturbWavelength: 1.0\n"
          "  PerturbWidth: 0.1\n"
          "  MagneticField: [0.6324555320336759, 0.0, 6.324555320336759]\n")
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& kh_instability =
      dynamic_cast<const grmhd::AnalyticData::MignoneKhInstability&>(
          *deserialized_option_solution);

  CHECK(kh_instability == grmhd::AnalyticData::MignoneKhInstability(
                              adiabatic_index, density, pressure,
                              electron_fraction, shear_velocity,
                              shear_thickness, perturbation_amplitude,
                              perturbation_wavelength, perturbation_width,
                              magnetic_field));

  MignoneKhInstabilityProxy kh_inst_to_move(
      adiabatic_index, density, pressure, electron_fraction, shear_velocity,
      shear_thickness, perturbation_amplitude, perturbation_wavelength,
      perturbation_width, magnetic_field);
  MignoneKhInstabilityProxy kh_inst(
      adiabatic_index, density, pressure, electron_fraction, shear_velocity,
      shear_thickness, perturbation_amplitude, perturbation_wavelength,
      perturbation_width, magnetic_field);
  test_move_semantics(std::move(kh_inst_to_move), kh_inst);  //  NOLINT

  // run post-serialized state through checks with random numbers
  pypp::check_with_random_values<1>(
      &MignoneKhInstabilityProxy::template primitive_variables<DataType>,
      serialize_and_deserialize(kh_inst), "MignoneKhInstability",
      {"rest_mass_density", "electron_fraction", "velocity",
       "specific_internal_energy", "pressure", "lorentz_factor",
       "specific_enthalpy", "magnetic_field", "divergence_cleaning_field"},
      {{{-1.0, 1.0}}}, members, used_for_size);

  // PLUTO's profile at a few points: the shear layer and the perturbation.
  const grmhd::AnalyticData::MignoneKhInstability kh(
      adiabatic_index, density, pressure, electron_fraction, shear_velocity,
      shear_thickness, perturbation_amplitude, perturbation_wavelength,
      perturbation_width, magnetic_field);
  tnsr::I<DataVector, 3, Frame::Inertial> points(3_st);
  get<0>(points) = DataVector{0.25, 0.75, 0.5};
  get<1>(points) = DataVector{0.0, 0.0, 1.0};
  get<2>(points) = DataVector{0.0, 0.0, 0.0};
  const auto v = get<hydro::Tags::SpatialVelocity<DataVector, 3>>(kh.variables(
      points, tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3>>{}));
  CHECK(get<0>(v)[0] == approx(0.0));
  CHECK(get<1>(v)[0] == approx(perturbation_amplitude));
  CHECK(get<1>(v)[1] == approx(-perturbation_amplitude));
  CHECK(get<0>(v)[2] == approx(-shear_velocity * std::tanh(100.0)));
  CHECK(get<1>(v)[2] == approx(0.0));
  CHECK(get<2>(v)[2] == 0.0);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.GrMhd.MignoneKhInstability",
    "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/GrMhd"};

  test(std::numeric_limits<double>::signaling_NaN());
  test(DataVector(5));
}
