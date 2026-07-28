// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>

#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/BeckwithStoneKhInstability.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
struct BeckwithStoneKhInstabilityProxy
    : public ::grmhd::AnalyticData::BeckwithStoneKhInstability {
  using grmhd::AnalyticData::BeckwithStoneKhInstability::
      BeckwithStoneKhInstability;

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
  const double adiabatic_index = 1.43;
  const double shear_velocity = 0.3;
  const double strip_half_width = 0.5;
  const double transition_thickness = 0.1;
  const double upper_density = 2.1;
  const double lower_density = 1.0;
  const double upper_ye = 0.4;
  const double lower_ye = 0.2;
  const double pressure = 1.1;
  const double perturbation_amplitude = 0.1;
  const double perturbation_width = 0.1;
  const std::array<double, 3> magnetic_field{{1.0e-3, 0.0, 0.0}};
  const auto members = std::make_tuple(
      adiabatic_index, shear_velocity, strip_half_width, transition_thickness,
      upper_density, lower_density, upper_ye, lower_ye, pressure,
      perturbation_amplitude, perturbation_width, magnetic_field);

  register_classes_with_charm<
      grmhd::AnalyticData::BeckwithStoneKhInstability>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          grmhd::AnalyticData::BeckwithStoneKhInstability>(
          "BeckwithStoneKhInstability:\n"
          "  AdiabaticIndex: 1.43\n"
          "  ShearVelocity: 0.3\n"
          "  StripHalfWidth: 0.5\n"
          "  TransitionThickness: 0.1\n"
          "  UpperDensity: 2.1\n"
          "  LowerDensity: 1.0\n"
          "  UpperYe: 0.4\n"
          "  LowerYe: 0.2\n"
          "  Pressure: 1.1\n"
          "  PerturbAmplitude: 0.1\n"
          "  PerturbWidth: 0.1\n"
          "  MagneticField: [1.0e-3, 0.0, 0.0]\n"
          "  RandomVzAmplitude: 0.0\n"
          "  RandomSeed: 0\n")
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& kh_instability =
      dynamic_cast<const grmhd::AnalyticData::BeckwithStoneKhInstability&>(
          *deserialized_option_solution);

  CHECK(kh_instability == grmhd::AnalyticData::BeckwithStoneKhInstability(
                              adiabatic_index, shear_velocity, strip_half_width,
                              transition_thickness, upper_density,
                              lower_density, upper_ye, lower_ye, pressure,
                              perturbation_amplitude, perturbation_width,
                              magnetic_field));

  BeckwithStoneKhInstabilityProxy kh_inst_to_move(
      adiabatic_index, shear_velocity, strip_half_width, transition_thickness,
      upper_density, lower_density, upper_ye, lower_ye, pressure,
      perturbation_amplitude, perturbation_width, magnetic_field);
  BeckwithStoneKhInstabilityProxy kh_inst(
      adiabatic_index, shear_velocity, strip_half_width, transition_thickness,
      upper_density, lower_density, upper_ye, lower_ye, pressure,
      perturbation_amplitude, perturbation_width, magnetic_field);
  test_move_semantics(std::move(kh_inst_to_move), kh_inst);  //  NOLINT

  // run post-serialized state through checks with random numbers
  pypp::check_with_random_values<1>(
      &BeckwithStoneKhInstabilityProxy::template primitive_variables<DataType>,
      serialize_and_deserialize(kh_inst), "BeckwithStoneKhInstability",
      {"rest_mass_density", "electron_fraction", "velocity",
       "specific_internal_energy", "pressure", "lorentz_factor",
       "specific_enthalpy", "magnetic_field", "divergence_cleaning_field"},
      {{{0.0, 1.0}}}, members, used_for_size);
}
void test_random_vz() {
  // With RandomVzAmplitude > 0, v_z should be a deterministic function of
  // spatial coordinates, uniformly distributed in [0, amplitude).
  const double amplitude = 0.01;
  const std::uint64_t seed = 42;
  const grmhd::AnalyticData::BeckwithStoneKhInstability kh(
      /*adiabatic_index=*/4.0 / 3.0, /*shear_velocity=*/0.5,
      /*strip_half_width=*/0.5, /*transition_thickness=*/0.01,
      /*upper_density=*/1.0, /*lower_density=*/0.01,
      /*upper_ye=*/0.1, /*lower_ye=*/0.1, /*pressure=*/1.0,
      /*perturbation_amplitude=*/0.1, /*perturbation_width=*/0.316227766,
      /*magnetic_field=*/{{0.0, 0.0, 0.0}},
      /*random_vz_amplitude=*/amplitude, /*random_seed=*/seed);

  // Two evaluations at the same coordinates produce the same v_z.
  tnsr::I<DataVector, 3, Frame::Inertial> point_a(1_st);
  get<0>(point_a) = DataVector{0.123};
  get<1>(point_a) = DataVector{-0.456};
  get<2>(point_a) = DataVector{0.789};
  const auto v_a1 =
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(kh.variables(
          point_a, tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3>>{}));
  const auto v_a2 =
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(kh.variables(
          point_a, tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3>>{}));
  CHECK(get<2>(v_a1)[0] == get<2>(v_a2)[0]);
  CHECK(get<2>(v_a1)[0] >= 0.0);
  CHECK(get<2>(v_a1)[0] < amplitude);

  // Different coordinates produce (almost certainly) different values, and
  // the values are all in [0, amplitude).
  const size_t n_samples = 1024;
  tnsr::I<DataVector, 3, Frame::Inertial> samples(n_samples);
  double sum = 0.0;
  double vmax = -1.0;
  double vmin = 1.0;
  for (size_t i = 0; i < n_samples; ++i) {
    get<0>(samples)[i] = -0.5 + static_cast<double>(i) / n_samples;
    get<1>(samples)[i] = -1.0 + 2.0 * static_cast<double>(i) / n_samples;
    get<2>(samples)[i] = -0.5 + static_cast<double>(i) / n_samples;
  }
  const auto vs = get<hydro::Tags::SpatialVelocity<DataVector, 3>>(kh.variables(
      samples, tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3>>{}));
  for (size_t i = 0; i < n_samples; ++i) {
    const double v = get<2>(vs)[i];
    CHECK(v >= 0.0);
    CHECK(v < amplitude);
    sum += v;
    vmax = std::max(vmax, v);
    vmin = std::min(vmin, v);
  }
  // Mean of Uniform[0, amplitude] is amplitude/2. Allow 10% tolerance on 1k
  // samples (Central Limit ~ std / sqrt(N) ≈ 0.29·amplitude/sqrt(1024) ~
  // 0.9% of amplitude, so 10% is very safe).
  CHECK(sum / n_samples == approx(0.5 * amplitude).epsilon(0.1));
  CHECK(vmax > 0.9 * amplitude);
  CHECK(vmin < 0.1 * amplitude);

  // Different seeds produce different fields.
  const grmhd::AnalyticData::BeckwithStoneKhInstability kh_other_seed(
      4.0 / 3.0, 0.5, 0.5, 0.01, 1.0, 0.01, 0.1, 0.1, 1.0, 0.1, 0.316227766,
      {{0.0, 0.0, 0.0}}, amplitude, /*random_seed=*/7);
  const auto v_other =
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(kh_other_seed.variables(
          point_a, tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3>>{}));
  CHECK(get<2>(v_a1)[0] != get<2>(v_other)[0]);

  // Amplitude = 0 disables the perturbation (v_z stays 0).
  const grmhd::AnalyticData::BeckwithStoneKhInstability kh_no_vz(
      4.0 / 3.0, 0.5, 0.5, 0.01, 1.0, 0.01, 0.1, 0.1, 1.0, 0.1, 0.316227766,
      {{0.0, 0.0, 0.0}});  // default random amplitude = 0
  const auto v_zero =
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(kh_no_vz.variables(
          point_a, tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3>>{}));
  CHECK(get<2>(v_zero)[0] == 0.0);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.GrMhd.BeckwithStoneKhInstability",
    "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/GrMhd"};

  test(std::numeric_limits<double>::signaling_NaN());
  test(DataVector(5));
  test_random_vz();
}
