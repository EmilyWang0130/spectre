// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/RiemannProblem.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/RotatedRiemannProblem.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct RotatedRiemannProblemProxy
    : grmhd::AnalyticData::RotatedRiemannProblem {
  using grmhd::AnalyticData::RotatedRiemannProblem::RotatedRiemannProblem;

  template <typename DataType>
  using hydro_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::SpatialVelocity<DataType, 3>,
                 hydro::Tags::SpecificInternalEnergy<DataType>,
                 hydro::Tags::Pressure<DataType>,
                 hydro::Tags::LorentzFactor<DataType>,
                 hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  using grmhd_variables_tags =
      tmpl::push_back<hydro_variables_tags<DataType>,
                      hydro::Tags::MagneticField<DataType, 3>,
                      hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<hydro_variables_tags<DataType>>
  hydro_variables(const tnsr::I<DataType, 3, Frame::Inertial>& x) const {
    return variables(x, hydro_variables_tags<DataType>{});
  }

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<grmhd_variables_tags<DataType>>
  grmhd_variables(const tnsr::I<DataType, 3, Frame::Inertial>& x) const {
    return variables(x, grmhd_variables_tags<DataType>{});
  }
};

// the large-B_x Dai-Woodward tube in c = 1 units (PLUTO's SodRot), tan(angle) = 1/2
const double angle = std::atan(0.5);
const std::array left_velocity{0.012, 0.0001, 0.005};
const std::array right_velocity{0.0, 0.0, 0.0};
const std::array left_magnetic_field{0.5641895835477563, 0.010155412503859613,
                                     0.005641895835477563};
const std::array right_magnetic_field{0.5641895835477563, 0.011283791670955126,
                                      0.005641895835477563};

void test_create_from_options() {
  register_classes_with_charm<grmhd::AnalyticData::RotatedRiemannProblem>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          grmhd::AnalyticData::RotatedRiemannProblem>(
          "RotatedRiemannProblem:\n"
          "  AdiabaticIndex: 1.6666666666666667\n"
          "  LeftDensity: 1.08\n"
          "  LeftPressure: 9.5e-05\n"
          "  LeftVelocity: [0.012, 0.0001, 0.005]\n"
          "  LeftMagneticField: [0.5641895835477563, 0.010155412503859613, "
          "0.005641895835477563]\n"
          "  RightDensity: 1.0\n"
          "  RightPressure: 1.0e-4\n"
          "  RightVelocity: [0.0, 0.0, 0.0]\n"
          "  RightMagneticField: [0.5641895835477563, 0.011283791670955126, "
          "0.005641895835477563]\n"
          "  Lapse: 1.0\n"
          "  ShiftX: 0.0\n"
          "  InterfaceAngle: 0.4636476090008061\n"
          "  SlabHalfWidth: None\n"
          "  SlabPeriod: None\n")
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& riemann_problem =
      dynamic_cast<const grmhd::AnalyticData::RotatedRiemannProblem&>(
          *deserialized_option_solution);

  CHECK(riemann_problem ==
        grmhd::AnalyticData::RotatedRiemannProblem(
            1.6666666666666667, 1.08, 1.0, 9.5e-05, 1.0e-4, left_velocity,
            right_velocity, left_magnetic_field, right_magnetic_field, 1.0,
            0.0, 0.4636476090008061));
}

void test_move() {
  grmhd::AnalyticData::RotatedRiemannProblem riemann_problem(
      2.0, 1.0, 0.125, 1.0, 0.1, left_velocity, right_velocity,
      left_magnetic_field, right_magnetic_field, 2.0, 0.4, angle);
  grmhd::AnalyticData::RotatedRiemannProblem riemann_problem_copy(
      2.0, 1.0, 0.125, 1.0, 0.1, left_velocity, right_velocity,
      left_magnetic_field, right_magnetic_field, 2.0, 0.4, angle);
  test_move_semantics(std::move(riemann_problem),
                      riemann_problem_copy);  //  NOLINT
}

void test_serialize() {
  grmhd::AnalyticData::RotatedRiemannProblem riemann_problem(
      2.0, 1.0, 0.125, 1.0, 0.1, left_velocity, right_velocity,
      left_magnetic_field, right_magnetic_field, 2.0, 0.4, angle);
  test_serialization(riemann_problem);
}

template <typename DataType>
void test_variables(const DataType& used_for_size) {
  const double adiabatic_index = 2.0;
  const double left_density = 1.0;
  const double right_density = 0.125;
  const double left_pressure = 1.0;
  const double right_pressure = 0.1;
  const double lapse = 2.0;
  const double shift = 0.4;

  const auto member_variables = std::make_tuple(
      adiabatic_index, left_density, right_density, left_pressure,
      right_pressure, left_velocity, right_velocity, left_magnetic_field,
      right_magnetic_field, lapse, shift, angle);  // the slab options stay
                                                   // out of the pypp tuple
                                                   // (tested in test_slab)

  RotatedRiemannProblemProxy riemann_problem(
      adiabatic_index, left_density, right_density, left_pressure,
      right_pressure, left_velocity, right_velocity, left_magnetic_field,
      right_magnetic_field, lapse, shift, angle);

  pypp::check_with_random_values<1>(
      &RotatedRiemannProblemProxy::hydro_variables<DataType>, riemann_problem,
      "RotatedRiemannProblem",
      {"rest_mass_density", "spatial_velocity", "specific_internal_energy",
       "compute_pressure", "lorentz_factor", "specific_enthalpy"},
      {{{-1.0, 1.0}}}, member_variables, used_for_size);

  pypp::check_with_random_values<1>(
      &RotatedRiemannProblemProxy::grmhd_variables<DataType>, riemann_problem,
      "RotatedRiemannProblem",
      {"rest_mass_density", "spatial_velocity", "specific_internal_energy",
       "compute_pressure", "lorentz_factor", "specific_enthalpy",
       "magnetic_field", "divergence_cleaning_field"},
      {{{-1.0, 1.0}}}, member_variables, used_for_size);
}

// InterfaceAngle = 0 is RiemannProblem; at the SodRot angle the interface
// passes between (0.1, -0.3) [left, s < 0] and (0.3, -0.1) [right, s > 0]
void test_against_riemann_problem() {
  const grmhd::AnalyticData::RiemannProblem unrotated(
      2.0, 1.0, 0.125, 1.0, 0.1, left_velocity, right_velocity,
      left_magnetic_field, right_magnetic_field, 1.0, 0.0);
  const grmhd::AnalyticData::RotatedRiemannProblem zero_angle(
      2.0, 1.0, 0.125, 1.0, 0.1, left_velocity, right_velocity,
      left_magnetic_field, right_magnetic_field, 1.0, 0.0, 0.0);
  const grmhd::AnalyticData::RotatedRiemannProblem rotated(
      2.0, 1.0, 0.125, 1.0, 0.1, left_velocity, right_velocity,
      left_magnetic_field, right_magnetic_field, 1.0, 0.0, angle);
  using tags = tmpl::list<hydro::Tags::RestMassDensity<double>,
                          hydro::Tags::SpatialVelocity<double, 3>,
                          hydro::Tags::MagneticField<double, 3>>;
  for (const auto& point :
       {std::array{-0.3, 0.2, 0.0}, std::array{0.3, -0.2, 0.0},
        std::array{0.0, 0.4, 0.0}, std::array{0.1, -0.3, 0.0},
        std::array{0.3, -0.1, 0.0}}) {
    const tnsr::I<double, 3> x{{{point[0], point[1], point[2]}}};
    const auto a = unrotated.variables(x, tags{});
    const auto b = zero_angle.variables(x, tags{});
    CHECK(get<hydro::Tags::RestMassDensity<double>>(a) ==
          get<hydro::Tags::RestMassDensity<double>>(b));
    CHECK(get<hydro::Tags::SpatialVelocity<double, 3>>(a) ==
          get<hydro::Tags::SpatialVelocity<double, 3>>(b));
    CHECK(get<hydro::Tags::MagneticField<double, 3>>(a) ==
          get<hydro::Tags::MagneticField<double, 3>>(b));
  }
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  {
    const tnsr::I<double, 3> x{{{0.1, -0.3, 0.0}}};  // s = -0.045: left
    const auto v = rotated.variables(x, tags{});
    CHECK(get(get<hydro::Tags::RestMassDensity<double>>(v)) == 1.0);
    const auto& b = get<hydro::Tags::MagneticField<double, 3>>(v);
    CHECK(get<0>(b) ==
          approx(left_magnetic_field[0] * c - left_magnetic_field[1] * s));
    CHECK(get<1>(b) ==
          approx(left_magnetic_field[0] * s + left_magnetic_field[1] * c));
    CHECK(get<2>(b) == left_magnetic_field[2]);
    // the normal component is B_n on both sides
    CHECK(get<0>(b) * c + get<1>(b) * s == approx(left_magnetic_field[0]));
  }
  {
    const tnsr::I<double, 3> x{{{0.3, -0.1, 0.0}}};  // s = +0.224: right
    const auto v = rotated.variables(x, tags{});
    CHECK(get(get<hydro::Tags::RestMassDensity<double>>(v)) == 0.125);
    const auto& b = get<hydro::Tags::MagneticField<double, 3>>(v);
    CHECK(get<0>(b) * c + get<1>(b) * s == approx(right_magnetic_field[0]));
    CHECK(-get<0>(b) * s + get<1>(b) * c == approx(right_magnetic_field[1]));
  }
}

// the periodic slab: on a 2 x 2 box at 45 degrees with period sqrt(2) and half-width sqrt(2)/4,
// the bands alternate consistently: (0.1, 0.1) is left, (0.6, 0.6) right, (1.0, 1.0) [s = 1.414,
// wrapped to 0] left again, and (-0.9, -0.9) [s = -1.273, wrapped to +0.141] left
void test_slab() {
  const double angle = 0.25 * M_PI;
  const double period = 2.0 / std::sqrt(2.0);
  const grmhd::AnalyticData::RotatedRiemannProblem slab(
      2.0, 1.0, 0.125, 1.0, 0.1, left_velocity, right_velocity,
      left_magnetic_field, right_magnetic_field, 1.0, 0.0, angle,
      0.25 * period, period);
  using tags = tmpl::list<hydro::Tags::RestMassDensity<double>>;
  for (const auto& [point, expected] :
       std::vector<std::pair<std::array<double, 2>, double>>{
           {{0.1, 0.1}, 1.0}, {{0.6, 0.6}, 0.125}, {{1.0, 1.0}, 1.0},
           {{-0.9, -0.9}, 1.0}, {{0.3, -0.3}, 1.0}, {{0.9, 0.0}, 0.125}}) {
    const tnsr::I<double, 3> x{{{point[0], point[1], 0.0}}};
    CHECK(get(get<hydro::Tags::RestMassDensity<double>>(
              slab.variables(x, tags{}))) == expected);
  }
  const auto deserialized = serialize_and_deserialize(slab);
  CHECK(deserialized == slab);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.GrMhd.RotatedRiemannProblem",
    "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/GrMhd"};

  test_create_from_options();
  test_serialize();
  test_move();
  test_against_riemann_problem();

  test_variables(std::numeric_limits<double>::signaling_NaN());
  test_variables(DataVector(5));
  test_slab();
}
