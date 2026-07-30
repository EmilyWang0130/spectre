// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated1D.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {

template <bool IsRelativistic>
void check_bounds_and_flags() {
  const auto eos = EquationsOfState::Tabulated1D<IsRelativistic>{128, true};
  CHECK(0.0 == eos.rest_mass_density_lower_bound());
  CHECK(std::numeric_limits<double>::max() ==
        eos.rest_mass_density_upper_bound());
  CHECK(0.0 == eos.specific_internal_energy_lower_bound());
  CHECK(std::numeric_limits<double>::max() ==
        eos.specific_internal_energy_upper_bound());
  if constexpr (IsRelativistic) {
    CHECK(1.0 == eos.specific_enthalpy_lower_bound());
  } else {
    CHECK(0.0 == eos.specific_enthalpy_lower_bound());
  }
  CHECK(eos.baryon_mass() ==
        approx(hydro::units::geometric::default_baryon_mass));
  CHECK(eos.is_barotropic());
}

template <bool IsRelativistic>
void check_equality_and_clone() {
  const auto eos_a = EquationsOfState::Tabulated1D<IsRelativistic>{128, true};
  const auto eos_b = EquationsOfState::Tabulated1D<IsRelativistic>{128, true};
  const auto eos_c = EquationsOfState::Tabulated1D<IsRelativistic>{256, true};
  const auto eos_d = EquationsOfState::Tabulated1D<IsRelativistic>{128, false};
  CHECK(eos_a == eos_b);
  CHECK(eos_a != eos_c);
  CHECK(eos_a != eos_d);
  const auto clone = eos_a.get_clone();
  CHECK(clone->is_equal(eos_a));
}

template <bool IsRelativistic>
void check_serialization() {
  const auto eos = EquationsOfState::Tabulated1D<IsRelativistic>{64, false};
  const auto deserialized = serialize_and_deserialize(eos);
  CHECK(eos == deserialized);
}

void check_factory_creation() {
  namespace EoS = EquationsOfState;
  const auto eos_from_yaml = TestHelpers::test_creation<
      std::unique_ptr<EoS::EquationOfState<true, 1>>>(
      "Tabulated1D:\n"
      "  NumberOfGridPoints: 128\n"
      "  LogSpacing: true\n");
  const auto expected = EoS::Tabulated1D<true>{128, true};
  CHECK(eos_from_yaml->is_equal(expected));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.Tabulated1D",
                  "[Unit][EquationsOfState]") {
  namespace EoS = EquationsOfState;
  register_derived_classes_with_charm<EoS::EquationOfState<true, 1>>();
  register_derived_classes_with_charm<EoS::EquationOfState<false, 1>>();

  check_bounds_and_flags<true>();
  check_bounds_and_flags<false>();
  check_equality_and_clone<true>();
  check_equality_and_clone<false>();
  check_serialization<true>();
  check_serialization<false>();
  check_factory_creation();
}
