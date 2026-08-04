// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/EosTable.hpp"
#include "IO/H5/File.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated1D.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {

struct SliceSpec {
  double nb_lo_fm3;
  double nb_hi_fm3;
  size_t num_points;
  double pressure_amplitude;
  double eps_amplitude;
  double chi_slope_value;
  double gamma;
};

void write_slice_subfile(h5::H5File<h5::AccessType::ReadWrite>& eos_file,
                         const std::string& subfilename,
                         const SliceSpec& spec) {
  const std::vector<std::string> ivar_names{"number density"};
  const std::vector<std::array<double, 2>> ivar_bounds{
      {spec.nb_lo_fm3, spec.nb_hi_fm3}};
  const std::vector<size_t> ivar_npts{spec.num_points};
  const std::vector<bool> ivar_log_spacing{true};

  DataVector pressure(spec.num_points);
  DataVector eps(spec.num_points);
  DataVector chi_slope(spec.num_points);
  const double log_lo = std::log(spec.nb_lo_fm3);
  const double log_hi = std::log(spec.nb_hi_fm3);
  const double dlog =
      (log_hi - log_lo) / static_cast<double>(spec.num_points - 1);
  for (size_t i = 0; i < spec.num_points; ++i) {
    const double nb = std::exp(log_lo + static_cast<double>(i) * dlog);
    pressure[i] = spec.pressure_amplitude * std::pow(nb, spec.gamma);
    eps[i] = spec.eps_amplitude * std::pow(nb, spec.gamma - 1.0);
    chi_slope[i] = spec.chi_slope_value;
  }

  auto& eos_table = eos_file.insert<h5::EosTable>(
      "/" + subfilename, ivar_names, ivar_bounds, ivar_npts, ivar_log_spacing,
      /*beta_equilibrium=*/true, /*version=*/uint32_t{1});
  eos_table.write_quantity("pressure", pressure);
  eos_table.write_quantity("specific internal energy", eps);
  eos_table.write_quantity("chi slope", chi_slope);
  eos_file.close_current_object();
}

SliceSpec default_spec() {
  return SliceSpec{/*nb_lo_fm3=*/1.0e-5,
                   /*nb_hi_fm3=*/1.0e-2,
                   /*num_points=*/64,
                   /*pressure_amplitude=*/1.0e5,
                   /*eps_amplitude=*/1.0e3,
                   /*chi_slope_value=*/2.0,
                   /*gamma=*/2.0};
}

template <bool IsRelativistic>
void check_bounds(const EquationsOfState::Tabulated1D<IsRelativistic>& eos,
                  const SliceSpec& spec) {
  constexpr double nb_fm3_to_geom = hydro::units::nuclear::neutron_mass /
                                    hydro::units::nuclear::pressure_unit;
  const double expected_rho_lo = nb_fm3_to_geom * spec.nb_lo_fm3;
  const double expected_rho_hi = nb_fm3_to_geom * spec.nb_hi_fm3;
  CHECK(eos.rest_mass_density_lower_bound() == approx(expected_rho_lo));
  CHECK(eos.rest_mass_density_upper_bound() == approx(expected_rho_hi));
  if constexpr (IsRelativistic) {
    CHECK(eos.specific_enthalpy_lower_bound() > 1.0);
  } else {
    CHECK(eos.specific_enthalpy_lower_bound() > 0.0);
  }
  CHECK(eos.baryon_mass() ==
        approx(hydro::units::geometric::default_baryon_mass));
  CHECK(eos.is_barotropic());
}

template <bool IsRelativistic>
void check_serialization_and_clone(
    const EquationsOfState::Tabulated1D<IsRelativistic>& eos) {
  const auto deserialized = serialize_and_deserialize(eos);
  CHECK(eos == deserialized);
  CHECK(eos.is_equal(deserialized));
  const auto clone = eos.get_clone();
  CHECK(clone->is_equal(eos));
}

void check_factory_creation(
    const std::string& filename, const std::string& subfilename,
    const EquationsOfState::Tabulated1D<true>& expected) {
  namespace EoS = EquationsOfState;
  const auto eos_from_yaml = TestHelpers::test_creation<
      std::unique_ptr<EoS::EquationOfState<true, 1>>>(
      "Tabulated1D:\n"
      "  TableFilename: " +
      filename +
      "\n"
      "  TableSubFilename: " +
      subfilename + "\n");
  CHECK(eos_from_yaml->is_equal(expected));
}

}  // namespace

// The h5 fixture write embeds the SpECTRE source tarball (~4 s cost),
// so bump the ctest timeout above the default 2 s.
// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.Tabulated1D",
                  "[Unit][EquationsOfState]") {
  namespace EoS = EquationsOfState;
  register_derived_classes_with_charm<EoS::EquationOfState<true, 1>>();
  register_derived_classes_with_charm<EoS::EquationOfState<false, 1>>();
  register_derived_classes_with_charm<EoS::EquationOfState<true, 3>>();
  register_derived_classes_with_charm<EoS::EquationOfState<false, 3>>();

  // Use a single h5 file with two EosTable subfiles. Creating an h5 file
  // embeds the entire SpECTRE source tarball as "/src" (~4 s cost);
  // making two files would double that with no benefit here.
  const std::string filename{"Unit.EquationsOfState.Tabulated1D.h5"};
  const std::string subfile_a{"tab1d_a.eos_beta"};
  const std::string subfile_b{"tab1d_b.eos_beta"};

  const auto spec_a = default_spec();
  SliceSpec spec_b = spec_a;
  spec_b.num_points = 32;  // different length -> distinct table

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  {
    h5::H5File<h5::AccessType::ReadWrite> eos_file{filename};
    write_slice_subfile(eos_file, subfile_a, spec_a);
    write_slice_subfile(eos_file, subfile_b, spec_b);
  }

  const EoS::Tabulated1D<true> eos_a_rel{filename, subfile_a};
  const EoS::Tabulated1D<false> eos_a_nonrel{filename, subfile_a};
  const EoS::Tabulated1D<true> eos_b_rel{filename, subfile_b};

  check_bounds(eos_a_rel, spec_a);
  check_bounds(eos_a_nonrel, spec_a);
  check_serialization_and_clone(eos_a_rel);
  check_serialization_and_clone(eos_a_nonrel);
  CHECK(eos_a_rel != eos_b_rel);
  check_factory_creation(filename, subfile_a, eos_a_rel);

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
