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
  double adiabatic_index_value;
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
  DataVector adiabatic_index(spec.num_points);
  const double log_lo = std::log(spec.nb_lo_fm3);
  const double log_hi = std::log(spec.nb_hi_fm3);
  const double dlog =
      (log_hi - log_lo) / static_cast<double>(spec.num_points - 1);
  for (size_t i = 0; i < spec.num_points; ++i) {
    const double nb = std::exp(log_lo + static_cast<double>(i) * dlog);
    pressure[i] = spec.pressure_amplitude * std::pow(nb, spec.gamma);
    eps[i] = spec.eps_amplitude * std::pow(nb, spec.gamma - 1.0);
    adiabatic_index[i] = spec.adiabatic_index_value;
  }

  auto& eos_table = eos_file.insert<h5::EosTable>(
      "/" + subfilename, ivar_names, ivar_bounds, ivar_npts, ivar_log_spacing,
      /*beta_equilibrium=*/true, /*version=*/uint32_t{1});
  eos_table.write_quantity("pressure", pressure);
  eos_table.write_quantity("specific internal energy", eps);
  eos_table.write_quantity("adiabatic index", adiabatic_index);
  eos_file.close_current_object();
}

SliceSpec default_spec() {
  return SliceSpec{/*nb_lo_fm3=*/1.0e-5,
                   /*nb_hi_fm3=*/1.0e-2,
                   /*num_points=*/64,
                   /*pressure_amplitude=*/1.0e5,
                   /*eps_amplitude=*/1.0e3,
                   /*adiabatic_index_value=*/2.0,
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

// For a synthetic polytropic slice p = A n_b^gamma, eps = B n_b^(gamma-1)
// in CompOSE-natural units, this predicts the (rho_geom, p_geom, eps)
// values a query at a given n_b should return through Tabulated1D.
struct PolytropicExpected {
  double rho_geom;
  double p_geom;
  double eps;
};

PolytropicExpected expected_at_nb(double nb_fm3, const SliceSpec& spec) {
  constexpr double nb_fm3_to_geom = hydro::units::nuclear::neutron_mass /
                                    hydro::units::nuclear::pressure_unit;
  constexpr double press_MeV_to_geom =
      1.0 / hydro::units::nuclear::pressure_unit;
  const double rho_geom = nb_fm3_to_geom * nb_fm3;
  const double p_MeVfm3 =
      spec.pressure_amplitude * std::pow(nb_fm3, spec.gamma);
  const double p_geom = press_MeV_to_geom * p_MeVfm3;
  const double eps = spec.eps_amplitude * std::pow(nb_fm3, spec.gamma - 1.0);
  return {rho_geom, p_geom, eps};
}

// Verify pressure_from_density and specific_internal_energy_from_density
// return the correct values at grid points (exact interp) and at
// interior points (linear-in-log-log interpolation error).
template <bool IsRelativistic>
void check_pressure_and_eps_interpolation(
    const EquationsOfState::Tabulated1D<IsRelativistic>& eos,
    const SliceSpec& spec) {
  // Sample a range of n_b values inside the grid.
  const double log_lo = std::log(spec.nb_lo_fm3);
  const double log_hi = std::log(spec.nb_hi_fm3);
  const size_t n_samples = 25;
  for (size_t i = 0; i < n_samples; ++i) {
    const double frac =
        static_cast<double>(i) / static_cast<double>(n_samples - 1);
    const double nb = std::exp(log_lo + frac * (log_hi - log_lo));
    const auto expected = expected_at_nb(nb, spec);
    const Scalar<double> rho{expected.rho_geom};
    const auto p = eos.pressure_from_density(rho);
    const auto eps = eos.specific_internal_energy_from_density(rho);
    // For a polytrope, both log(p) and log(eps) are linear in log(rho),
    // so the log-log linear interp is analytically exact at every point
    // and agreement should be to roundoff (~1e-14).
    CHECK(get(p) == approx(expected.p_geom).epsilon(1e-12));
    CHECK(get(eps) == approx(expected.eps).epsilon(1e-12));
  }
}

// DataVector variant: bulk query on 100 random-in-grid densities matches
// pointwise queries and the analytic expected values.
template <bool IsRelativistic>
void check_datavector_query_matches_pointwise(
    const EquationsOfState::Tabulated1D<IsRelativistic>& eos,
    const SliceSpec& spec) {
  const size_t n = 100;
  DataVector rho(n);
  DataVector expected_p(n);
  DataVector expected_eps(n);
  const double log_lo = std::log(spec.nb_lo_fm3);
  const double log_hi = std::log(spec.nb_hi_fm3);
  for (size_t i = 0; i < n; ++i) {
    const double frac = static_cast<double>(i) / static_cast<double>(n - 1);
    const double nb = std::exp(log_lo + frac * (log_hi - log_lo));
    const auto e = expected_at_nb(nb, spec);
    rho[i] = e.rho_geom;
    expected_p[i] = e.p_geom;
    expected_eps[i] = e.eps;
  }
  const Scalar<DataVector> rho_scalar{rho};
  const auto p_bulk = eos.pressure_from_density(rho_scalar);
  const auto eps_bulk = eos.specific_internal_energy_from_density(rho_scalar);
  for (size_t i = 0; i < n; ++i) {
    CHECK(get(p_bulk)[i] == approx(expected_p[i]).epsilon(1e-12));
    CHECK(get(eps_bulk)[i] == approx(expected_eps[i]).epsilon(1e-12));
  }
}

// For the synthetic polytropic slice adiabatic_index is a constant (=
// spec.adiabatic_index_value). Chi_geom at any query rho should equal
// (p_geom / rho_geom) * adiabatic_index.
template <bool IsRelativistic>
void check_chi_interpolation(
    const EquationsOfState::Tabulated1D<IsRelativistic>& eos,
    const SliceSpec& spec) {
  const double log_lo = std::log(spec.nb_lo_fm3);
  const double log_hi = std::log(spec.nb_hi_fm3);
  const size_t n_samples = 25;
  for (size_t i = 0; i < n_samples; ++i) {
    const double frac =
        static_cast<double>(i) / static_cast<double>(n_samples - 1);
    const double nb = std::exp(log_lo + frac * (log_hi - log_lo));
    const auto expected = expected_at_nb(nb, spec);
    const Scalar<double> rho{expected.rho_geom};
    const auto chi = eos.chi_from_density(rho);
    const double chi_expected =
        (expected.p_geom / expected.rho_geom) * spec.adiabatic_index_value;
    // adiabatic_index is stored linearly and constant across the polytropic
    // slice; log(p) is linear in log(rho); the product is exact to
    // roundoff.
    CHECK(get(chi) == approx(chi_expected).epsilon(1e-12));
  }
}

// h -> rho inversion. Round-trip test: pick random rho in-grid, get h
// from specific_internal_energy + pressure, invert h -> rho, verify we
// recover the original rho.
template <bool IsRelativistic>
void check_h_to_rho_roundtrip(
    const EquationsOfState::Tabulated1D<IsRelativistic>& eos,
    const SliceSpec& spec) {
  const double log_lo = std::log(spec.nb_lo_fm3);
  const double log_hi = std::log(spec.nb_hi_fm3);
  const size_t n_samples = 25;
  for (size_t i = 0; i < n_samples; ++i) {
    const double frac =
        static_cast<double>(i) / static_cast<double>(n_samples - 1);
    const double nb = std::exp(log_lo + frac * (log_hi - log_lo));
    const auto expected = expected_at_nb(nb, spec);
    // Compute h from the analytic values (matches Tabulated1D::initialize).
    double h_expected = 0.0;
    if constexpr (IsRelativistic) {
      h_expected = 1.0 + expected.eps + expected.p_geom / expected.rho_geom;
    } else {
      h_expected = expected.eps + expected.p_geom / expected.rho_geom;
    }
    const Scalar<double> h_scalar{h_expected};
    const auto rho_recovered = eos.rest_mass_density_from_enthalpy(h_scalar);
    // Linear interp in log(rho) is not analytically exact for a
    // polytropic h(rho) (which is a nonlinear function of rho),
    // so we allow a modest interpolation-error tolerance. With 64
    // log-spaced grid points across 3 decades, the interp error is
    // O((delta log rho)^2 * curvature), typically < 0.1%.
    CHECK(get(rho_recovered) == approx(expected.rho_geom).epsilon(2e-3));
  }
}

// Out-of-range h queries should clamp to the endpoint density.
template <bool IsRelativistic>
void check_h_to_rho_clamps(
    const EquationsOfState::Tabulated1D<IsRelativistic>& eos) {
  // h at the upper end of the density grid — larger than any
  // physically-reachable h inside the table.
  const Scalar<double> rho_max{eos.rest_mass_density_upper_bound()};
  const double h_at_rho_max =
      1.0 + get(eos.specific_internal_energy_from_density(rho_max)) +
      get(eos.pressure_from_density(rho_max)) / get(rho_max);
  const double h_below = 0.5 * eos.specific_enthalpy_lower_bound();
  const double h_above = 2.0 * h_at_rho_max;
  const auto rho_below =
      eos.rest_mass_density_from_enthalpy(Scalar<double>{h_below});
  const auto rho_above =
      eos.rest_mass_density_from_enthalpy(Scalar<double>{h_above});
  CHECK(get(rho_below) == approx(eos.rest_mass_density_lower_bound()));
  CHECK(get(rho_above) == approx(eos.rest_mass_density_upper_bound()));
}

// Queries at rho outside the table range should clamp (matching
// Tabulated3D's convention) — no throw, no NaN.
template <bool IsRelativistic>
void check_out_of_range_clamps(
    const EquationsOfState::Tabulated1D<IsRelativistic>& eos) {
  const Scalar<double> rho_lo{0.5 * eos.rest_mass_density_lower_bound()};
  const Scalar<double> rho_hi{2.0 * eos.rest_mass_density_upper_bound()};
  const auto p_lo = eos.pressure_from_density(rho_lo);
  const auto p_hi = eos.pressure_from_density(rho_hi);
  CHECK(std::isfinite(get(p_lo)));
  CHECK(std::isfinite(get(p_hi)));
  // Clamped queries should return the table's endpoint values.
  const Scalar<double> rho_at_lo{eos.rest_mass_density_lower_bound()};
  const Scalar<double> rho_at_hi{eos.rest_mass_density_upper_bound()};
  CHECK(get(p_lo) == approx(get(eos.pressure_from_density(rho_at_lo))));
  CHECK(get(p_hi) == approx(get(eos.pressure_from_density(rho_at_hi))));
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
  check_pressure_and_eps_interpolation(eos_a_rel, spec_a);
  check_pressure_and_eps_interpolation(eos_a_nonrel, spec_a);
  check_datavector_query_matches_pointwise(eos_a_rel, spec_a);
  check_chi_interpolation(eos_a_rel, spec_a);
  check_chi_interpolation(eos_a_nonrel, spec_a);
  check_h_to_rho_roundtrip(eos_a_rel, spec_a);
  check_h_to_rho_clamps(eos_a_rel);
  check_out_of_range_clamps(eos_a_rel);

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
