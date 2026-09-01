// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/HllcGr.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace {
namespace helpers = TestHelpers::evolution::dg;

struct ConvertPolytropic {
  using unpacked_container = bool;
  using packed_container = EquationsOfState::EquationOfState<true, 3>;
  using packed_type = bool;

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    return true;
  }

  [[noreturn]] static inline void pack(
      const gsl::not_null<packed_container*> /*packed*/,
      const unpacked_container& /*unpacked*/,
      const size_t /*grid_point_index*/) {
    ERROR("Should not be converting an EOS from an unpacked to a packed type");
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.HllcGr",
                  "[Unit][GrMhd]") {
  PUPable_reg(grmhd::ValenciaDivClean::BoundaryCorrections::HllcGr);

  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections"};

  MAKE_GENERATOR(gen);

  using system = grmhd::ValenciaDivClean::System;
  using HllcGr = grmhd::ValenciaDivClean::BoundaryCorrections::HllcGr;

  const HllcGr hllc_boundary_correction{1.0e-30, 1.0e-8};

  const Mesh<2> mesh{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss};

  const tuples::TaggedTuple<hydro::Tags::GrmhdEquationOfState> volume_data{
      EquationsOfState::PolytropicFluid<true>{100.0, 2.0}.promote_to_3d_eos()};

  // General random-background conservation test. This checks that the
  // numerical flux is antisymmetric under exchanging the two sides of the
  // interface.
  //
  // ZeroOnSmoothSolution::No is used because the present MB05 star-state
  // formulas are implemented directly in coordinate-frame Valencia
  // variables. They exactly satisfy the identical-state consistency
  // property in Minkowski spacetime, but not necessarily for arbitrary
  // lapse and shift.
  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>>
      ranges{std::array{0.3, 1.0}, std::array{0.01, 0.02}};

  helpers::test_boundary_correction_conservation<system>(
      make_not_null(&gen), hllc_boundary_correction, mesh, volume_data, ranges,
      helpers::ZeroOnSmoothSolution::No);

  // Minkowski conservation test. The `test_boundary_correction_conservation`
  // helper populates state variables and fluxes with independent random
  // values, so the flux tensor is not the physical flux of the state. HLL
  // is robust to this (it never invokes the F = F(U) identity), but HLLC's
  // MB05 quadratic assumes F(E) = m^x, so identical L = R inputs with
  // inconsistent packaged fluxes do not produce an exactly zero correction.
  // We therefore use `ZeroOnSmoothSolution::No` here as well; the smooth-
  // solution property is verified in practice by the Python cross-check
  // below, which computes fluxes consistently with state.
  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>,
      helpers::Tags::Range<grmhd::ValenciaDivClean::Tags::TildeB<>>>
      ranges_minkowski{std::array{1.0, 1.0}, std::array{0.0, 0.0},
                       std::array{0.0, 0.0}};

  helpers::test_boundary_correction_conservation<system>(
      make_not_null(&gen), hllc_boundary_correction, mesh, volume_data,
      ranges_minkowski, helpers::ZeroOnSmoothSolution::No);

  // General cross-check against the Python reference implementation.
  helpers::test_boundary_correction_with_python<system,
                                                tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "HllcGr", "dg_package_data", "dg_boundary_terms",
      hllc_boundary_correction, mesh, volume_data, ranges);

  // Pure-hydrodynamic branch. B is exactly zero and the density is above
  // the atmosphere cutoff, so this exercises the hydro characteristic
  // speeds and the full HLLC star-region calculation, including:
  //
  // - the HLL averages;
  // - the quadratic solve for lambda_star;
  // - the corrected star-pressure expression;
  // - the left/right star-state construction.
  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>,
      helpers::Tags::Range<grmhd::ValenciaDivClean::Tags::TildeB<>>>
      ranges_hydro{std::array{0.3, 1.0}, std::array{0.01, 0.02},
                   std::array{0.0, 0.0}};

  helpers::test_boundary_correction_with_python<system,
                                                tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "HllcGr", "dg_package_data", "dg_boundary_terms",
      hllc_boundary_correction, mesh, volume_data, ranges_hydro);

  // Explicit nonzero-magnetic-field branch. The magnetic-field magnitude
  // is far above MagneticFieldMagnitudeForHydro, so the packaged outer
  // characteristic speeds use the light-speed bounds. The hydrodynamic
  // variables are still treated by the HLLC star-state calculation, while
  // the magnetic and divergence-cleaning variables use HLL fluxes.
  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>,
      helpers::Tags::Range<grmhd::ValenciaDivClean::Tags::TildeB<>>>
      ranges_nonzero_magnetic_field{std::array{0.3, 1.0},
                                    std::array{0.01, 0.02},
                                    std::array{1.0e-4, 1.0e-3}};

  helpers::test_boundary_correction_with_python<system,
                                                tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "HllcGr", "dg_package_data", "dg_boundary_terms",
      hllc_boundary_correction, mesh, volume_data,
      ranges_nonzero_magnetic_field);

  // Below-atmosphere density: use light-speed characteristic bounds even
  // when the magnetic field is negligible.
  const tuples::TaggedTuple<
      helpers::Tags::Range<hydro::Tags::RestMassDensity<DataVector>>,
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>,
      helpers::Tags::Range<grmhd::ValenciaDivClean::Tags::TildeB<>>>
      ranges_atmosphere{std::array{1.0e-10, 1.0e-9}, std::array{0.3, 1.0},
                        std::array{0.01, 0.02}, std::array{0.0, 0.0}};

  helpers::test_boundary_correction_with_python<system,
                                                tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "HllcGr", "dg_package_data", "dg_boundary_terms",
      hllc_boundary_correction, mesh, volume_data, ranges_atmosphere);

  // Factory creation and serialization options.
  const auto factory_hllc =
      TestHelpers::test_factory_creation<evolution::BoundaryCorrection, HllcGr>(
          "HllcGr:\n"
          "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
          "  LightSpeedDensityCutoff: 1.0e-8\n");

  helpers::test_boundary_correction_with_python<system,
                                                tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "HllcGr", "dg_package_data", "dg_boundary_terms",
      dynamic_cast<const HllcGr&>(*factory_hllc), mesh, volume_data, ranges);

  // Equality operators.
  CHECK_FALSE(HllcGr{1.0e-30, 1.0e-8} != HllcGr{1.0e-30, 1.0e-8});

  CHECK(HllcGr{1.0e-30, 1.0e-8} != HllcGr{2.0e-30, 1.0e-8});

  CHECK(HllcGr{1.0e-30, 1.0e-8} != HllcGr{1.0e-30, 2.0e-8});
}
}  // namespace
