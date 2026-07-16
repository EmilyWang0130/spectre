// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Marquina.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/MarquinaCpm.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.MarquinaCpm",
                  "[Unit][GrMhd]") {
  PUPable_reg(grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCpm);
  MAKE_GENERATOR(gen);

  using system = grmhd::ValenciaDivClean::System;
  namespace helpers = TestHelpers::evolution::dg;

  const tuples::TaggedTuple<hydro::Tags::GrmhdEquationOfState> volume_data{
      EquationsOfState::IdealFluid<true>{1.5, 0.0}.promote_to_3d_eos()};

  const tuples::TaggedTuple<
      helpers::Tags::Range<hydro::Tags::RestMassDensity<DataVector>>,
      helpers::Tags::Range<hydro::Tags::SpecificInternalEnergy<DataVector>>,
      helpers::Tags::Range<
          hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>,
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3, Frame::Inertial>>>
      ranges(std::array<double, 2>{{0.1, 1.0}},    // Density
             std::array<double, 2>{{0.1, 1.0}},    // Internal Energy
             std::array<double, 2>{{0.0, 0.5}},    // Velocity
             std::array<double, 2>{{0.5, 1.0}},    // Lapse
             std::array<double, 2>{{-0.1, 0.1}});  // Shift

  // The Complementary Projection Method is analytically identical to the
  // full-decomposition Marquina flux at every face point (see the equivalence
  // proof in spectre_runs/cpm_marquina_plan.md), so it must satisfy
  // conservation with the same tolerance.
  for (int i = 0; i < 1000; ++i) {
    TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
        make_not_null(&gen),
        grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCpm{},
        Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
        volume_data, ranges, helpers::ZeroOnSmoothSolution::Yes, 1.0e-12, true);
  }

  const auto marquina_cpm = TestHelpers::test_factory_creation<
      evolution::BoundaryCorrection,
      grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCpm>(
      "MarquinaCpm:");

  // Differential test: verify that MarquinaCpm and full-decomposition
  // Marquina produce numerically identical boundary corrections on the
  // same random face state. Each outer iteration exercises the full
  // dg_package_data + dg_boundary_terms path for both schemes under both
  // dg_formulations and both mesh-velocity settings, so 1000 iterations
  // gives 4000 diff checks over 25 face points each = 100k per-component
  // comparisons per conserved variable. Tolerance 1e-12 matches the
  // conservation-test tolerance and is well above the ~1e-14 arithmetic
  // floor for round-off in this size of dot-product / matrix-assembly
  // arithmetic.
  PUPable_reg(grmhd::ValenciaDivClean::BoundaryCorrections::Marquina);
  for (int i = 0; i < 1000; ++i) {
    TestHelpers::evolution::dg::test_boundary_correction_agreement<system>(
        make_not_null(&gen),
        grmhd::ValenciaDivClean::BoundaryCorrections::Marquina{},
        grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCpm{},
        Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
        volume_data, ranges, 1.0e-12, true);
  }
}
