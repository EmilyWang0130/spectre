// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/FirstOrder.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/PrimReconstructor.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GrMhd.ValenciaDivClean.Fd.FirstOrderPrim",
    "[Unit][Evolution]") {
  // Note: the linear-reconstruction accuracy helper (test_prim_reconstructor)
  // is not used here because first-order (piecewise-constant) reconstruction is
  // exact only for constants, not for the linear prims that helper sets up.
  // The reconstruction itself is validated by the shock-tube exact-solution
  // comparison. Here we test factory creation, options, and equality.
  const grmhd::ValenciaDivClean::fd::FirstOrderPrim recons{false};
  const auto from_options_base = TestHelpers::test_factory_creation<
      grmhd::ValenciaDivClean::fd::Reconstructor,
      grmhd::ValenciaDivClean::fd::OptionTags::Reconstructor>(
      "FirstOrderPrim:\n"
      "  ReconstructRhoTimesTemperature: false\n");
  auto* const from_options =
      dynamic_cast<const grmhd::ValenciaDivClean::fd::FirstOrderPrim*>(
          from_options_base.get());
  REQUIRE(from_options != nullptr);
  CHECK(*from_options == recons);
  CHECK(recons.ghost_zone_size() == 2);

  const grmhd::ValenciaDivClean::fd::FirstOrderPrim recons_rho_times_t{true};
  CHECK_FALSE(recons_rho_times_t == recons);
  CHECK(recons_rho_times_t != recons);
}
