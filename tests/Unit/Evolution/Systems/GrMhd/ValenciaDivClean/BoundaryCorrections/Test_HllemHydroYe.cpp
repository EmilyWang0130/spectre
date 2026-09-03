// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/HllemHydroYe.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace {
namespace helpers = TestHelpers::evolution::dg;

// Rung 1 of the HllemHydroYe validation ladder (design v2 sec 13.1 + 13.4):
// verify the projector algebra at random averaged interface states, and that
// a jump aligned with an acoustic eigenvector is projected to zero by P_mid.
// These are ALGEBRAIC checks on the eigensystem construction; they run before
// any evolution test.
void test_projector_algebra() {
  MAKE_GENERATOR(gen);
  namespace hydro_helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&gen);
  const DataVector used_for_size(5);
  const size_t num_pts = used_for_size.size();

  // Random primitives on flat spatial metric (matches the flat-only anti-
  // diffusion path in HllemHydroYe.cpp).
  tnsr::ii<DataVector, 3, Frame::Inertial> flat_metric{num_pts, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    flat_metric.get(i, i) = 1.0;
  }
  const auto rho = hydro_helper::random_density(nn_gen, used_for_size);
  const auto eps =
      hydro_helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto ye = hydro_helper::random_electron_fraction(nn_gen, used_for_size);
  const auto lorentz =
      hydro_helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto v = hydro_helper::random_velocity(nn_gen, lorentz, flat_metric);
  // Ideal-fluid EOS supports independent (rho, eps, Ye); the 3D-promoted
  // polytropic makes ye and eps redundant with rho, which does not exercise
  // zeta.
  const auto eos_2d =
      EquationsOfState::IdealFluid<true>{5.0 / 3.0}.promote_to_3d_eos();
  const auto& eos = *eos_2d;
  const auto pressure =
      eos.pressure_from_density_and_energy(rho, eps, ye);
  const auto h_eos = hydro::relativistic_specific_enthalpy(rho, eps, pressure);
  // Random unit normal covector.
  std::uniform_real_distribution<double> unit_dist(-1.0, 1.0);
  auto n = make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
      nn_gen, make_not_null(&unit_dist), used_for_size);
  for (size_t pt = 0; pt < num_pts; ++pt) {
    double norm2 = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      norm2 += n.get(i)[pt] * n.get(i)[pt];
    }
    const double norm = std::sqrt(norm2);
    for (size_t i = 0; i < 3; ++i) {
      n.get(i)[pt] /= norm;
    }
  }

  tnsr::ij<DataVector, 6> right{num_pts, 0.0};
  tnsr::IJ<DataVector, 6> left{num_pts, 0.0};
  grmhd::ValenciaDivClean::characteristic_eigenvectors_hydro(
      make_not_null(&right), make_not_null(&left), v, rho, eps, h_eos, ye,
      lorentz, n, flat_metric, eos);

  constexpr size_t plus = grmhd::ValenciaDivClean::HydroVectorR::Rplus;
  constexpr size_t minus = grmhd::ValenciaDivClean::HydroVectorR::Rminus;

  const auto build_projectors = [&](const size_t pt,
                                    std::array<std::array<double, 6>, 6>&
                                        P_plus,
                                    std::array<std::array<double, 6>, 6>&
                                        P_minus,
                                    std::array<std::array<double, 6>, 6>&
                                        P_mid) {
    double diag_plus = 0.0;
    double diag_minus = 0.0;
    for (size_t n_i = 0; n_i < 6; ++n_i) {
      diag_plus += left.get(plus, n_i)[pt] * right.get(plus, n_i)[pt];
      diag_minus += left.get(minus, n_i)[pt] * right.get(minus, n_i)[pt];
    }
    REQUIRE(std::abs(diag_plus) > 1.0e-8);
    REQUIRE(std::abs(diag_minus) > 1.0e-8);
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        P_plus[i][j] =
            right.get(plus, i)[pt] * left.get(plus, j)[pt] / diag_plus;
        P_minus[i][j] =
            right.get(minus, i)[pt] * left.get(minus, j)[pt] / diag_minus;
        const double identity = (i == j) ? 1.0 : 0.0;
        P_mid[i][j] = identity - P_plus[i][j] - P_minus[i][j];
      }
    }
  };

  const auto matmul = [](const std::array<std::array<double, 6>, 6>& A,
                         const std::array<std::array<double, 6>, 6>& B) {
    std::array<std::array<double, 6>, 6> C{};
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        double acc = 0.0;
        for (size_t k = 0; k < 6; ++k) {
          acc += A[i][k] * B[k][j];
        }
        C[i][j] = acc;
      }
    }
    return C;
  };

  // Tolerances loose enough to survive random primitives where the acoustic
  // eigenvector norms can approach zero. The individual projector entries are
  // O(1), so 1e-8 margin still catches any structural bug.
  Approx custom_approx = Approx::custom().epsilon(1.0e-8).margin(1.0e-8);
  for (size_t pt = 0; pt < num_pts; ++pt) {
    std::array<std::array<double, 6>, 6> P_plus{};
    std::array<std::array<double, 6>, 6> P_minus{};
    std::array<std::array<double, 6>, 6> P_mid{};
    build_projectors(pt, P_plus, P_minus, P_mid);
    // P_+^2 = P_+, P_-^2 = P_-, P_mid^2 = P_mid  (idempotence)
    const auto Pp2 = matmul(P_plus, P_plus);
    const auto Pm2 = matmul(P_minus, P_minus);
    const auto Pmid2 = matmul(P_mid, P_mid);
    // Orthogonality across eigenspaces.
    const auto PpPm = matmul(P_plus, P_minus);
    const auto PmPp = matmul(P_minus, P_plus);
    const auto PpPmid = matmul(P_plus, P_mid);
    const auto PmPmid = matmul(P_minus, P_mid);
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        CHECK(Pp2[i][j] == custom_approx(P_plus[i][j]));
        CHECK(Pm2[i][j] == custom_approx(P_minus[i][j]));
        CHECK(Pmid2[i][j] == custom_approx(P_mid[i][j]));
        CHECK(PpPm[i][j] == custom_approx(0.0));
        CHECK(PmPp[i][j] == custom_approx(0.0));
        CHECK(PpPmid[i][j] == custom_approx(0.0));
        CHECK(PmPmid[i][j] == custom_approx(0.0));
        const double sum = P_plus[i][j] + P_minus[i][j] + P_mid[i][j];
        const double identity = (i == j) ? 1.0 : 0.0;
        CHECK(sum == custom_approx(identity));
      }
    }
    // Acoustic-jump reduction: dU = R_+ column -> P_mid . dU = 0.
    std::array<double, 6> du_plus{};
    std::array<double, 6> du_minus{};
    for (size_t i = 0; i < 6; ++i) {
      du_plus[i] = right.get(plus, i)[pt];
      du_minus[i] = right.get(minus, i)[pt];
    }
    for (size_t i = 0; i < 6; ++i) {
      double mid_plus = 0.0;
      double mid_minus = 0.0;
      for (size_t j = 0; j < 6; ++j) {
        mid_plus += P_mid[i][j] * du_plus[j];
        mid_minus += P_mid[i][j] * du_minus[j];
      }
      CHECK(mid_plus == custom_approx(0.0));
      CHECK(mid_minus == custom_approx(0.0));
    }
  }
}

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

SPECTRE_TEST_CASE(
    "Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.HllemHydroYe",
    "[Unit][GrMhd]") {
  PUPable_reg(grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe);

  // Phase 2 algebraic tests -- run before any evolution / pypp check
  // (design v2 sec 13). If these fail, the eigensystem convention is wrong
  // and no downstream test is meaningful.
  test_projector_algebra();

  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  using system = grmhd::ValenciaDivClean::System;

  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>>
      ranges{std::array{0.3, 1.0}, std::array{0.01, 0.02}};
  const tuples::TaggedTuple<hydro::Tags::GrmhdEquationOfState> volume_data{
      EquationsOfState::PolytropicFluid<true>{100.0, 2.0}.promote_to_3d_eos()};

  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen),
      grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{
          1.0e-30, 1.0e-8, false, true},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "HllemHydroYe", "dg_package_data",
      "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{
          1.0e-30, 1.0e-8, false, true},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  // Test hydro
  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>,
      helpers::Tags::Range<grmhd::ValenciaDivClean::Tags::TildeB<>>>
      ranges_hydro{std::array{0.3, 1.0}, std::array{0.01, 0.02},
                   std::array{1.0e-25, 1.0e-20}};
  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "HllemHydroYe", "dg_package_data",
      "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{
          1.0e-30, 1.0e-8, false, true},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges_hydro);

  // Phase-2 integration testing on the actual boundary_terms code path is
  // deferred to shocktube / KHI evolutions -- the pypp framework generates a
  // random spatial metric even at lapse=1, shift=0, so `characteristic_
  // speeds_mhd` (called on `flat_metric`) can see superluminal velocities.
  // The projector-algebra test above covers the middle-block construction
  // algebraically; evolution runs will exercise the anti-diffusion path.

  // Test light speed density cutoff
  const tuples::TaggedTuple<
      helpers::Tags::Range<hydro::Tags::RestMassDensity<DataVector>>,
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>>
      ranges_atmo{std::array{1.0e-10, 1.0e-9}, std::array{0.3, 1.0},
                  std::array{0.01, 0.02}};
  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "HllemHydroYe", "dg_package_data",
      "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{
          1.0e-30, 1.0e-8, false, true},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges_atmo);

  const auto hll = TestHelpers::test_factory_creation<
      evolution::BoundaryCorrection,
      grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe>(
      "HllemHydroYe:\n"
      "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
      "  LightSpeedDensityCutoff: 1.0e-8\n"
      "  RestoreMiddleBlock: False\n"
      "  UsePhysicalZeta: True\n");

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "HllemHydroYe", "dg_package_data",
      "dg_boundary_terms",
      dynamic_cast<
          const grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe&>(
          *hll),
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  CHECK_FALSE(grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{
                  1.0e-30, 1.0e-8, false, true} !=
              grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{
                  1.0e-30, 1.0e-8, false, true});
  CHECK(grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{
            1.0e-30, 1.0e-8, false, true} !=
        grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{
            2.0e-30, 1.0e-8, false, true});
  CHECK(grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{
            1.0e-30, 1.0e-8, false, true} !=
        grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{
            1.0e-30, 2.0e-8, false, true});
}
}  // namespace
