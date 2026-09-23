// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/HllcGr.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/HllemHydroYe.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SoundSpeedSquared.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"

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

  // Random primitives on a random CURVED spatial metric. The anti-diffusion
  // path is no longer flat-space-only: `characteristic_eigenvectors_hydro`
  // builds the Eulerian-frame eigensystem for a general gamma_ij, and the
  // coordinate-frame Jacobian differs from it only by
  // A_coord = alpha A_Eul - beta^n I, which leaves the eigenvectors -- and
  // therefore the projectors tested here -- unchanged.
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto rho = hydro_helper::random_density(nn_gen, used_for_size);
  const auto eps =
      hydro_helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto ye = hydro_helper::random_electron_fraction(nn_gen, used_for_size);
  const auto lorentz =
      hydro_helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto v = hydro_helper::random_velocity(nn_gen, lorentz, spatial_metric);
  // Ideal-fluid EOS supports independent (rho, eps, Ye); the 3D-promoted
  // polytropic makes ye and eps redundant with rho, which does not exercise
  // zeta.
  const auto eos_2d =
      EquationsOfState::IdealFluid<true>{5.0 / 3.0}.promote_to_3d_eos();
  const auto& eos = *eos_2d;
  const auto pressure = eos.pressure_from_density_and_energy(rho, eps, ye);
  const auto h_eos = hydro::relativistic_specific_enthalpy(rho, eps, pressure);
  // Random normal covector, normalized in the (curved) spatial metric so that
  // gamma^ij n_i n_j = 1, which is what the eigensystem assumes.
  std::uniform_real_distribution<double> unit_dist(-1.0, 1.0);
  auto n = make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
      nn_gen, make_not_null(&unit_dist), used_for_size);
  for (size_t pt = 0; pt < num_pts; ++pt) {
    double norm2 = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        norm2 += inv_spatial_metric.get(i, j)[pt] * n.get(i)[pt] * n.get(j)[pt];
      }
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
      lorentz, n, spatial_metric, eos);

  constexpr size_t plus = grmhd::ValenciaDivClean::HydroVectorR::Rplus;
  constexpr size_t minus = grmhd::ValenciaDivClean::HydroVectorR::Rminus;

  const auto build_projectors =
      [&](const size_t pt, std::array<std::array<double, 6>, 6>& P_plus,
          std::array<std::array<double, 6>, 6>& P_minus,
          std::array<std::array<double, 6>, 6>& P_mid) {
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

  // Tolerances are scaled by the size of the projector entries at each point.
  // The acoustic eigenvectors are biorthogonal but not biorthonormal, so on
  // an ill-conditioned random draw (small L_+ . R_+) the individual entries of
  // P_+ and P_- can be many orders of magnitude larger than one, and with them
  // the absolute round-off in the products below. A fixed 1e-8 absolute margin
  // made this test flaky at roughly the 1-in-15 level. A structural error in
  // the eigensystem convention shows up as an O(1) *relative* error, which the
  // scaled margin still catches.
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
    double projector_scale = 1.0;
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        projector_scale =
            std::max({projector_scale, std::abs(P_plus[i][j]),
                      std::abs(P_minus[i][j]), std::abs(P_mid[i][j])});
      }
    }
    Approx custom_approx =
        Approx::custom().epsilon(1.0e-8).margin(1.0e-8 * projector_scale);
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

// Packaged-data bundle for one side of an interface, so the two GR tests
// below can call `dg_boundary_terms` directly without going through the
// pypp helper (which has no Python reference for
// `characteristic_eigenvectors_hydro`).
struct SidePackage {
  Scalar<DataVector> tilde_d{};
  Scalar<DataVector> tilde_ye{};
  Scalar<DataVector> tilde_tau{};
  tnsr::i<DataVector, 3, Frame::Inertial> tilde_s{};
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_b{};
  Scalar<DataVector> tilde_phi{};
  Scalar<DataVector> nf_tilde_d{};
  Scalar<DataVector> nf_tilde_ye{};
  Scalar<DataVector> nf_tilde_tau{};
  tnsr::i<DataVector, 3, Frame::Inertial> nf_tilde_s{};
  tnsr::I<DataVector, 3, Frame::Inertial> nf_tilde_b{};
  Scalar<DataVector> nf_tilde_phi{};
  Scalar<DataVector> largest_outgoing{};
  Scalar<DataVector> largest_ingoing{};
  tnsr::i<DataVector, 3, Frame::Inertial> normal{};
  Scalar<DataVector> lapse{};
  Scalar<DataVector> shift_dot_normal{};
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{};
  Scalar<DataVector> rest_mass_density{};
  Scalar<DataVector> electron_fraction{};
  Scalar<DataVector> sound_speed_squared{};
  Scalar<DataVector> temperature{};
  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{};
  Scalar<DataVector> pressure{};
  Scalar<DataVector> lorentz_factor{};
  Scalar<DataVector> specific_internal_energy{};
};

// Six-slot view of a boundary correction, in the slot order used by
// `characteristic_eigenvectors_hydro`: [D, S_x, S_y, S_z, tau, D Y_e].
struct Correction {
  Scalar<DataVector> tilde_d{};
  Scalar<DataVector> tilde_ye{};
  Scalar<DataVector> tilde_tau{};
  tnsr::i<DataVector, 3, Frame::Inertial> tilde_s{};
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_b{};
  Scalar<DataVector> tilde_phi{};

  explicit Correction(const size_t num_pts)
      : tilde_d{DataVector{num_pts, 0.0}},
        tilde_ye{DataVector{num_pts, 0.0}},
        tilde_tau{DataVector{num_pts, 0.0}},
        tilde_s{num_pts, 0.0},
        tilde_b{num_pts, 0.0},
        tilde_phi{DataVector{num_pts, 0.0}} {}

  double slot(const size_t n, const size_t pt) const {
    switch (n) {
      case 0:
        return get(tilde_d)[pt];
      case 1:
      case 2:
      case 3:
        return tilde_s.get(n - 1)[pt];
      case 4:
        return get(tilde_tau)[pt];
      default:
        return get(tilde_ye)[pt];
    }
  }
};

Correction apply_boundary_terms(
    const grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe&
        correction,
    const SidePackage& in, const SidePackage& out,
    const dg::Formulation formulation,
    const EquationsOfState::EquationOfState<true, 3>& eos) {
  Correction result{get(in.tilde_d).size()};
  correction.dg_boundary_terms(
      make_not_null(&result.tilde_d), make_not_null(&result.tilde_ye),
      make_not_null(&result.tilde_tau), make_not_null(&result.tilde_s),
      make_not_null(&result.tilde_b), make_not_null(&result.tilde_phi),
      in.tilde_d, in.tilde_ye, in.tilde_tau, in.tilde_s, in.tilde_b,
      in.tilde_phi, in.nf_tilde_d, in.nf_tilde_ye, in.nf_tilde_tau,
      in.nf_tilde_s, in.nf_tilde_b, in.nf_tilde_phi, in.largest_outgoing,
      in.largest_ingoing, in.normal, in.lapse, in.shift_dot_normal,
      in.spatial_metric, in.rest_mass_density, in.electron_fraction,
      in.sound_speed_squared, in.temperature, in.spatial_velocity, in.pressure,
      in.lorentz_factor, in.specific_internal_energy, out.tilde_d, out.tilde_ye,
      out.tilde_tau, out.tilde_s, out.tilde_b, out.tilde_phi, out.nf_tilde_d,
      out.nf_tilde_ye, out.nf_tilde_tau, out.nf_tilde_s, out.nf_tilde_b,
      out.nf_tilde_phi, out.largest_outgoing, out.largest_ingoing, out.normal,
      out.lapse, out.shift_dot_normal, out.spatial_metric,
      out.rest_mass_density, out.electron_fraction, out.sound_speed_squared,
      out.temperature, out.spatial_velocity, out.pressure, out.lorentz_factor,
      out.specific_internal_energy, formulation, eos);
  return result;
}

// Curved-space stationary-contact exactness (design v2 sec 13.3, promoted to
// GR). Build an interface with alpha != 1, beta^n != 0 and a curved
// gamma_ij, at a contact that is stationary *in the coordinate frame*:
//
//   lambda_mid = alpha (v . n) - beta^n_eff = 0 .
//
// Give the two sides the same physical normal flux f and a jump lying
// entirely inside the middle block, Delta U = P_mid w. Then
// delta_mid = 1, P_mid Delta U = Delta U, and the HLL dissipation cancels
// the anti-diffusion exactly, so the numerical flux must be the physical
// flux f (weak form) and the strong-form correction must vanish.
//
// This is the test that pins lambda_mid to the COORDINATE-frame value: with
// the Eulerian-frame lambda_mid = v . n that the flat-space code used,
// delta_mid = 1 - (v.n)/S_R != 1 and the identity fails by O(v.n / S_R).
void test_gr_stationary_contact() {
  MAKE_GENERATOR(gen);
  namespace hydro_helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&gen);
  const DataVector used_for_size(5);
  const size_t num_pts = used_for_size.size();

  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto rho = hydro_helper::random_density(nn_gen, used_for_size);
  const auto eps =
      hydro_helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto ye = hydro_helper::random_electron_fraction(nn_gen, used_for_size);
  const auto lorentz =
      hydro_helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto v = hydro_helper::random_velocity(nn_gen, lorentz, spatial_metric);
  const auto eos_2d =
      EquationsOfState::IdealFluid<true>{5.0 / 3.0}.promote_to_3d_eos();
  const auto& eos = *eos_2d;
  const auto pressure = eos.pressure_from_density_and_energy(rho, eps, ye);
  const auto h_eos = hydro::relativistic_specific_enthalpy(rho, eps, pressure);
  const Scalar<DataVector> cs2{clamp(
      get(eos.sound_speed_squared_from_density_and_temperature(
          rho, eos.temperature_from_density_and_energy(rho, eps, ye), ye)),
      1.0e-4, 0.9)};

  // Normal covector, unit in the spatial metric.
  std::uniform_real_distribution<double> unit_dist(-1.0, 1.0);
  auto n = make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
      nn_gen, make_not_null(&unit_dist), used_for_size);
  DataVector v_dot_n{num_pts, 0.0};
  for (size_t pt = 0; pt < num_pts; ++pt) {
    double norm2 = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        norm2 += inv_spatial_metric.get(i, j)[pt] * n.get(i)[pt] * n.get(j)[pt];
      }
    }
    const double norm = std::sqrt(norm2);
    for (size_t i = 0; i < 3; ++i) {
      n.get(i)[pt] /= norm;
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    v_dot_n += v.get(i) * n.get(i);
  }

  // Lapse well away from 1 and a shift chosen so that the contact is
  // stationary in coordinates. beta^n is packaged with each side's own
  // outward normal, so the exterior copy carries the opposite sign; the
  // antisymmetric average in dg_boundary_terms then recovers alpha (v.n).
  std::uniform_real_distribution<double> lapse_dist(0.4, 0.9);
  auto lapse = make_with_random_values<Scalar<DataVector>>(
      nn_gen, make_not_null(&lapse_dist), used_for_size);
  const DataVector beta_n = get(lapse) * v_dot_n;

  // Right eigenvectors / left projectors at the (identical on both sides,
  // hence exactly averaged) interface state, so the test can build a jump
  // that lies in the middle block.
  tnsr::ij<DataVector, 6> right{num_pts, 0.0};
  tnsr::IJ<DataVector, 6> left{num_pts, 0.0};
  grmhd::ValenciaDivClean::characteristic_eigenvectors_hydro(
      make_not_null(&right), make_not_null(&left), v, rho, eps, h_eos, ye,
      lorentz, n, spatial_metric, eos);
  constexpr size_t plus = grmhd::ValenciaDivClean::HydroVectorR::Rplus;
  constexpr size_t minus = grmhd::ValenciaDivClean::HydroVectorR::Rminus;

  std::uniform_real_distribution<double> jump_dist(-0.2, 0.2);
  const auto raw_jump = make_with_random_values<tnsr::i<DataVector, 6>>(
      nn_gen, make_not_null(&jump_dist), used_for_size);
  // du_mid = P_mid . raw_jump; du_acoustic = R_+ (annihilated by P_mid).
  tnsr::i<DataVector, 6> du_mid{num_pts, 0.0};
  tnsr::i<DataVector, 6> du_acoustic{num_pts, 0.0};
  // Round-off amplification of the P_mid projection at each point. The
  // acoustic eigenvectors are biorthogonal but not biorthonormal, so
  // P_pm = R_pm (x) L_pm / (L_pm . R_pm) has entries of size
  // |R| |L| / |L . R|, which on an ill-conditioned random draw is many orders
  // of magnitude above one. Every identity below is exact in real arithmetic
  // and accurate to round-off TIMES this factor, so the tolerances are scaled
  // by it. A wrong lambda_mid (the flat-space Eulerian v.n instead of the
  // coordinate-frame alpha v.n - beta^n) breaks the identity by
  // O(delta_mid deviation) ~ 1e-1 relative, which the scaled tolerance still
  // catches by many orders of magnitude.
  DataVector projection_conditioning{num_pts, 1.0};
  for (size_t pt = 0; pt < num_pts; ++pt) {
    double diag_plus = 0.0;
    double diag_minus = 0.0;
    double coeff_plus = 0.0;
    double coeff_minus = 0.0;
    double max_right_plus = 0.0;
    double max_right_minus = 0.0;
    double max_left_plus = 0.0;
    double max_left_minus = 0.0;
    for (size_t k = 0; k < 6; ++k) {
      diag_plus += left.get(plus, k)[pt] * right.get(plus, k)[pt];
      diag_minus += left.get(minus, k)[pt] * right.get(minus, k)[pt];
      coeff_plus += left.get(plus, k)[pt] * raw_jump.get(k)[pt];
      coeff_minus += left.get(minus, k)[pt] * raw_jump.get(k)[pt];
      max_right_plus =
          std::max(max_right_plus, std::abs(right.get(plus, k)[pt]));
      max_right_minus =
          std::max(max_right_minus, std::abs(right.get(minus, k)[pt]));
      max_left_plus = std::max(max_left_plus, std::abs(left.get(plus, k)[pt]));
      max_left_minus =
          std::max(max_left_minus, std::abs(left.get(minus, k)[pt]));
    }
    REQUIRE(std::abs(diag_plus) > 1.0e-8);
    REQUIRE(std::abs(diag_minus) > 1.0e-8);
    projection_conditioning[pt] =
        std::max({1.0, max_right_plus * max_left_plus / std::abs(diag_plus),
                  max_right_minus * max_left_minus / std::abs(diag_minus)});
    coeff_plus /= diag_plus;
    coeff_minus /= diag_minus;
    for (size_t k = 0; k < 6; ++k) {
      du_mid.get(k)[pt] = raw_jump.get(k)[pt] -
                          coeff_plus * right.get(plus, k)[pt] -
                          coeff_minus * right.get(minus, k)[pt];
      du_acoustic.get(k)[pt] = right.get(plus, k)[pt];
    }
  }

  // Arbitrary (but identical) physical normal flux on the two sides, and an
  // arbitrary base state. Only the JUMP matters for the identity being
  // tested, so the base state is unconstrained.
  std::uniform_real_distribution<double> flux_dist(-0.5, 0.5);
  const auto base_flux = make_with_random_values<tnsr::i<DataVector, 6>>(
      nn_gen, make_not_null(&flux_dist), used_for_size);
  std::uniform_real_distribution<double> state_dist(0.5, 1.5);
  const auto base_state = make_with_random_values<tnsr::i<DataVector, 6>>(
      nn_gen, make_not_null(&state_dist), used_for_size);

  const auto build_sides = [&](const tnsr::i<DataVector, 6>& jump,
                               const gsl::not_null<SidePackage*> in,
                               const gsl::not_null<SidePackage*> out) {
    for (SidePackage* side : {in.get(), out.get()}) {
      side->tilde_b = tnsr::I<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
      side->nf_tilde_b = tnsr::I<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
      side->tilde_phi = Scalar<DataVector>{DataVector{num_pts, 0.0}};
      side->nf_tilde_phi = Scalar<DataVector>{DataVector{num_pts, 0.0}};
      side->spatial_metric = spatial_metric;
      side->rest_mass_density = rho;
      side->electron_fraction = ye;
      side->sound_speed_squared = cs2;
      side->temperature = rho;
      side->spatial_velocity = v;
      side->pressure = pressure;
      side->lorentz_factor = lorentz;
      side->specific_internal_energy = eps;
      side->lapse = lapse;
      side->tilde_s = tnsr::i<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
      side->nf_tilde_s = tnsr::i<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
    }
    in->normal = n;
    out->normal = n;
    for (size_t i = 0; i < 3; ++i) {
      out->normal.get(i) *= -1.0;
    }
    // beta^n and the light-speed bounds carried by each side, signed with
    // that side's own outward normal.
    in->shift_dot_normal = Scalar<DataVector>{beta_n};
    out->shift_dot_normal = Scalar<DataVector>{-beta_n};
    in->largest_outgoing = Scalar<DataVector>{get(lapse) - beta_n};
    in->largest_ingoing = Scalar<DataVector>{-get(lapse) - beta_n};
    out->largest_outgoing = Scalar<DataVector>{get(lapse) + beta_n};
    out->largest_ingoing = Scalar<DataVector>{-get(lapse) + beta_n};
    // State: U_int = base - jump/2, U_ext = base + jump/2. Normal fluxes:
    // the same physical flux on both sides, but the exterior packages its
    // own outward normal, so its normal flux carries the opposite sign.
    const auto assign = [&](const gsl::not_null<SidePackage*> side,
                            const double state_sign, const double flux_sign) {
      const auto slot = [&](const size_t k) {
        return base_state.get(k) + state_sign * 0.5 * jump.get(k);
      };
      side->tilde_d = Scalar<DataVector>{slot(0)};
      for (size_t i = 0; i < 3; ++i) {
        side->tilde_s.get(i) = slot(i + 1);
      }
      side->tilde_tau = Scalar<DataVector>{slot(4)};
      side->tilde_ye = Scalar<DataVector>{slot(5)};
      side->nf_tilde_d = Scalar<DataVector>{flux_sign * base_flux.get(0)};
      for (size_t i = 0; i < 3; ++i) {
        side->nf_tilde_s.get(i) = flux_sign * base_flux.get(i + 1);
      }
      side->nf_tilde_tau = Scalar<DataVector>{flux_sign * base_flux.get(4)};
      side->nf_tilde_ye = Scalar<DataVector>{flux_sign * base_flux.get(5)};
    };
    assign(in, -1.0, 1.0);
    assign(out, 1.0, -1.0);
  };

  const grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe hllem{
      1.0e-30, 1.0e-8, true};
  const grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe plain_hll{
      1.0e-30, 1.0e-8, false};

  const auto point_approx = [&projection_conditioning](const size_t pt) {
    return Approx::custom()
        .epsilon(1.0e-11 * projection_conditioning[pt])
        .margin(1.0e-11 * projection_conditioning[pt]);
  };

  // (1) Middle-block jump at a coordinate-stationary contact: the weak-form
  //     numerical flux is the physical flux, and the strong-form correction
  //     vanishes.
  {
    SidePackage in{};
    SidePackage out{};
    build_sides(du_mid, make_not_null(&in), make_not_null(&out));
    const auto weak = apply_boundary_terms(hllem, in, out,
                                           dg::Formulation::WeakInertial, eos);
    const auto strong = apply_boundary_terms(
        hllem, in, out, dg::Formulation::StrongInertial, eos);
    for (size_t pt = 0; pt < num_pts; ++pt) {
      for (size_t k = 0; k < 6; ++k) {
        CHECK(weak.slot(k, pt) == point_approx(pt)(base_flux.get(k)[pt]));
        CHECK(strong.slot(k, pt) == point_approx(pt)(0.0));
      }
    }
    // Sanity: with the anti-diffusion switched off the identity must FAIL,
    // i.e. the middle block is doing real work here.
    const auto weak_hll = apply_boundary_terms(
        plain_hll, in, out, dg::Formulation::WeakInertial, eos);
    double max_difference = 0.0;
    for (size_t pt = 0; pt < num_pts; ++pt) {
      for (size_t k = 0; k < 6; ++k) {
        max_difference = std::max(
            max_difference, std::abs(weak_hll.slot(k, pt) - weak.slot(k, pt)));
      }
    }
    CHECK(max_difference > 1.0e-6);
  }

  // (2) Acoustic jump (Delta U = R_+, so P_mid Delta U = 0): the scheme must
  //     reduce EXACTLY to plain HLL (design v2 sec 13.4) -- also on a curved
  //     background with alpha != 1 and beta^n != 0.
  {
    SidePackage in{};
    SidePackage out{};
    build_sides(du_acoustic, make_not_null(&in), make_not_null(&out));
    const auto with_middle_block = apply_boundary_terms(
        hllem, in, out, dg::Formulation::WeakInertial, eos);
    const auto without = apply_boundary_terms(
        plain_hll, in, out, dg::Formulation::WeakInertial, eos);
    for (size_t pt = 0; pt < num_pts; ++pt) {
      for (size_t k = 0; k < 6; ++k) {
        CHECK(with_middle_block.slot(k, pt) ==
              point_approx(pt)(without.slot(k, pt)));
      }
    }
  }
}

// Lapse covariance. With beta^n = 0 and gamma_ij = delta_ij, rescaling the
// lapse by a constant c rescales every coordinate-frame normal flux and every
// characteristic speed by c, and the Valencia system is otherwise unchanged.
// The boundary correction must therefore come out exactly c times larger.
//
// This exercises the alpha half of lambda = alpha nu - beta^n through the
// averaged-state outer bounds, lambda_mid, delta_mid and the anti-diffusion
// coefficient at once. Note delta_mid is invariant under the rescaling, so a
// missing (or spurious) factor of alpha anywhere in that chain breaks it.
void test_lapse_scaling() {
  MAKE_GENERATOR(gen);
  namespace hydro_helper = TestHelpers::hydro;
  const auto nn_gen = make_not_null(&gen);
  const DataVector used_for_size(5);
  const size_t num_pts = used_for_size.size();

  tnsr::ii<DataVector, 3, Frame::Inertial> flat_metric{num_pts, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    flat_metric.get(i, i) = 1.0;
  }
  const auto eos_2d =
      EquationsOfState::IdealFluid<true>{5.0 / 3.0}.promote_to_3d_eos();
  const auto& eos = *eos_2d;

  std::uniform_real_distribution<double> dist(-0.4, 0.4);
  std::uniform_real_distribution<double> positive_dist(0.5, 1.5);

  const auto make_side = [&](const double lapse_value) {
    SidePackage side{};
    side.spatial_metric = flat_metric;
    side.lapse = Scalar<DataVector>{DataVector{num_pts, lapse_value}};
    side.shift_dot_normal = Scalar<DataVector>{DataVector{num_pts, 0.0}};
    return side;
  };

  const auto rho = hydro_helper::random_density(nn_gen, used_for_size);
  const auto eps =
      hydro_helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto ye = hydro_helper::random_electron_fraction(nn_gen, used_for_size);
  const auto lorentz =
      hydro_helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto v = hydro_helper::random_velocity(nn_gen, lorentz, flat_metric);
  const auto pressure = eos.pressure_from_density_and_energy(rho, eps, ye);
  const Scalar<DataVector> cs2{clamp(
      get(eos.sound_speed_squared_from_density_and_temperature(
          rho, eos.temperature_from_density_and_energy(rho, eps, ye), ye)),
      1.0e-4, 0.9)};

  tnsr::i<DataVector, 3, Frame::Inertial> normal{num_pts, 0.0};
  get<0>(normal) = 1.0;
  const auto states = make_with_random_values<tnsr::i<DataVector, 6>>(
      nn_gen, make_not_null(&positive_dist), used_for_size);
  const auto states_ext = make_with_random_values<tnsr::i<DataVector, 6>>(
      nn_gen, make_not_null(&positive_dist), used_for_size);
  const auto fluxes = make_with_random_values<tnsr::i<DataVector, 6>>(
      nn_gen, make_not_null(&dist), used_for_size);
  const auto fluxes_ext = make_with_random_values<tnsr::i<DataVector, 6>>(
      nn_gen, make_not_null(&dist), used_for_size);

  const auto build = [&](const double scale, const bool exterior) {
    SidePackage side = make_side(scale);
    side.normal = normal;
    if (exterior) {
      for (size_t i = 0; i < 3; ++i) {
        side.normal.get(i) *= -1.0;
      }
    }
    side.rest_mass_density = rho;
    side.electron_fraction = ye;
    side.sound_speed_squared = cs2;
    side.temperature = rho;
    side.spatial_velocity = v;
    side.pressure = pressure;
    side.lorentz_factor = lorentz;
    side.specific_internal_energy = eps;
    side.tilde_b = tnsr::I<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
    side.nf_tilde_b = tnsr::I<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
    side.tilde_phi = Scalar<DataVector>{DataVector{num_pts, 0.0}};
    side.nf_tilde_phi = Scalar<DataVector>{DataVector{num_pts, 0.0}};
    side.tilde_s = tnsr::i<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
    side.nf_tilde_s = tnsr::i<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
    const auto& u = exterior ? states_ext : states;
    const auto& f = exterior ? fluxes_ext : fluxes;
    side.tilde_d = Scalar<DataVector>{u.get(0)};
    side.tilde_tau = Scalar<DataVector>{u.get(4)};
    side.tilde_ye = Scalar<DataVector>{u.get(5)};
    side.nf_tilde_d = Scalar<DataVector>{scale * f.get(0)};
    side.nf_tilde_tau = Scalar<DataVector>{scale * f.get(4)};
    side.nf_tilde_ye = Scalar<DataVector>{scale * f.get(5)};
    for (size_t i = 0; i < 3; ++i) {
      side.tilde_s.get(i) = u.get(i + 1);
      side.nf_tilde_s.get(i) = scale * f.get(i + 1);
    }
    side.largest_outgoing = Scalar<DataVector>{DataVector{num_pts, scale}};
    side.largest_ingoing = Scalar<DataVector>{DataVector{num_pts, -scale}};
    return side;
  };

  const grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe hllem{
      1.0e-30, 1.0e-8, true};
  constexpr double scale = 0.6;
  const auto reference =
      apply_boundary_terms(hllem, build(1.0, false), build(1.0, true),
                           dg::Formulation::WeakInertial, eos);
  const auto scaled =
      apply_boundary_terms(hllem, build(scale, false), build(scale, true),
                           dg::Formulation::WeakInertial, eos);

  Approx custom_approx = Approx::custom().epsilon(1.0e-12).margin(1.0e-13);
  double max_magnitude = 0.0;
  for (size_t pt = 0; pt < num_pts; ++pt) {
    for (size_t k = 0; k < 6; ++k) {
      CHECK(scaled.slot(k, pt) == custom_approx(scale * reference.slot(k, pt)));
      max_magnitude = std::max(max_magnitude, std::abs(reference.slot(k, pt)));
    }
  }
  // Guard against the test passing trivially on an all-zero correction.
  CHECK(max_magnitude > 1.0e-3);
}

// ---------------------------------------------------------------------------
// Task A of notes/projects/active/hllem_hydroye/mbot_handoff_2026-09-16.md:
// check the analytic HllemHydroYe - HllcGr flux difference against the code.
//
// The symbolic derivation (flux_difference_reference.md) predicts
//
//   F_HLLEM - F_HLLC = B[dU_ac, dU_mid] + A[dU_ac, dU_ac] + O(||dU||^3) ,
//
// i.e. the two solvers agree to FIRST order in the jump -- both linearise to
// the exact upwind (Roe) flux -- and every contact and shear direction is
// exactly null. This test pins that against the running C++ at one fully
// specified state, with numbers predicted before the test existed.
//
// Setup (handoff "Setup, exactly"): IdealFluid Gamma = 5/3 so zeta == 0,
// flat spatial metric, lapse 1, zero shift, B = 0, n_hat = x_hat, and
//
//   V_*  = (rho, v_x, v_y, v_z, eps, Y_e) = (1.0, 0.2, 0.15, -0.1, 0.5, 0.3)
//   dV                                    = (0.5, -0.2, 0.3,  0.1, 0.4, -0.2)
//   V_L = V_* - (eta/2) dV ,   V_R = V_* + (eta/2) dV .
//
// Both solvers are driven through their OWN dg_package_data, so the packaging
// and sign conventions under test are the code's and not the test's. The
// interior carries V_L with n = +x_hat, the exterior V_R with n = -x_hat; in
// the weak form the returned "correction" is the numerical flux itself.
//
// An independent numpy implementation of the same analytic spec lives at
// notes/projects/active/hllem_hydroye/scripts/task_a_reference.py and
// reproduces every constant below.
//
// Measured on mbot 2026-09-17, clang Release, and agreeing with BOTH the
// Mathematica prediction and the numpy reference on every entry:
//
//   bounds at eta = 1e-2:  S_L = -0.38737594   S_R = +0.67106011
//
//        eta   ||F_HLLEM-F_HLLC||      /eta^2   ||F_HLLEM-F_HLL||
//       1e-1         1.066453e-3   1.066453e-1        1.344176e-2
//       1e-2         1.056416e-5   1.056416e-1        1.312538e-3
//       1e-3         1.055392e-7   1.055392e-1        1.309359e-4
//       1e-4         1.055290e-9   1.055290e-1        1.309041e-5
//
//   null directions at eta = 1e-3, against 1.06e-7 for a generic one:
//       rho contact at p-equilibrium   6.40e-17
//       Y_e contact at p-equilibrium   5.59e-17
//       shear v_y                      4.406251e-11
//       shear v_z                      2.918855e-11
//
//   The two shear values are a deterministic cancellation floor, not a
//   signal: the numpy reference reproduces them to seven significant
//   figures, which a real O(eta^2) term could not do.

// One interface point per value of eta, so a single pair of calls yields the
// whole convergence table.
constexpr std::array<double, 4> task_a_etas{{1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4}};

struct FlatPointStates {
  Scalar<DataVector> rest_mass_density{};
  Scalar<DataVector> electron_fraction{};
  Scalar<DataVector> specific_internal_energy{};
  Scalar<DataVector> pressure{};
  Scalar<DataVector> temperature{};
  Scalar<DataVector> lorentz_factor{};
  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{};
  tnsr::i<DataVector, 3, Frame::Inertial> spatial_velocity_one_form{};
  // Conserved variables and their full spatial fluxes.
  Scalar<DataVector> tilde_d{};
  Scalar<DataVector> tilde_ye{};
  Scalar<DataVector> tilde_tau{};
  tnsr::i<DataVector, 3, Frame::Inertial> tilde_s{};
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_b{};
  Scalar<DataVector> tilde_phi{};
  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_d{};
  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_ye{};
  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_tau{};
  tnsr::Ij<DataVector, 3, Frame::Inertial> flux_tilde_s{};
  tnsr::IJ<DataVector, 3, Frame::Inertial> flux_tilde_b{};
  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_phi{};
};

// Build one side's primitives, conserved variables and fluxes at a flat
// background from the six primitives at each point.
FlatPointStates make_flat_side(
    const std::vector<std::array<double, 6>>& primitives,
    const EquationsOfState::EquationOfState<true, 3>& eos,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric) {
  const size_t num_pts = primitives.size();
  FlatPointStates s{};
  s.rest_mass_density = Scalar<DataVector>{DataVector{num_pts, 0.0}};
  s.electron_fraction = Scalar<DataVector>{DataVector{num_pts, 0.0}};
  s.specific_internal_energy = Scalar<DataVector>{DataVector{num_pts, 0.0}};
  s.lorentz_factor = Scalar<DataVector>{DataVector{num_pts, 0.0}};
  s.spatial_velocity = tnsr::I<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  s.spatial_velocity_one_form =
      tnsr::i<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  s.tilde_b = tnsr::I<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  s.tilde_phi = Scalar<DataVector>{DataVector{num_pts, 0.0}};

  for (size_t pt = 0; pt < num_pts; ++pt) {
    const auto& v = primitives[pt];
    get(s.rest_mass_density)[pt] = v[0];
    get(s.specific_internal_energy)[pt] = v[4];
    get(s.electron_fraction)[pt] = v[5];
    double v_squared = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      s.spatial_velocity.get(i)[pt] = gsl::at(v, i + 1);
      // Flat metric, so the one-form components equal the vector ones.
      s.spatial_velocity_one_form.get(i)[pt] = gsl::at(v, i + 1);
      v_squared += gsl::at(v, i + 1) * gsl::at(v, i + 1);
    }
    get(s.lorentz_factor)[pt] = 1.0 / std::sqrt(1.0 - v_squared);
  }

  s.pressure = eos.pressure_from_density_and_energy(
      s.rest_mass_density, s.specific_internal_energy, s.electron_fraction);
  s.temperature = eos.temperature_from_density_and_energy(
      s.rest_mass_density, s.specific_internal_energy, s.electron_fraction);

  // No magnetic field and no divergence-cleaning field anywhere here.
  const tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{num_pts, 0.0};
  const Scalar<DataVector> divergence_cleaning_field{DataVector{num_pts, 0.0}};

  s.tilde_d = Scalar<DataVector>{DataVector{num_pts, 0.0}};
  s.tilde_ye = Scalar<DataVector>{DataVector{num_pts, 0.0}};
  s.tilde_tau = Scalar<DataVector>{DataVector{num_pts, 0.0}};
  s.tilde_s = tnsr::i<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&s.tilde_d), make_not_null(&s.tilde_ye),
      make_not_null(&s.tilde_tau), make_not_null(&s.tilde_s),
      make_not_null(&s.tilde_b), make_not_null(&s.tilde_phi),
      s.rest_mass_density, s.electron_fraction, s.specific_internal_energy,
      s.pressure, s.spatial_velocity, s.lorentz_factor, magnetic_field,
      sqrt_det_spatial_metric, spatial_metric, divergence_cleaning_field);

  s.flux_tilde_d = tnsr::I<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  s.flux_tilde_ye = tnsr::I<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  s.flux_tilde_tau = tnsr::I<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  s.flux_tilde_s = tnsr::Ij<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  s.flux_tilde_b = tnsr::IJ<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  s.flux_tilde_phi = tnsr::I<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  grmhd::ValenciaDivClean::ComputeFluxes::apply(
      make_not_null(&s.flux_tilde_d), make_not_null(&s.flux_tilde_ye),
      make_not_null(&s.flux_tilde_tau), make_not_null(&s.flux_tilde_s),
      make_not_null(&s.flux_tilde_b), make_not_null(&s.flux_tilde_phi),
      s.tilde_d, s.tilde_ye, s.tilde_tau, s.tilde_s, s.tilde_b, s.tilde_phi,
      lapse, shift, sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
      s.pressure, s.spatial_velocity, s.lorentz_factor, magnetic_field);
  return s;
}

// Packaged data for HllcGr -- the tag list differs from HllemHydroYe's, so it
// needs its own bundle.
struct HllcSidePackage {
  Scalar<DataVector> tilde_d{};
  Scalar<DataVector> tilde_ye{};
  Scalar<DataVector> tilde_tau{};
  tnsr::i<DataVector, 3, Frame::Inertial> tilde_s{};
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_b{};
  Scalar<DataVector> tilde_phi{};
  Scalar<DataVector> nf_tilde_d{};
  Scalar<DataVector> nf_tilde_ye{};
  Scalar<DataVector> nf_tilde_tau{};
  tnsr::i<DataVector, 3, Frame::Inertial> nf_tilde_s{};
  tnsr::I<DataVector, 3, Frame::Inertial> nf_tilde_b{};
  Scalar<DataVector> nf_tilde_phi{};
  Scalar<DataVector> largest_outgoing{};
  Scalar<DataVector> largest_ingoing{};
  Scalar<DataVector> normal_dot_tilde_s{};
  Scalar<DataVector> nf_normal_dot_tilde_s{};
  Scalar<DataVector> advection_speed{};
  Scalar<DataVector> pressure_flux_coefficient{};
  tnsr::i<DataVector, 3, Frame::Inertial> normal{};
  Scalar<DataVector> lapse{};
  Scalar<DataVector> shift_dot_normal{};
};

void resize_hllem_package(const gsl::not_null<SidePackage*> p,
                          const size_t num_pts) {
  const auto scalar = [num_pts]() {
    return Scalar<DataVector>{DataVector{num_pts, 0.0}};
  };
  p->tilde_d = scalar();
  p->tilde_ye = scalar();
  p->tilde_tau = scalar();
  p->tilde_s = tnsr::i<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  p->tilde_b = tnsr::I<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  p->tilde_phi = scalar();
  p->nf_tilde_d = scalar();
  p->nf_tilde_ye = scalar();
  p->nf_tilde_tau = scalar();
  p->nf_tilde_s = tnsr::i<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  p->nf_tilde_b = tnsr::I<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  p->nf_tilde_phi = scalar();
  p->largest_outgoing = scalar();
  p->largest_ingoing = scalar();
  p->normal = tnsr::i<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  p->lapse = scalar();
  p->shift_dot_normal = scalar();
  p->spatial_metric = tnsr::ii<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  p->rest_mass_density = scalar();
  p->electron_fraction = scalar();
  p->sound_speed_squared = scalar();
  p->temperature = scalar();
  p->spatial_velocity = tnsr::I<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  p->pressure = scalar();
  p->lorentz_factor = scalar();
  p->specific_internal_energy = scalar();
}

void resize_hllc_package(const gsl::not_null<HllcSidePackage*> p,
                         const size_t num_pts) {
  const auto scalar = [num_pts]() {
    return Scalar<DataVector>{DataVector{num_pts, 0.0}};
  };
  p->tilde_d = scalar();
  p->tilde_ye = scalar();
  p->tilde_tau = scalar();
  p->tilde_s = tnsr::i<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  p->tilde_b = tnsr::I<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  p->tilde_phi = scalar();
  p->nf_tilde_d = scalar();
  p->nf_tilde_ye = scalar();
  p->nf_tilde_tau = scalar();
  p->nf_tilde_s = tnsr::i<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  p->nf_tilde_b = tnsr::I<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  p->nf_tilde_phi = scalar();
  p->largest_outgoing = scalar();
  p->largest_ingoing = scalar();
  p->normal_dot_tilde_s = scalar();
  p->nf_normal_dot_tilde_s = scalar();
  p->advection_speed = scalar();
  p->pressure_flux_coefficient = scalar();
  p->normal = tnsr::i<DataVector, 3, Frame::Inertial>{num_pts, 0.0};
  p->lapse = scalar();
  p->shift_dot_normal = scalar();
}

// Result of one side of the interface, packaged for both solvers.
struct BothPackages {
  SidePackage hllem{};
  HllcSidePackage hllc{};
};

BothPackages package_flat_side(
    const grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe& hllem_bc,
    const grmhd::ValenciaDivClean::BoundaryCorrections::HllcGr& hllc_bc,
    const FlatPointStates& s,
    const EquationsOfState::EquationOfState<true, 3>& eos,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector) {
  const size_t num_pts = get(s.tilde_d).size();
  BothPackages out{};
  resize_hllem_package(make_not_null(&out.hllem), num_pts);
  resize_hllc_package(make_not_null(&out.hllc), num_pts);

  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>
      no_mesh_velocity{};
  const std::optional<Scalar<DataVector>> no_normal_dot_mesh_velocity{};

  hllem_bc.dg_package_data(
      make_not_null(&out.hllem.tilde_d), make_not_null(&out.hllem.tilde_ye),
      make_not_null(&out.hllem.tilde_tau), make_not_null(&out.hllem.tilde_s),
      make_not_null(&out.hllem.tilde_b), make_not_null(&out.hllem.tilde_phi),
      make_not_null(&out.hllem.nf_tilde_d),
      make_not_null(&out.hllem.nf_tilde_ye),
      make_not_null(&out.hllem.nf_tilde_tau),
      make_not_null(&out.hllem.nf_tilde_s),
      make_not_null(&out.hllem.nf_tilde_b),
      make_not_null(&out.hllem.nf_tilde_phi),
      make_not_null(&out.hllem.largest_outgoing),
      make_not_null(&out.hllem.largest_ingoing),
      make_not_null(&out.hllem.normal), make_not_null(&out.hllem.lapse),
      make_not_null(&out.hllem.shift_dot_normal),
      make_not_null(&out.hllem.spatial_metric),
      make_not_null(&out.hllem.rest_mass_density),
      make_not_null(&out.hllem.electron_fraction),
      make_not_null(&out.hllem.sound_speed_squared),
      make_not_null(&out.hllem.temperature),
      make_not_null(&out.hllem.spatial_velocity),
      make_not_null(&out.hllem.pressure),
      make_not_null(&out.hllem.lorentz_factor),
      make_not_null(&out.hllem.specific_internal_energy), s.tilde_d, s.tilde_ye,
      s.tilde_tau, s.tilde_s, s.tilde_b, s.tilde_phi, s.flux_tilde_d,
      s.flux_tilde_ye, s.flux_tilde_tau, s.flux_tilde_s, s.flux_tilde_b,
      s.flux_tilde_phi, lapse, shift, s.spatial_velocity_one_form,
      spatial_metric, s.rest_mass_density, s.electron_fraction, s.temperature,
      s.spatial_velocity, s.specific_internal_energy, s.pressure,
      s.lorentz_factor, normal_covector, normal_vector, no_mesh_velocity,
      no_normal_dot_mesh_velocity, eos);

  hllc_bc.dg_package_data(
      make_not_null(&out.hllc.tilde_d), make_not_null(&out.hllc.tilde_ye),
      make_not_null(&out.hllc.tilde_tau), make_not_null(&out.hllc.tilde_s),
      make_not_null(&out.hllc.tilde_b), make_not_null(&out.hllc.tilde_phi),
      make_not_null(&out.hllc.nf_tilde_d), make_not_null(&out.hllc.nf_tilde_ye),
      make_not_null(&out.hllc.nf_tilde_tau),
      make_not_null(&out.hllc.nf_tilde_s), make_not_null(&out.hllc.nf_tilde_b),
      make_not_null(&out.hllc.nf_tilde_phi),
      make_not_null(&out.hllc.largest_outgoing),
      make_not_null(&out.hllc.largest_ingoing),
      make_not_null(&out.hllc.normal_dot_tilde_s),
      make_not_null(&out.hllc.nf_normal_dot_tilde_s),
      make_not_null(&out.hllc.advection_speed),
      make_not_null(&out.hllc.pressure_flux_coefficient),
      make_not_null(&out.hllc.normal), make_not_null(&out.hllc.lapse),
      make_not_null(&out.hllc.shift_dot_normal), s.tilde_d, s.tilde_ye,
      s.tilde_tau, s.tilde_s, s.tilde_b, s.tilde_phi, s.flux_tilde_d,
      s.flux_tilde_ye, s.flux_tilde_tau, s.flux_tilde_s, s.flux_tilde_b,
      s.flux_tilde_phi, lapse, shift, s.spatial_velocity_one_form,
      spatial_metric, s.rest_mass_density, s.electron_fraction, s.temperature,
      s.spatial_velocity, s.specific_internal_energy, s.pressure,
      s.lorentz_factor, normal_covector, normal_vector, no_mesh_velocity,
      no_normal_dot_mesh_velocity, eos);
  return out;
}

Correction apply_hllc_boundary_terms(const HllcSidePackage& in,
                                     const HllcSidePackage& out,
                                     const dg::Formulation formulation) {
  Correction result{get(in.tilde_d).size()};
  grmhd::ValenciaDivClean::BoundaryCorrections::HllcGr::dg_boundary_terms(
      make_not_null(&result.tilde_d), make_not_null(&result.tilde_ye),
      make_not_null(&result.tilde_tau), make_not_null(&result.tilde_s),
      make_not_null(&result.tilde_b), make_not_null(&result.tilde_phi),
      in.tilde_d, in.tilde_ye, in.tilde_tau, in.tilde_s, in.tilde_b,
      in.tilde_phi, in.nf_tilde_d, in.nf_tilde_ye, in.nf_tilde_tau,
      in.nf_tilde_s, in.nf_tilde_b, in.nf_tilde_phi, in.largest_outgoing,
      in.largest_ingoing, in.normal_dot_tilde_s, in.nf_normal_dot_tilde_s,
      in.advection_speed, in.pressure_flux_coefficient, in.normal, in.lapse,
      in.shift_dot_normal, out.tilde_d, out.tilde_ye, out.tilde_tau,
      out.tilde_s, out.tilde_b, out.tilde_phi, out.nf_tilde_d, out.nf_tilde_ye,
      out.nf_tilde_tau, out.nf_tilde_s, out.nf_tilde_b, out.nf_tilde_phi,
      out.largest_outgoing, out.largest_ingoing, out.normal_dot_tilde_s,
      out.nf_normal_dot_tilde_s, out.advection_speed,
      out.pressure_flux_coefficient, out.normal, out.lapse,
      out.shift_dot_normal, formulation);
  return result;
}

void test_task_a_hllem_vs_hllc() {
  const std::array<double, 6> v_star{{1.0, 0.2, 0.15, -0.1, 0.5, 0.3}};
  const std::array<double, 6> dv{{0.5, -0.2, 0.3, 0.1, 0.4, -0.2}};

  // The jump directions of handoff prediction 3, which must be exactly null.
  // chi/kappa = eps/rho at this state, so the density contact at pressure
  // equilibrium is dV = (1, 0, 0, 0, -eps/rho, 0) = (1, 0, 0, 0, -0.5, 0).
  const double chi_over_kappa = v_star[4] / v_star[0];
  const std::array<std::pair<std::string, std::array<double, 6>>, 4>
      null_directions{{{"rho contact at pressure equilibrium",
                        {{1.0, 0.0, 0.0, 0.0, -chi_over_kappa, 0.0}}},
                       {"Y_e contact at pressure equilibrium",
                        {{0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}},
                       {"shear v_y", {{0.0, 0.0, 1.0, 0.0, 0.0, 0.0}}},
                       {"shear v_z", {{0.0, 0.0, 0.0, 1.0, 0.0, 0.0}}}}};

  const auto eos_2d =
      EquationsOfState::IdealFluid<true>{5.0 / 3.0}.promote_to_3d_eos();
  const auto& eos = *eos_2d;

  // RestoreMiddleBlock = true. zeta == 0 for IdealFluid regardless.
  const grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe hllem_bc{
      1.0e-30, 1.0e-8, true};
  const grmhd::ValenciaDivClean::BoundaryCorrections::HllcGr hllc_bc{1.0e-30,
                                                                     1.0e-8};

  // Evaluate the flux difference along `direction` at every eta at once.
  // Returns, per point, {||F_HLLEM - F_HLLC||, ||F_HLLEM - F_HLL||, S_L, S_R}
  // plus the six components of the difference and of each flux.
  struct Row {
    double diff_norm{};
    double hllem_minus_hll_norm{};
    double s_left{};
    double s_right{};
    std::array<double, 6> difference{};
    std::array<double, 6> hllem{};
    std::array<double, 6> hllc{};
  };

  const auto evaluate = [&](const std::array<double, 6>& direction,
                            const std::vector<double>& etas) {
    const size_t num_pts = etas.size();
    std::vector<std::array<double, 6>> left(num_pts);
    std::vector<std::array<double, 6>> right(num_pts);
    for (size_t pt = 0; pt < num_pts; ++pt) {
      for (size_t k = 0; k < 6; ++k) {
        gsl::at(left[pt], k) =
            gsl::at(v_star, k) - 0.5 * etas[pt] * gsl::at(direction, k);
        gsl::at(right[pt], k) =
            gsl::at(v_star, k) + 0.5 * etas[pt] * gsl::at(direction, k);
      }
    }

    // Flat background: lapse 1, zero shift, gamma_ij = delta_ij.
    const Scalar<DataVector> lapse{DataVector{num_pts, 1.0}};
    const tnsr::I<DataVector, 3, Frame::Inertial> shift{num_pts, 0.0};
    const Scalar<DataVector> sqrt_det_spatial_metric{DataVector{num_pts, 1.0}};
    tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{num_pts, 0.0};
    tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric{num_pts, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      spatial_metric.get(i, i) = 1.0;
      inv_spatial_metric.get(i, i) = 1.0;
    }

    // Interior carries V_L with n = +x_hat, exterior V_R with n = -x_hat.
    tnsr::i<DataVector, 3, Frame::Inertial> normal_int{num_pts, 0.0};
    tnsr::I<DataVector, 3, Frame::Inertial> normal_vector_int{num_pts, 0.0};
    tnsr::i<DataVector, 3, Frame::Inertial> normal_ext{num_pts, 0.0};
    tnsr::I<DataVector, 3, Frame::Inertial> normal_vector_ext{num_pts, 0.0};
    get<0>(normal_int) = 1.0;
    get<0>(normal_vector_int) = 1.0;
    get<0>(normal_ext) = -1.0;
    get<0>(normal_vector_ext) = -1.0;

    const auto state_left =
        make_flat_side(left, eos, lapse, shift, sqrt_det_spatial_metric,
                       spatial_metric, inv_spatial_metric);
    const auto state_right =
        make_flat_side(right, eos, lapse, shift, sqrt_det_spatial_metric,
                       spatial_metric, inv_spatial_metric);

    const auto in =
        package_flat_side(hllem_bc, hllc_bc, state_left, eos, lapse, shift,
                          spatial_metric, normal_int, normal_vector_int);
    const auto out =
        package_flat_side(hllem_bc, hllc_bc, state_right, eos, lapse, shift,
                          spatial_metric, normal_ext, normal_vector_ext);

    // Weak form: the returned correction IS the numerical flux.
    const auto f_hllem = apply_boundary_terms(
        hllem_bc, in.hllem, out.hllem, dg::Formulation::WeakInertial, eos);
    const auto f_hllc = apply_hllc_boundary_terms(
        in.hllc, out.hllc, dg::Formulation::WeakInertial);
    // RestoreMiddleBlock = false is bit-identical to Hll, which is the
    // cheapest in-class way to get the HLL flux at exactly these bounds.
    const grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe plain_hll{
        1.0e-30, 1.0e-8, false};
    const auto f_hll = apply_boundary_terms(plain_hll, in.hllem, out.hllem,
                                            dg::Formulation::WeakInertial, eos);

    std::vector<Row> rows(num_pts);
    for (size_t pt = 0; pt < num_pts; ++pt) {
      double diff_squared = 0.0;
      double hll_diff_squared = 0.0;
      for (size_t k = 0; k < 6; ++k) {
        const double d = f_hllem.slot(k, pt) - f_hllc.slot(k, pt);
        const double dh = f_hllem.slot(k, pt) - f_hll.slot(k, pt);
        gsl::at(rows[pt].difference, k) = d;
        gsl::at(rows[pt].hllem, k) = f_hllem.slot(k, pt);
        gsl::at(rows[pt].hllc, k) = f_hllc.slot(k, pt);
        diff_squared += d * d;
        hll_diff_squared += dh * dh;
      }
      rows[pt].diff_norm = std::sqrt(diff_squared);
      rows[pt].hllem_minus_hll_norm = std::sqrt(hll_diff_squared);
      // The bounds each solver actually used, reconstructed from the
      // packaged per-side speeds exactly as Hllc.cpp combines them.
      rows[pt].s_right = std::max({0.0, get(in.hllc.largest_outgoing)[pt],
                                   -get(out.hllc.largest_ingoing)[pt]});
      rows[pt].s_left = std::min({0.0, get(in.hllc.largest_ingoing)[pt],
                                  -get(out.hllc.largest_outgoing)[pt]});
    }
    return rows;
  };

  const std::vector<double> etas{task_a_etas[0], task_a_etas[1], task_a_etas[2],
                                 task_a_etas[3]};
  const auto rows = evaluate(dv, etas);

  INFO("Task A: HllemHydroYe vs HllcGr at the handoff state");

  // --- the bounds, checked first: if these are wrong the state is not being
  // --- built the way the derivation assumed and nothing below is meaningful.
  {
    INFO("predicted bounds at eta = 1e-2");
    const Approx bounds_approx = Approx::custom().epsilon(1.0e-7);
    CHECK(rows[1].s_left == bounds_approx(-0.38737594));
    CHECK(rows[1].s_right == bounds_approx(0.67106011));
  }

  // --- prediction 1: the difference is second order in the jump, while each
  // --- solver's difference from Hll is first order.
  {
    INFO("prediction 1: O(eta^2) convergence");
    const std::array<double, 4> predicted_diff{
        {1.0665e-3, 1.0564e-5, 1.0554e-7, 1.0553e-9}};
    const std::array<double, 4> predicted_hll{
        {1.3442e-2, 1.3125e-3, 1.3094e-4, 1.3090e-5}};
    // Five-digit predictions, so a relative tolerance of 1e-4 is the
    // precision of the prediction itself rather than a loose margin.
    const Approx table_approx = Approx::custom().epsilon(1.0e-4);
    for (size_t pt = 0; pt < 4; ++pt) {
      CAPTURE(gsl::at(task_a_etas, pt));
      CHECK(rows[pt].diff_norm == table_approx(gsl::at(predicted_diff, pt)));
      CHECK(rows[pt].hllem_minus_hll_norm ==
            table_approx(gsl::at(predicted_hll, pt)));
    }
    // The third column of the handoff table CONVERGING to a constant is the
    // result -- not the constant holding at every eta. The predicted column
    // is 1.0665e-1, 1.0564e-1, 1.0554e-1, 1.0553e-1, so the ratio is still
    // drifting at eta = 1e-1 and has settled by eta = 1e-3. Assert both the
    // listed values and the monotone approach to the limit.
    const std::array<double, 4> predicted_scaled{
        {1.0665e-1, 1.0564e-1, 1.0554e-1, 1.0553e-1}};
    std::array<double, 4> scaled{};
    for (size_t pt = 0; pt < 4; ++pt) {
      gsl::at(scaled, pt) = rows[pt].diff_norm / (gsl::at(task_a_etas, pt) *
                                                  gsl::at(task_a_etas, pt));
      CAPTURE(gsl::at(scaled, pt));
      CHECK(gsl::at(scaled, pt) == table_approx(gsl::at(predicted_scaled, pt)));
    }
    // Monotone convergence, and settled to better than 1e-3 relative between
    // the last two rows: that is what "the difference is O(eta^2)" means.
    for (size_t pt = 1; pt < 4; ++pt) {
      CHECK(std::abs(gsl::at(scaled, pt) - gsl::at(scaled, 3)) <=
            std::abs(gsl::at(scaled, pt - 1) - gsl::at(scaled, 3)));
    }
    CHECK(std::abs(scaled[3] - scaled[2]) / scaled[3] < 1.0e-3);
  }

  // --- prediction 2: the six components at eta = 1e-2.
  {
    INFO("prediction 2: component breakdown at eta = 1e-2");
    const std::array<double, 6> predicted_difference{{-2.85070e-6, -2.19849e-6,
                                                      -7.76438e-6, -7.17845e-7,
                                                      -6.03105e-6, 1.21128e-6}};
    const std::array<double, 6> predicted_hllem{{0.20612478, 0.41204147,
                                                 0.05815825, -0.03936560,
                                                 0.18598115, 0.06204563}};
    const std::array<double, 6> predicted_hllc{{0.20612763, 0.41204366,
                                                0.05816602, -0.03936488,
                                                0.18598718, 0.06204442}};
    const Approx component_approx = Approx::custom().epsilon(1.0e-4);
    const Approx flux_approx = Approx::custom().epsilon(1.0e-7);
    for (size_t k = 0; k < 6; ++k) {
      CAPTURE(k);
      CHECK(gsl::at(rows[1].difference, k) ==
            component_approx(gsl::at(predicted_difference, k)));
      CHECK(gsl::at(rows[1].hllem, k) ==
            flux_approx(gsl::at(predicted_hllem, k)));
      CHECK(gsl::at(rows[1].hllc, k) ==
            flux_approx(gsl::at(predicted_hllc, k)));
    }
  }

  // --- prediction 3: four null directions. These are the sharpest part of
  // --- the prediction: they are structural claims, not fitted numbers.
  {
    INFO("prediction 3: contact and shear directions are null");
    const std::vector<double> null_etas{1.0e-3};
    for (const auto& [name, direction] : null_directions) {
      INFO(name);
      const auto null_rows = evaluate(direction, null_etas);
      CAPTURE(null_rows[0].diff_norm);
      // The contacts are exactly zero and the shears sit on a round-off
      // floor a few times 1e-11 (reproduced independently in numpy). The
      // signal this is distinguishing itself from is the 1.06e-7 that a
      // generic direction gives at this eta -- four orders of magnitude up.
      CHECK(null_rows[0].diff_norm < 1.0e-9);
    }
  }
}

// ---------------------------------------------------------------------------
// U_L == U_R: the weak-form numerical flux must BE the physical flux.
//
// None of the four tests above covers a zero jump, and the invariant is
// stronger than it looks. From the `hll` lambda in HllemHydroYe.cpp, the
// weak form is
//
//   F = [l_max nf_int + l_min nf_ext + l_max l_min (u_ext - u_int)] / dl ,
//   dl = l_max - l_min  (floored at 1e-30) .
//
// With the two sides built from the SAME state, `u_ext - u_int` is exactly
// 0.0 and `nf_ext = -nf_int` exactly (the exterior normal is the negated
// interior one), so
//
//   F = (l_max - l_min) nf_int / dl = nf_int
//
// for ANY bounds with l_max > 0 > l_min. The middle-block anti-diffusion
// adds coeff * P_mid * du with du = 0, so it contributes nothing by algebra.
// The identity is therefore insensitive to the bound RECIPE and sensitive
// only to the bounds DEGENERATING: if `dl` ever collapses into the 1e-30
// floor the weak flux returns (l_max - l_min) nf / 1e-30 instead of nf.
//
// This is why the assertion is "weak flux == nf_int" and NOT
// "correction == 0". The strong form is
// [l_min (nf_int + nf_ext) + l_max l_min (u_ext - u_int)] / dl; with
// nf_ext = -nf_int and du = 0 BOTH numerator terms are identically zero, so
// the strong-form correction is 0 whatever `dl` does -- including a `dl`
// clamped to 1e-30 with the bounds collapsed to ~1e-161. A
// correction-is-zero test passes while the flux has underflowed.
//
// HONEST SCOPE. This is a useful invariant, NOT a regression test for the
// 2026-09-15 underflow. That defect was a composite Blaze expression passed
// into `sqrt(clamp(...))` while building the averaged-state acoustic
// discriminant; it was fixed in 0d37021e3 and the whole averaged-state
// bounds block it lived in was then deleted by faaaba1de ("HLL bounds from
// the input states, not the average"), so the code that carried it no
// longer exists and cannot be re-broken here.
//
// What it DOES guard, demonstrated by injecting each failure and watching
// this test go red: the `dl < 1e-30` floor in the `hll` lambda firing on a
// healthy interface (scaling both bounds by 1e-160 fails all 3780 weak-form
// checks and zero of the correction-is-zero ones), and a sign regression in
// the packaged normal flux.
//
// What it does NOT guard, also measured rather than assumed: anything that
// perturbs the WAVE SPEEDS while leaving them finite and ordered. Injecting
// a 63% error into `sound_speed_squared` right after its clamp leaves this
// test completely green -- 0 of its 16395 assertions fail -- because with
// du = 0 the weak form returns nf_int for ANY bounds with l_max > 0 > l_min.
// So this does not cover the class's `clamp` sites or the averaged-state
// block, and must not be cited as if it did. The guard that demonstrably
// catches the original defect remains the 30-step uniform driver in
// spectre_runs/shocktube/togashi_uniform_onset/.
//
// SIMD SIZING. This build compiles with -march=native -mno-avx512f, so
// blaze::SIMDTrait<double>::size == 4 (measured against the build's own
// blaze 3.8 / xsimd 12.1.1 headers), and BLAZE_USE_PADDING=0 leaves a
// genuine scalar tail. `num_points` is therefore swept over {41, 42, 43}:
// ODD sizes have a nonzero remainder at every power-of-two width (2, 4, 8,
// 16), so the test keeps its tail on an AVX-512 or SSE build too, and the
// three values sweep every nonzero remainder at width 4. 41 = 32 + 8 + 1
// also exercises all three of Blaze's tiers (4-way-unrolled block, single
// SIMD step, scalar remainder) even at width 8. The existing tests use 3-5
// points, which at width 4 is epilogue only.

// Mirror of Correction::slot for a PACKAGED normal flux: same
// [D, S_x, S_y, S_z, tau, D Y_e] ordering.
double nf_slot(const SidePackage& p, const size_t n, const size_t pt) {
  switch (n) {
    case 0:
      return get(p.nf_tilde_d)[pt];
    case 1:
    case 2:
    case 3:
      return p.nf_tilde_s.get(n - 1)[pt];
    case 4:
      return get(p.nf_tilde_tau)[pt];
    default:
      return get(p.nf_tilde_ye)[pt];
  }
}

void test_zero_jump_flux_identity() {
  const auto eos_2d =
      EquationsOfState::IdealFluid<true>{5.0 / 3.0}.promote_to_3d_eos();
  const auto& eos = *eos_2d;
  const grmhd::ValenciaDivClean::BoundaryCorrections::HllcGr hllc_bc{1.0e-30,
                                                                     1.0e-8};

  struct Arm {
    std::string name;
    bool restore_middle_block{};
    // 1.0 keeps rho above LightSpeedDensityCutoff (hydro branch, packaged
    // cs^2 > 0); 1.0e-10 puts it below, so cs^2 stays at its 0 sentinel and
    // dg_boundary_terms falls back to the light-speed bounds.
    double rho_scale{};
    bool hydro_branch_expected{};
    // A deterministic point-varying state is the more discriminating one for
    // a stale/misaligned SIMD read, because a value borrowed from a
    // neighbouring point no longer coincides with the right answer. The
    // exactly uniform arm is kept because that is the configuration the
    // 2026-09-15 defect was observed in.
    bool point_varying{};
    double min_flux_magnitude{};
  };
  const std::array<Arm, 5> arms{
      {{"hydro, RestoreMiddleBlock = true, varying", true, 1.0, true, true,
        1.0e-3},
       {"hydro, RestoreMiddleBlock = true, uniform", true, 1.0, true, false,
        1.0e-3},
       {"hydro, RestoreMiddleBlock = false (Hll-equivalent), varying", false,
        1.0, true, true, 1.0e-3},
       {"below atmosphere, light-speed bounds, varying", true, 1.0e-10, false,
        true, 1.0e-12},
       {"below atmosphere, light-speed bounds, uniform", true, 1.0e-10, false,
        false, 1.0e-12}}};

  const std::array<size_t, 3> point_counts{{41, 42, 43}};

  for (const size_t num_pts : point_counts) {
    for (const Arm& arm : arms) {
      CAPTURE(num_pts);
      CAPTURE(arm.name);
      const grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe hllem_bc{
          1.0e-30, 1.0e-8, arm.restore_middle_block};

      std::vector<std::array<double, 6>> prims(num_pts);
      for (size_t pt = 0; pt < num_pts; ++pt) {
        const double i = arm.point_varying ? static_cast<double>(pt) : 0.0;
        prims[pt] = {{arm.rho_scale * (1.0 + 0.01 * i), 0.2 + 0.001 * i, 0.15,
                      -0.1, 0.5 + 0.002 * i, 0.30 + 0.001 * i}};
      }

      // Flat background: lapse 1, zero shift, gamma_ij = delta_ij.
      const Scalar<DataVector> lapse{DataVector{num_pts, 1.0}};
      const tnsr::I<DataVector, 3, Frame::Inertial> shift{num_pts, 0.0};
      const Scalar<DataVector> sqrt_det_spatial_metric{
          DataVector{num_pts, 1.0}};
      tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{num_pts, 0.0};
      tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric{num_pts, 0.0};
      for (size_t i = 0; i < 3; ++i) {
        spatial_metric.get(i, i) = 1.0;
        inv_spatial_metric.get(i, i) = 1.0;
      }

      // Interior face carries n = +x_hat, exterior n = -x_hat.
      tnsr::i<DataVector, 3, Frame::Inertial> normal_int{num_pts, 0.0};
      tnsr::I<DataVector, 3, Frame::Inertial> normal_vector_int{num_pts, 0.0};
      tnsr::i<DataVector, 3, Frame::Inertial> normal_ext{num_pts, 0.0};
      tnsr::I<DataVector, 3, Frame::Inertial> normal_vector_ext{num_pts, 0.0};
      get<0>(normal_int) = 1.0;
      get<0>(normal_vector_int) = 1.0;
      get<0>(normal_ext) = -1.0;
      get<0>(normal_vector_ext) = -1.0;

      // ONE state for both sides -- that is what makes U_L == U_R bitwise.
      const auto s =
          make_flat_side(prims, eos, lapse, shift, sqrt_det_spatial_metric,
                         spatial_metric, inv_spatial_metric);
      const auto in =
          package_flat_side(hllem_bc, hllc_bc, s, eos, lapse, shift,
                            spatial_metric, normal_int, normal_vector_int);
      const auto out =
          package_flat_side(hllem_bc, hllc_bc, s, eos, lapse, shift,
                            spatial_metric, normal_ext, normal_vector_ext);

      // Which branch of dg_package_data / dg_boundary_terms we actually
      // landed in. Without this the arm could silently stop testing the
      // path it is named for.
      for (size_t pt = 0; pt < num_pts; ++pt) {
        if (arm.hydro_branch_expected) {
          CHECK(get(in.hllem.sound_speed_squared)[pt] > 0.0);
          CHECK(get(out.hllem.sound_speed_squared)[pt] > 0.0);
        } else {
          CHECK(get(in.hllem.sound_speed_squared)[pt] == 0.0);
          CHECK(get(out.hllem.sound_speed_squared)[pt] == 0.0);
        }
        // The packaged primitives are the INPUT primitives, not an EOS
        // re-derivation. This pins the premise of the dead-chain deletion in
        // dg_package_data: the locals that used to be evaluated there
        // shadowed these parameters and were never read. No pypp arm reaches
        // the hydro branch, so this is the only place it is asserted.
        CHECK(get(in.hllem.pressure)[pt] == get(s.pressure)[pt]);
        CHECK(get(in.hllem.specific_internal_energy)[pt] ==
              get(s.specific_internal_energy)[pt]);
      }

      const auto weak = apply_boundary_terms(
          hllem_bc, in.hllem, out.hllem, dg::Formulation::WeakInertial, eos);
      const auto strong = apply_boundary_terms(
          hllem_bc, in.hllem, out.hllem, dg::Formulation::StrongInertial, eos);

      double max_flux_magnitude = 0.0;
      for (size_t pt = 0; pt < num_pts; ++pt) {
        CAPTURE(pt);
        double scale_pt = 0.0;
        for (size_t k = 0; k < 6; ++k) {
          scale_pt = std::max(scale_pt, std::abs(nf_slot(in.hllem, k, pt)));
        }
        max_flux_magnitude = std::max(max_flux_magnitude, scale_pt);
        // nf_int is the EXACT reference here (it is the physical flux of the
        // one state, computed by production ComputeFluxes), not a second
        // approximate solver, so a relative tolerance is legitimate. The
        // margin only covers a component that happens to be ~0. Either way
        // this is ~12 orders of magnitude above the ~1e-131 that a collapsed
        // `dl` would produce.
        const Approx point_approx =
            Approx::custom().epsilon(1.0e-13).margin(1.0e-14 * scale_pt);
        for (size_t k = 0; k < 6; ++k) {
          CAPTURE(k);
          const double nf_int = nf_slot(in.hllem, k, pt);
          CHECK(weak.slot(k, pt) == point_approx(nf_int));
          CHECK(strong.slot(k, pt) == point_approx(0.0));
          // Free invariant that catches a normal-sign regression in
          // dg_package_data, and is the other half of the identity above.
          CHECK(nf_slot(out.hllem, k, pt) == point_approx(-nf_int));
        }
        // B and phi are zero throughout, but they go through the same `hll`
        // lambda, so they get the same identity.
        CHECK(get(weak.tilde_phi)[pt] ==
              point_approx(get(in.hllem.nf_tilde_phi)[pt]));
        for (size_t i = 0; i < 3; ++i) {
          CHECK(weak.tilde_b.get(i)[pt] ==
                point_approx(in.hllem.nf_tilde_b.get(i)[pt]));
        }
      }
      // Guard against the whole arm passing on an all-zero (or underflowed)
      // flux -- without this, `nf == 0` makes every CHECK above vacuous.
      CHECK(max_flux_magnitude > arm.min_flux_magnitude);
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
  test_gr_stationary_contact();
  test_lapse_scaling();
  test_task_a_hllem_vs_hllc();
  test_zero_jump_flux_identity();

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
      grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{1.0e-30,
                                                                 1.0e-8, false},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "HllemHydroYe", "dg_package_data",
      "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{1.0e-30,
                                                                 1.0e-8, false},
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
      grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{1.0e-30,
                                                                 1.0e-8, false},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges_hydro);

  // The pypp cross-checks above never enter the hydro branch: every range
  // either has |B| > MagneticFieldMagnitudeForHydro or a below-atmosphere
  // density, so the packaged sound speed stays at its zero sentinel and the
  // C++ falls back to the light-speed bounds on every slot. The averaged-
  // state bounds, the interface geometry and the middle-block anti-diffusion
  // are covered instead by the direct `dg_boundary_terms` tests above
  // (`test_gr_stationary_contact` and `test_lapse_scaling`), which do not
  // need a Python reference for `characteristic_eigenvectors_hydro`.

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
      grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{1.0e-30,
                                                                 1.0e-8, false},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges_atmo);

  const auto hll = TestHelpers::test_factory_creation<
      evolution::BoundaryCorrection,
      grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe>(
      "HllemHydroYe:\n"
      "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
      "  LightSpeedDensityCutoff: 1.0e-8\n"
      "  RestoreMiddleBlock: False\n");

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
                  1.0e-30, 1.0e-8, false} !=
              grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{
                  1.0e-30, 1.0e-8, false});
  CHECK(grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{
            1.0e-30, 1.0e-8, false} !=
        grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{
            2.0e-30, 1.0e-8, false});
  CHECK(grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{
            1.0e-30, 1.0e-8, false} !=
        grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe{
            1.0e-30, 2.0e-8, false});
}
}  // namespace
