// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
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
      1.0e-30, 1.0e-8, true, true};
  const grmhd::ValenciaDivClean::BoundaryCorrections::HllemHydroYe plain_hll{
      1.0e-30, 1.0e-8, false, true};

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
      1.0e-30, 1.0e-8, true, true};
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
