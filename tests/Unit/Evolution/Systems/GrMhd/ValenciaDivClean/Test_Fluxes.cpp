// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace {
// Rung 2 of the HLLC-GR validation ladder (see
// `spectre_runs/kh_solver_comparison/HLLC_GR_ONF_PLAN.md` §7.1, §10 rung 2).
// Verifies the Valencia densitization identity
//   sqrt(gamma) * alpha^2 * T^{0i}  =  F^i(TildeTau) + F^i(TildeD)
// on random curved 3+1 data + random primitives, using the existing
// ComputeFluxes implementation. Restricted to B = 0 (the identity
// holds for the pure-hydro sector; magnetic sector adds separate terms).
//
// The identity is the SR-analog `F(E) = m^x` in curved coordinate-frame
// Valencia — MB05's whole derivation rests on it. Rung 1 (flat-space
// bitwise reduction to Hllc) tests it at alpha = 1, beta = 0 only; this
// rung is the first that exercises the mapping at generic 3+1 data
// independently of any Riemann solver.
void test_energy_flux_identity_curved_hydro(const DataVector& used_for_size) {
  MAKE_GENERATOR(gen);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&gen);
  const size_t num_pts = used_for_size.size();

  // 3+1 geometry.
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const Scalar<DataVector> sqrt_det_spatial_metric{
      sqrt(get(det_and_inv.first))};
  const auto& inv_spatial_metric = det_and_inv.second;
  // Randomize lapse in [0.5, 1.0] to keep it well away from zero
  // (matches TestHelpers::gr conventions).
  std::uniform_real_distribution<double> lapse_dist(0.5, 1.0);
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      nn_gen, make_not_null(&lapse_dist), used_for_size);
  // Shift components in [-0.3, 0.3] so beta^n has a real magnitude
  // without inducing superluminal drift (matches Test_Hllc's ranges).
  std::uniform_real_distribution<double> shift_dist(-0.3, 0.3);
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          nn_gen, make_not_null(&shift_dist), used_for_size);

  // Hydro primitives — chosen so v_i v^i < 1 via the helper.
  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const auto electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);
  const auto specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  // Random pressure in (0, 1].
  std::uniform_real_distribution<double> pressure_dist(1.0e-3, 1.0);
  const auto pressure = make_with_random_values<Scalar<DataVector>>(
      nn_gen, make_not_null(&pressure_dist), used_for_size);
  // B = 0 sector: magnetic field, divergence-cleaning field identically
  // zero. The pure-hydro identity holds; the magnetic sector adds
  // separate terms that are outside the scope of this rung.
  const tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{num_pts, 0.0};
  const Scalar<DataVector> divergence_cleaning_field{num_pts, 0.0};

  // Compute the conserved variables consistently from the primitives.
  Scalar<DataVector> tilde_d{num_pts};
  Scalar<DataVector> tilde_ye{num_pts};
  Scalar<DataVector> tilde_tau{num_pts};
  tnsr::i<DataVector, 3, Frame::Inertial> tilde_s{num_pts};
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_b{num_pts};
  Scalar<DataVector> tilde_phi{num_pts};
  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&tilde_d), make_not_null(&tilde_ye),
      make_not_null(&tilde_tau), make_not_null(&tilde_s),
      make_not_null(&tilde_b), make_not_null(&tilde_phi), rest_mass_density,
      electron_fraction, specific_internal_energy, pressure, spatial_velocity,
      lorentz_factor, magnetic_field, sqrt_det_spatial_metric, spatial_metric,
      divergence_cleaning_field);

  // Compute the Valencia fluxes.
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_d_flux{num_pts};
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_ye_flux{num_pts};
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_tau_flux{num_pts};
  tnsr::Ij<DataVector, 3, Frame::Inertial> tilde_s_flux{num_pts};
  tnsr::IJ<DataVector, 3, Frame::Inertial> tilde_b_flux{num_pts};
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_phi_flux{num_pts};
  grmhd::ValenciaDivClean::ComputeFluxes::apply(
      make_not_null(&tilde_d_flux), make_not_null(&tilde_ye_flux),
      make_not_null(&tilde_tau_flux), make_not_null(&tilde_s_flux),
      make_not_null(&tilde_b_flux), make_not_null(&tilde_phi_flux), tilde_d,
      tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
      sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric, pressure,
      spatial_velocity, lorentz_factor, magnetic_field);

  // Independent computation of sqrt(gamma) * alpha^2 * T^{0i} from
  // primitives + geometry, for B = 0 perfect fluid:
  //   alpha^2 * T^{0i}  =  rho * h * W^2 * (alpha v^i - beta^i) + p * beta^i
  //   sqrt_gamma * alpha^2 * T^{0i}  =  sqrt_gamma * [ same ]
  //
  // Also: h = 1 + eps + p/rho (relativistic specific enthalpy).
  const Scalar<DataVector> specific_enthalpy =
      hydro::relativistic_specific_enthalpy(rest_mass_density,
                                            specific_internal_energy, pressure);
  const DataVector rho_h_w2 = get(rest_mass_density) * get(specific_enthalpy) *
                              square(get(lorentz_factor));
  for (size_t i = 0; i < 3; ++i) {
    const DataVector expected_sqrt_gamma_alpha_squared_T0i =
        get(sqrt_det_spatial_metric) *
        (rho_h_w2 * (get(lapse) * spatial_velocity.get(i) - shift.get(i)) +
         get(pressure) * shift.get(i));
    const DataVector flux_sum_i = tilde_d_flux.get(i) + tilde_tau_flux.get(i);
    CHECK_ITERABLE_APPROX(flux_sum_i, expected_sqrt_gamma_alpha_squared_T0i);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Fluxes", "[Unit][GrMhd]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  pypp::check_with_random_values<1>(
      &grmhd::ValenciaDivClean::ComputeFluxes::apply, "Fluxes",
      {"tilde_d_flux", "tilde_ye_flux", "tilde_tau_flux", "tilde_s_flux",
       "tilde_b_flux", "tilde_phi_flux"},
      {{{0.0, 1.0}}}, DataVector{5});

  // Rung 2 of the HLLC-GR validation ladder.
  test_energy_flux_identity_curved_hydro(DataVector{5});
}
