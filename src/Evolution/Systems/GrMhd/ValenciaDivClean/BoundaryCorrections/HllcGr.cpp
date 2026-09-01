// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/HllcGr.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "PointwiseFunctions/Hydro/SoundSpeedSquared.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
HllcGr::HllcGr(const double magnetic_field_magnitude_for_hydro,
               const double light_speed_density_cutoff)
    : magnetic_field_magnitude_for_hydro_(magnetic_field_magnitude_for_hydro),
      light_speed_density_cutoff_(light_speed_density_cutoff) {}

HllcGr::HllcGr(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> HllcGr::get_clone() const {
  return std::make_unique<HllcGr>(*this);
}

void HllcGr::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | magnetic_field_magnitude_for_hydro_;
  p | light_speed_density_cutoff_;
}

double HllcGr::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_largest_outgoing_char_speed,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_largest_ingoing_char_speed,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_tilde_s,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_normal_dot_flux_normal_dot_tilde_s,
    const gsl::not_null<Scalar<DataVector>*> packaged_advection_speed,
    const gsl::not_null<Scalar<DataVector>*> packaged_pressure_flux_coefficient,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_interface_normal_covector,

    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
    const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,

    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_d,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_ye,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_tau,
    const tnsr::Ij<DataVector, 3, Frame::Inertial>& flux_tilde_s,
    const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_tilde_b,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_phi,

    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3, Frame::Inertial>& spatial_velocity_one_form,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,

    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& temperature,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& /*specific_internal_energy*/,
    const Scalar<DataVector>& pressure,
    const Scalar<DataVector>& /*lorentz_factor*/,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) const {
  const size_t num_points = get(rest_mass_density).size();

  // Char speeds: reuse the HLL logic (light-speed default, hydro speeds when
  // B is negligible and density is above atmosphere).
  {
    Scalar<DataVector> shift_dot_normal = tilde_d;
    dot_product(make_not_null(&shift_dot_normal), shift, normal_covector);

    get(*packaged_largest_outgoing_char_speed) =
        get(lapse) - get(shift_dot_normal);
    get(*packaged_largest_ingoing_char_speed) =
        -get(lapse) - get(shift_dot_normal);

    if (const bool has_b_field =
            max(get(magnitude(tilde_b))) > magnetic_field_magnitude_for_hydro_;
        not has_b_field and
        (max(get(rest_mass_density)) > light_speed_density_cutoff_)) {
      Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                           ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                           ::Tags::TempScalar<4>, ::Tags::TempScalar<5>,
                           ::Tags::TempScalar<6>>>
          temp_buffer{num_points};
      auto& v_dot_normal = get<::Tags::TempScalar<0>>(temp_buffer);
      auto& v_squared = get<::Tags::TempScalar<1>>(temp_buffer);
      auto& discriminant = get(get<::Tags::TempScalar<2>>(temp_buffer));
      auto& one_minus_v2_cs2 = get<::Tags::TempScalar<3>>(temp_buffer);
      auto& one_minus_cs2 = get<::Tags::TempScalar<4>>(temp_buffer);
      auto& lapse_over_one_minus_v2_cs2 =
          get<::Tags::TempScalar<5>>(temp_buffer);
      auto& v_dot_normal_times_one_minus_cs2 =
          get<::Tags::TempScalar<6>>(temp_buffer);

      const Scalar<DataVector> specific_internal_energy =
          equation_of_state
              .specific_internal_energy_from_density_and_temperature(
                  rest_mass_density, temperature, electron_fraction);
      const Scalar<DataVector> sound_speed_squared{
          clamp(get(equation_of_state
                        .sound_speed_squared_from_density_and_temperature(
                            rest_mass_density, temperature, electron_fraction)),
                0.0, 1.0)};

      dot_product(make_not_null(&v_dot_normal), spatial_velocity,
                  normal_covector);
      dot_product(make_not_null(&v_squared), spatial_velocity,
                  spatial_velocity_one_form);
      get(v_squared) = clamp(get(v_squared), 0.0, 1.0 - 1.0e-8);

      get(one_minus_v2_cs2) = 1.0 - get(v_squared) * get(sound_speed_squared);
      get(one_minus_cs2) = 1.0 - get(sound_speed_squared);
      discriminant =
          get(sound_speed_squared) * (1.0 - get(v_squared)) *
          (get(one_minus_v2_cs2) -
           get(v_dot_normal) * get(v_dot_normal) * get(one_minus_cs2));
      discriminant = max(discriminant, 0.0);
      discriminant = sqrt(discriminant);

      get(lapse_over_one_minus_v2_cs2) = get(lapse) / get(one_minus_v2_cs2);
      get(v_dot_normal_times_one_minus_cs2) =
          get(v_dot_normal) * get(one_minus_cs2);

      for (size_t i = 0; i < num_points; ++i) {
        if (get(rest_mass_density)[i] > light_speed_density_cutoff_) {
          get(*packaged_largest_outgoing_char_speed)[i] =
              get(lapse_over_one_minus_v2_cs2)[i] *
                  (get(v_dot_normal_times_one_minus_cs2)[i] + discriminant[i]) -
              get(shift_dot_normal)[i];
          get(*packaged_largest_ingoing_char_speed)[i] =
              get(lapse_over_one_minus_v2_cs2)[i] *
                  (get(v_dot_normal_times_one_minus_cs2)[i] - discriminant[i]) -
              get(shift_dot_normal)[i];
        }
      }
    }

    // Coordinate advection speed u = alpha v^n - beta^n (before mesh
    // velocity), needed for the HLLC star-state formulas.
    Scalar<DataVector> v_dot_normal = tilde_d;
    dot_product(make_not_null(&v_dot_normal), spatial_velocity,
                normal_covector);
    get(*packaged_advection_speed) =
        get(lapse) * get(v_dot_normal) - get(shift_dot_normal);

    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_largest_outgoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
      get(*packaged_largest_ingoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
      get(*packaged_advection_speed) -= get(*normal_dot_mesh_velocity);
    }
  }

  *packaged_tilde_d = tilde_d;
  *packaged_tilde_ye = tilde_ye;
  *packaged_tilde_tau = tilde_tau;
  *packaged_tilde_s = tilde_s;
  *packaged_tilde_b = tilde_b;
  *packaged_tilde_phi = tilde_phi;

  normal_dot_flux(packaged_normal_dot_flux_tilde_d, normal_covector,
                  flux_tilde_d);
  normal_dot_flux(packaged_normal_dot_flux_tilde_ye, normal_covector,
                  flux_tilde_ye);
  normal_dot_flux(packaged_normal_dot_flux_tilde_tau, normal_covector,
                  flux_tilde_tau);
  normal_dot_flux(packaged_normal_dot_flux_tilde_s, normal_covector,
                  flux_tilde_s);
  normal_dot_flux(packaged_normal_dot_flux_tilde_b, normal_covector,
                  flux_tilde_b);
  normal_dot_flux(packaged_normal_dot_flux_tilde_phi, normal_covector,
                  flux_tilde_phi);

  // The FD subcell reconstruction path in
  //   Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivative.hpp
  // (see lines 316-359 and the "we probably should compute the normal
  // vector in addition to the co-vector" comment) does not populate the
  // NormalVector<3> tag that ends up being passed as `normal_vector` --
  // it holds uninitialised memory when the subcell path calls us. Compute
  // the raised normal locally from `spatial_metric` and `normal_covector`
  // instead. (Marquina does the same, see Marquina.cpp.)
  (void)normal_vector;
  const auto local_inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  tnsr::I<DataVector, 3, Frame::Inertial> local_normal_vector{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      local_normal_vector.get(i) +=
          local_inv_spatial_metric.get(i, j) * normal_covector.get(j);
    }
  }

  // Normal projection of the covariant momentum, M = n^i tilde_S_i.
  dot_product(packaged_normal_dot_tilde_s, local_normal_vector, tilde_s);
  // Normal-normal projection of the covariant momentum flux,
  // F(M) = n^i (n_j F^j(tilde_S_i)). We reuse the already-computed
  // packaged_normal_dot_flux_tilde_s as (n_j F^j).
  dot_product(packaged_normal_dot_flux_normal_dot_tilde_s, local_normal_vector,
              *packaged_normal_dot_flux_tilde_s);

  // Pressure-flux coefficient P_tilde = alpha sqrt(gamma) p. Compute
  // sqrt(det(gamma_ij)) locally rather than requesting it as a temporary
  // tag, to avoid perturbing the merged Variables list that the ghost
  // boundary conditions have to fill.
  Scalar<DataVector> sqrt_det_spatial_metric_local = pressure;
  determinant(make_not_null(&sqrt_det_spatial_metric_local), spatial_metric);
  get(sqrt_det_spatial_metric_local) = sqrt(get(sqrt_det_spatial_metric_local));
  get(*packaged_pressure_flux_coefficient) =
      get(lapse) * get(sqrt_det_spatial_metric_local) * get(pressure);

  // Store the interior normal covector so the boundary_terms function can
  // reconstruct the covariant star momentum tilde_S^*_i.
  *packaged_interface_normal_covector = normal_covector;

  using std::max;
  return max(max(abs(get(*packaged_largest_outgoing_char_speed))),
             max(abs(get(*packaged_largest_ingoing_char_speed))));
}

void HllcGr::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
    const Scalar<DataVector>& tilde_d_int,
    const Scalar<DataVector>& tilde_ye_int,
    const Scalar<DataVector>& tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
    const Scalar<DataVector>& tilde_phi_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_ye_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_int,
    const Scalar<DataVector>& largest_outgoing_char_speed_int,
    const Scalar<DataVector>& largest_ingoing_char_speed_int,
    const Scalar<DataVector>& normal_dot_tilde_s_int,
    const Scalar<DataVector>& normal_dot_flux_normal_dot_tilde_s_int,
    const Scalar<DataVector>& advection_speed_int,
    const Scalar<DataVector>& pressure_flux_coefficient_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        interface_normal_covector_int,
    const Scalar<DataVector>& tilde_d_ext,
    const Scalar<DataVector>& tilde_ye_ext,
    const Scalar<DataVector>& tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_ext,
    const Scalar<DataVector>& tilde_phi_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_ye_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_ext,
    const Scalar<DataVector>& largest_outgoing_char_speed_ext,
    const Scalar<DataVector>& largest_ingoing_char_speed_ext,
    const Scalar<DataVector>& normal_dot_tilde_s_ext,
    const Scalar<DataVector>& normal_dot_flux_normal_dot_tilde_s_ext,
    const Scalar<DataVector>& advection_speed_ext,
    const Scalar<DataVector>& pressure_flux_coefficient_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        interface_normal_covector_ext,
    const dg::Formulation dg_formulation) {
  const size_t num_points = get(tilde_d_int).size();
  const bool weak_form = dg_formulation == dg::Formulation::WeakInertial;

  // The MB05 HLLC algorithm is inherently branchy (four regions selected
  // by the sign of lambda_L, lambda*, lambda_R), so we do the work in a
  // per-point loop. Sign convention: on the "L" side (interior) all
  // packaged quantities keep their sign; on the "R" side (exterior) flux-
  // like quantities and quantities linear in the normal flip sign, because
  // the exterior packaged its data with its own outward normal (opposite
  // to the interior's). Even-normal-count scalars (F(M), P_tilde) do not
  // flip. See the notes in HllcGr.hpp for the derivation.
  for (size_t i = 0; i < num_points; ++i) {
    // Signal speeds in the interior's normal frame, clamped so
    // lambda_L <= 0 <= lambda_R by construction.
    const double lambda_R =
        std::max({0.0, get(largest_outgoing_char_speed_int)[i],
                  -get(largest_ingoing_char_speed_ext)[i]});
    const double lambda_L =
        std::min({0.0, get(largest_ingoing_char_speed_int)[i],
                  -get(largest_outgoing_char_speed_ext)[i]});

    // Interior-frame L (= int) state / flux / scalars.
    const double D_L = get(tilde_d_int)[i];
    const double Ye_L = get(tilde_ye_int)[i];
    const double Tau_L = get(tilde_tau_int)[i];
    const double Phi_L = get(tilde_phi_int)[i];
    const double E_L = Tau_L + D_L;  // MB05's E, densitized
    const double M_L = get(normal_dot_tilde_s_int)[i];
    const double u_L = get(advection_speed_int)[i];
    const double Ptilde_L = get(pressure_flux_coefficient_int)[i];

    const double f_D_L = get(normal_dot_flux_tilde_d_int)[i];
    const double f_Ye_L = get(normal_dot_flux_tilde_ye_int)[i];
    const double f_Tau_L = get(normal_dot_flux_tilde_tau_int)[i];
    const double f_Phi_L = get(normal_dot_flux_tilde_phi_int)[i];
    const double f_M_L = get(normal_dot_flux_normal_dot_tilde_s_int)[i];
    const double f_E_L = f_Tau_L + f_D_L;

    // Interior-frame R (= ext) state / flux / scalars. State variables and
    // even-normal-count scalars are unflipped; flux-like quantities and
    // quantities linear in the normal are sign-flipped.
    const double D_R = get(tilde_d_ext)[i];
    const double Ye_R = get(tilde_ye_ext)[i];
    const double Tau_R = get(tilde_tau_ext)[i];
    const double Phi_R = get(tilde_phi_ext)[i];
    const double E_R = Tau_R + D_R;
    const double M_R = -get(normal_dot_tilde_s_ext)[i];
    const double u_R = -get(advection_speed_ext)[i];
    const double Ptilde_R = get(pressure_flux_coefficient_ext)[i];

    const double f_D_R = -get(normal_dot_flux_tilde_d_ext)[i];
    const double f_Ye_R = -get(normal_dot_flux_tilde_ye_ext)[i];
    const double f_Tau_R = -get(normal_dot_flux_tilde_tau_ext)[i];
    const double f_Phi_R = -get(normal_dot_flux_tilde_phi_ext)[i];
    const double f_M_R = get(normal_dot_flux_normal_dot_tilde_s_ext)[i];
    const double f_E_R = f_Tau_R + f_D_R;

    // Interior-frame per-index R quantities for tilde_S and tilde_B and
    // their fluxes. tilde_S_i, tilde_B^i (state) do not flip; their normal
    // fluxes flip.
    std::array<double, 3> S_L{{get<0>(tilde_s_int)[i], get<1>(tilde_s_int)[i],
                               get<2>(tilde_s_int)[i]}};
    std::array<double, 3> S_R{{get<0>(tilde_s_ext)[i], get<1>(tilde_s_ext)[i],
                               get<2>(tilde_s_ext)[i]}};
    std::array<double, 3> B_L{{get<0>(tilde_b_int)[i], get<1>(tilde_b_int)[i],
                               get<2>(tilde_b_int)[i]}};
    std::array<double, 3> B_R{{get<0>(tilde_b_ext)[i], get<1>(tilde_b_ext)[i],
                               get<2>(tilde_b_ext)[i]}};
    std::array<double, 3> f_S_L{{get<0>(normal_dot_flux_tilde_s_int)[i],
                                 get<1>(normal_dot_flux_tilde_s_int)[i],
                                 get<2>(normal_dot_flux_tilde_s_int)[i]}};
    std::array<double, 3> f_S_R{{-get<0>(normal_dot_flux_tilde_s_ext)[i],
                                 -get<1>(normal_dot_flux_tilde_s_ext)[i],
                                 -get<2>(normal_dot_flux_tilde_s_ext)[i]}};
    std::array<double, 3> f_B_L{{get<0>(normal_dot_flux_tilde_b_int)[i],
                                 get<1>(normal_dot_flux_tilde_b_int)[i],
                                 get<2>(normal_dot_flux_tilde_b_int)[i]}};
    std::array<double, 3> f_B_R{{-get<0>(normal_dot_flux_tilde_b_ext)[i],
                                 -get<1>(normal_dot_flux_tilde_b_ext)[i],
                                 -get<2>(normal_dot_flux_tilde_b_ext)[i]}};

    // Interior normal covector for the star-flux ansatz F*(S_i) =
    // S*_i lambda* + P* n_i. Averaged as (n_int - n_ext)/2 — with a
    // perfectly-matched face (n_ext = -n_int) this equals n_int, but on
    // slightly-mismatched normals (as the test helper generates by
    // perturbing the exterior metric) the average is exactly antisymmetric
    // under int/ext swap, which is what the conservation identity
    // G(int, ext) = -G(ext, int) requires.
    const std::array<double, 3> n_i{
        {0.5 * (get<0>(interface_normal_covector_int)[i] -
                get<0>(interface_normal_covector_ext)[i]),
         0.5 * (get<1>(interface_normal_covector_int)[i] -
                get<1>(interface_normal_covector_ext)[i]),
         0.5 * (get<2>(interface_normal_covector_int)[i] -
                get<2>(interface_normal_covector_ext)[i])}};

    // Helper: HLL flux for a state variable (odd-normal-count quantity, so
    // the ext-flip sign is baked in via the "+ f_R" combination). This is
    // used as the fallback / for the magnetic + divergence-cleaning
    // sector, where HLLC doesn't add contact resolution.
    const double delta_lambda = lambda_R - lambda_L;
    const auto hll_flux = [lambda_L, lambda_R, delta_lambda](
                              const double f_L, const double f_R,
                              const double U_L, const double U_R) {
      // Note: f_R here is already in the L-frame (sign-flipped from ext).
      return (lambda_R * f_L - lambda_L * f_R +
              lambda_L * lambda_R * (U_R - U_L)) /
             delta_lambda;
    };

    // Region selection follows MB05 §3.1.3.
    // For the trivial supersonic branches we pick a fixed side's flux; for
    // the star region we solve the MB05 quadratic for lambda*.

    // We assemble G_* for each component in the interior's normal frame.
    // For weak form, boundary_correction = G. For strong form,
    // boundary_correction = G - f_int.
    double G_D = 0.0;
    double G_Ye = 0.0;
    double G_Tau = 0.0;
    std::array<double, 3> G_S{{0.0, 0.0, 0.0}};
    std::array<double, 3> G_B{{0.0, 0.0, 0.0}};
    double G_Phi = 0.0;

    if (lambda_L >= 0.0) {
      // Fully outgoing: numerical flux = f_L (packaged interior flux).
      G_D = f_D_L;
      G_Ye = f_Ye_L;
      G_Tau = f_Tau_L;
      G_S = f_S_L;
      G_B = f_B_L;
      G_Phi = f_Phi_L;
    } else if (lambda_R <= 0.0) {
      // Fully incoming: numerical flux = f_R (packaged exterior flux in
      // interior's normal frame; already sign-flipped above).
      G_D = f_D_R;
      G_Ye = f_Ye_R;
      G_Tau = f_Tau_R;
      G_S = f_S_R;
      G_B = f_B_R;
      G_Phi = f_Phi_R;
    } else {
      // Star region. Solve the MB05 quadratic (Eq. 18).
      // HLL averages for (E, M) and their fluxes (in interior's frame).
      const double E_hll =
          (lambda_R * E_R - lambda_L * E_L + f_E_L - f_E_R) / delta_lambda;
      const double M_hll =
          (lambda_R * M_R - lambda_L * M_L + f_M_L - f_M_R) / delta_lambda;
      const double f_E_hll = (lambda_R * f_E_L - lambda_L * f_E_R +
                              lambda_L * lambda_R * (E_R - E_L)) /
                             delta_lambda;
      const double f_M_hll = (lambda_R * f_M_L - lambda_L * f_M_R +
                              lambda_L * lambda_R * (M_R - M_L)) /
                             delta_lambda;

      // MB05 Eq. 18 quadratic in the form
      //   a lambda*^2 - b lambda* + c = 0,
      // with a = F_E_hll, b = E_hll + F_M_hll, c = M_hll. Solve using
      // Numerical Recipes' stable-root form to avoid catastrophic
      // cancellation in b - sqrt(D). PLUTO's `Src/RHD/hllc.c` uses the
      // same trick. No fallback -- MB05 proves lambda* is real and inside
      // [lambda_L, lambda_R] for physical L/R states, and we trust the
      // reconstructor to hand us physical states (any failure is upstream).
      const double a = f_E_hll;
      const double b = E_hll + f_M_hll;
      const double c = M_hll;
      // MB05 guarantees b^2 - 4ac >= 0 for physical states. The `max(0, .)`
      // is a defensive `sqrt` guard against unphysical L/R data (e.g. the
      // random-data unit test) -- not an algorithmic fallback to HLL.
      const double discriminant = std::sqrt(std::max(0.0, b * b - 4.0 * a * c));
      // DIAG: check for degenerate quadratic (division by zero in either
      // form) or non-finite inputs, and dump the offending L, R state.
      const bool bad_rat_branch =
          (b >= 0.0) and std::abs(b + discriminant) < 1.0e-30;
      const bool bad_dir_branch = (b < 0.0) and std::abs(a) < 1.0e-30;
      const bool bad_input =
          not std::isfinite(a) or not std::isfinite(b) or not std::isfinite(c);
      if (bad_rat_branch or bad_dir_branch or bad_input) {
        ERROR("[HllcGr DIAG] Degenerate quadratic at grid point i="
              << i << "\n  a=" << a << " b=" << b << " c=" << c
              << " disc=" << discriminant << "\n  lambda_L=" << lambda_L
              << " lambda_R=" << lambda_R << "\n  L: D=" << D_L
              << " Ye=" << Ye_L << " Tau=" << Tau_L << " Phi=" << Phi_L
              << " M=" << M_L << " u=" << u_L << " Ptilde=" << Ptilde_L
              << "\n     Sx=" << S_L[0] << " Sy=" << S_L[1] << " Sz=" << S_L[2]
              << "\n     fD=" << f_D_L << " fE=" << f_E_L << " fM=" << f_M_L
              << " fPhi=" << f_Phi_L << "\n     fSx=" << f_S_L[0] << " fSy="
              << f_S_L[1] << " fSz=" << f_S_L[2] << "\n  R: D=" << D_R
              << " Ye=" << Ye_R << " Tau=" << Tau_R << " Phi=" << Phi_R
              << " M=" << M_R << " u=" << u_R << " Ptilde=" << Ptilde_R
              << "\n     Sx=" << S_R[0] << " Sy=" << S_R[1] << " Sz=" << S_R[2]
              << "\n     fD=" << f_D_R << " fE=" << f_E_R << " fM=" << f_M_R
              << " fPhi=" << f_Phi_R << "\n     fSx=" << f_S_R[0]
              << " fSy=" << f_S_R[1] << " fSz=" << f_S_R[2] << "\n  n_i=("
              << n_i[0] << ", " << n_i[1] << ", " << n_i[2] << ")");
      }
      const double lambda_star = (b >= 0.0) ? (2.0 * c / (b + discriminant))
                                            : ((b - discriminant) / (2.0 * a));

      // Common star pressure from the HLL consistency relations:
      //   Ptilde_star = f_M_hll - lambda_star * f_E_hll.
      // This follows from MB05 Eq. (18) together with
      //   M_hll = lambda_star * (E_hll + Ptilde_star),
      // and is equivalent to the corrected form of MB05 Eq. (17),
      //   (A + lambda Ptilde_star) lambda_star = B + Ptilde_star.
      // Using the HLL averages avoids choosing one side and preserves the
      // int/ext swap symmetry up to roundoff.
      const double Ptilde_star = f_M_hll - lambda_star * f_E_hll;

      // Star states via MB05 Eq. 16, in the interior's normal frame.
      // Common factor r_alpha = (lambda_alpha - u_alpha) / (lambda_alpha
      // - lambda*). transverse/passive quantities scale by r; normal
      // momentum and energy pick up pressure-difference terms.
      auto star_state = [lambda_star, Ptilde_star, &n_i](
                            const double lambda_alpha, const double u_alpha,
                            const double D_alpha, const double Ye_alpha,
                            const double Tau_alpha, const double E_alpha,
                            const double M_alpha, const double Ptilde_alpha,
                            const std::array<double, 3>& S_alpha,
                            double& D_star, double& Ye_star, double& Tau_star,
                            double& M_star, std::array<double, 3>& S_star) {
        const double denom = lambda_alpha - lambda_star;
        const double r = (lambda_alpha - u_alpha) / denom;
        D_star = D_alpha * r;
        Ye_star = Ye_alpha * r;
        const double E_star =
            (E_alpha * (lambda_alpha - u_alpha) + Ptilde_star * lambda_star -
             Ptilde_alpha * u_alpha) /
            denom;
        Tau_star = E_star - D_star;
        // Consistency: keep Tau* via E* - D* rather than through a
        // separate MB05-style formula to avoid roundoff drift.
        (void)Tau_alpha;
        M_star =
            (M_alpha * (lambda_alpha - u_alpha) + Ptilde_star - Ptilde_alpha) /
            denom;
        const double pressure_jump = Ptilde_star - Ptilde_alpha;
        for (size_t k = 0; k < 3; ++k) {
          gsl::at(S_star, k) = (gsl::at(S_alpha, k) * (lambda_alpha - u_alpha) +
                                pressure_jump * gsl::at(n_i, k)) /
                               denom;
        }
      };

      double D_star_L = 0.0, Ye_star_L = 0.0, Tau_star_L = 0.0, M_star_L = 0.0;
      std::array<double, 3> S_star_L{{0.0, 0.0, 0.0}};
      double D_star_R = 0.0, Ye_star_R = 0.0, Tau_star_R = 0.0, M_star_R = 0.0;
      std::array<double, 3> S_star_R{{0.0, 0.0, 0.0}};
      star_state(lambda_L, u_L, D_L, Ye_L, Tau_L, E_L, M_L, Ptilde_L, S_L,
                 D_star_L, Ye_star_L, Tau_star_L, M_star_L, S_star_L);
      star_state(lambda_R, u_R, D_R, Ye_R, Tau_R, E_R, M_R, Ptilde_R, S_R,
                 D_star_R, Ye_star_R, Tau_star_R, M_star_R, S_star_R);

      // Star flux via Rankine-Hugoniot: F*_alpha = F_alpha + lambda_alpha
      // (U*_alpha - U_alpha). Guaranteed conservative.
      if (lambda_star >= 0.0) {
        G_D = f_D_L + lambda_L * (D_star_L - D_L);
        G_Ye = f_Ye_L + lambda_L * (Ye_star_L - Ye_L);
        G_Tau = f_Tau_L + lambda_L * (Tau_star_L - Tau_L);
        for (size_t k = 0; k < 3; ++k) {
          gsl::at(G_S, k) = gsl::at(f_S_L, k) +
                            lambda_L * (gsl::at(S_star_L, k) - gsl::at(S_L, k));
        }
      } else {
        G_D = f_D_R + lambda_R * (D_star_R - D_R);
        G_Ye = f_Ye_R + lambda_R * (Ye_star_R - Ye_R);
        G_Tau = f_Tau_R + lambda_R * (Tau_star_R - Tau_R);
        for (size_t k = 0; k < 3; ++k) {
          gsl::at(G_S, k) = gsl::at(f_S_R, k) +
                            lambda_R * (gsl::at(S_star_R, k) - gsl::at(S_R, k));
        }
      }
      // Ensure the M / S consistency: (n^i S*_i) should equal M*_side
      // (the scalar we solved for). Not enforced hard here; the numerical
      // consistency is checked in the unit test.
      (void)M_star_L;
      (void)M_star_R;

      // Magnetic + divergence-cleaning sector: HLL flux. HLLC doesn't
      // resolve the Alfven / slow waves; the class is documented as B = 0
      // only. The supersonic branches above already handled the trivial
      // cases; this only runs in the star region.
      G_Phi = hll_flux(f_Phi_L, f_Phi_R, Phi_L, Phi_R);
      for (size_t k = 0; k < 3; ++k) {
        gsl::at(G_B, k) = hll_flux(gsl::at(f_B_L, k), gsl::at(f_B_R, k),
                                   gsl::at(B_L, k), gsl::at(B_R, k));
      }
    }

    // Convert G to the boundary correction expected by the DG action.
    // Weak form: correction = G; strong form: correction = G - f_L.
    const double sub = weak_form ? 0.0 : 1.0;
    get(*boundary_correction_tilde_d)[i] = G_D - sub * f_D_L;
    get(*boundary_correction_tilde_ye)[i] = G_Ye - sub * f_Ye_L;
    get(*boundary_correction_tilde_tau)[i] = G_Tau - sub * f_Tau_L;
    get(*boundary_correction_tilde_phi)[i] = G_Phi - sub * f_Phi_L;
    for (size_t k = 0; k < 3; ++k) {
      boundary_correction_tilde_s->get(k)[i] =
          gsl::at(G_S, k) - sub * gsl::at(f_S_L, k);
      boundary_correction_tilde_b->get(k)[i] =
          gsl::at(G_B, k) - sub * gsl::at(f_B_L, k);
    }
  }
}

bool operator==(const HllcGr& lhs, const HllcGr& rhs) {
  return lhs.magnetic_field_magnitude_for_hydro_ ==
             rhs.magnetic_field_magnitude_for_hydro_ and
         lhs.light_speed_density_cutoff_ == rhs.light_speed_density_cutoff_;
}
bool operator!=(const HllcGr& lhs, const HllcGr& rhs) {
  return not(lhs == rhs);
}

// NOLINTNEXTLINE
PUP::able::PUP_ID HllcGr::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
