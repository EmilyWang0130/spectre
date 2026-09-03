// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"

#include <cmath>
#include <pup.h>

#include <memory>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "PointwiseFunctions/Hydro/SoundSpeedSquared.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
Hll::Hll(const double magnetic_field_magnitude_for_hydro,
         const double light_speed_density_cutoff)
    : magnetic_field_magnitude_for_hydro_(magnetic_field_magnitude_for_hydro),
      light_speed_density_cutoff_(light_speed_density_cutoff) {}

Hll::Hll(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> Hll::get_clone() const {
  return std::make_unique<Hll>(*this);
}

void Hll::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | magnetic_field_magnitude_for_hydro_;
  p | light_speed_density_cutoff_;
}

double Hll::dg_package_data(
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
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_interface_unit_normal,
    const gsl::not_null<Scalar<DataVector>*> packaged_metric_flatness,
    const gsl::not_null<Scalar<DataVector>*> packaged_rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> packaged_electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> packaged_sound_speed_squared,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> packaged_pressure,
    const gsl::not_null<Scalar<DataVector>*> packaged_lorentz_factor,
    const gsl::not_null<Scalar<DataVector>*> packaged_specific_internal_energy,

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

    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& temperature,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& pressure,
    const Scalar<DataVector>& lorentz_factor,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) const {
  {
    Scalar<DataVector> shift_dot_normal = tilde_d;
    dot_product(make_not_null(&shift_dot_normal), shift, normal_covector);

    // Initialize characteristic speeds to light speed
    // (default choice)
    get(*packaged_largest_outgoing_char_speed) =
        get(lapse) - get(shift_dot_normal);
    get(*packaged_largest_ingoing_char_speed) =
        -get(lapse) - get(shift_dot_normal);
    // Default packaged sound speed to zero. dg_boundary_terms uses this as
    // a "no averaged-state fast bounds available" sentinel and falls back
    // to the packaged light-speed bounds; only populated below in the
    // hydro / above-atmosphere branch.
    get(*packaged_sound_speed_squared) = 0.0;

    if (const bool has_b_field =
            max(get(magnitude(tilde_b))) > magnetic_field_magnitude_for_hydro_;
        not has_b_field and
        (max(get(rest_mass_density)) > light_speed_density_cutoff_)) {
      // Since we have no magnetic field (and we have grid points not in the
      // atmosphere), we can reduce the char speeds to the hydro speeds only.
      // This makes the scheme less dissipative.
      //
      // Calculate sound speed squared
      const size_t num_points = get(rest_mass_density).size();
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
      const Scalar<DataVector> pressure =
          equation_of_state.pressure_from_density_and_energy(
              rest_mass_density, specific_internal_energy, electron_fraction);
      const Scalar<DataVector> specific_enthalpy =
          hydro::relativistic_specific_enthalpy(
              rest_mass_density, specific_internal_energy, pressure);
      const Scalar<DataVector> sound_speed_squared{
          clamp(get(equation_of_state
                        .sound_speed_squared_from_density_and_temperature(
                            rest_mass_density, temperature, electron_fraction)),
                0.0, 1.0)};
      // Save the sound speed for the averaged-state fast bounds in
      // dg_boundary_terms (avoids a second EOS lookup there).
      *packaged_sound_speed_squared = sound_speed_squared;

      // Compute v_dot_normal, v^i n_i
      dot_product(make_not_null(&v_dot_normal), spatial_velocity,
                  normal_covector);

      // Compute v^2=v^i v_i
      dot_product(make_not_null(&v_squared), spatial_velocity,
                  spatial_velocity_one_form);
      get(v_squared) = clamp(get(v_squared), 0.0, 1.0 - 1.0e-8);

      // Calculate characteristic speeds in inertial frame
      //
      // Ideally we'd use the Lorentz factor instead of 1-v^2, but I (Nils
      // Deppe) don't have the bandwidth to change this right now.
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

    // Correct for mesh velocity
    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_largest_outgoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
      get(*packaged_largest_ingoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
    }
  }

  // Package the interface unit normal (for the scalar/MHD field split) and the
  // metric-flatness measure (the split only holds in flat space). The
  // fast-magnetosonic HLL bounds are no longer computed per-side here; they are
  // computed at the AVERAGED interface state inside dg_boundary_terms
  // (mirroring the HLLEM boundary correction), which fixes a top/bottom
  // asymmetry that the per-side, sign-of-v_n-dependent bounds seeded in the
  // Kelvin-Helmholtz test.
  *packaged_interface_unit_normal = normal_covector;
  get(*packaged_metric_flatness) = abs(get(lapse) - 1.0) + abs(get<0>(shift)) +
                                   abs(get<1>(shift)) + abs(get<2>(shift));

  // Package the primitives needed to reconstruct the averaged fast-magnetosonic
  // bounds in dg_boundary_terms.
  *packaged_rest_mass_density = rest_mass_density;
  *packaged_electron_fraction = electron_fraction;
  for (size_t i = 0; i < 3; ++i) {
    packaged_spatial_velocity->get(i) = spatial_velocity.get(i);
  }
  *packaged_pressure = pressure;
  *packaged_lorentz_factor = lorentz_factor;
  *packaged_specific_internal_energy = specific_internal_energy;

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

  using std::max;
  return max(max(abs(get(*packaged_largest_outgoing_char_speed))),
             max(abs(get(*packaged_largest_ingoing_char_speed))));
}

void Hll::dg_boundary_terms(
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
    const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_int,
    const Scalar<DataVector>& metric_flatness_int,
    const Scalar<DataVector>& /*rest_mass_density_int*/,
    const Scalar<DataVector>& /*electron_fraction_int*/,
    const Scalar<DataVector>& sound_speed_squared_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_int,
    const Scalar<DataVector>& /*pressure_int*/,
    const Scalar<DataVector>& /*lorentz_factor_int*/,
    const Scalar<DataVector>& /*specific_internal_energy_int*/,
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
    const tnsr::i<DataVector, 3, Frame::Inertial>& /*iface_normal_ext*/,
    const Scalar<DataVector>& metric_flatness_ext,
    const Scalar<DataVector>& /*rest_mass_density_ext*/,
    const Scalar<DataVector>& /*electron_fraction_ext*/,
    const Scalar<DataVector>& sound_speed_squared_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_ext,
    const Scalar<DataVector>& /*pressure_ext*/,
    const Scalar<DataVector>& /*lorentz_factor_ext*/,
    const Scalar<DataVector>& /*specific_internal_energy_ext*/,
    const dg::Formulation dg_formulation,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) const {
  const size_t num_points = get(tilde_d_int).size();
  const bool weak = dg_formulation == dg::Formulation::WeakInertial;

  // Light-speed (divergence-cleaning) HLL bounds: used for Phi and the normal
  // magnetic field, which form the decoupled GLM subsystem at +/-c.
  const DataVector lambda_max = max(0., get(largest_outgoing_char_speed_int),
                                    -get(largest_ingoing_char_speed_ext));
  const DataVector lambda_min = min(0., get(largest_ingoing_char_speed_int),
                                    -get(largest_outgoing_char_speed_ext));
  // Fast-magnetosonic HLL bounds: used for the MHD variables (D, Ye, Tau, S and
  // the tangential magnetic field), which travel slower than light.
  //
  // The bounds are computed from the fast-magnetosonic characteristic speeds
  // evaluated at the ARITHMETIC-AVERAGE interface state (exactly like the HLLEM
  // boundary correction). Evaluating a single interface eigensystem -- rather
  // than the previous per-side, sign-of-v_n-dependent packaged fast speeds --
  // makes the bounds identical on mirror-image faces (invariant under
  // v_n -> -v_n) and removes a small top/bottom asymmetry that the per-side
  // approach seeded in the Kelvin-Helmholtz test. In curved space the flat
  // decomposition does not hold, so we fall back to the light bounds and the
  // scheme reduces to the standard HLL flux there.
  DataVector fast_max = lambda_max;
  DataVector fast_min = lambda_min;
  // Optimised averaged-state fast bounds. Iago's 96d94d83e (which fixed the
  // KH mirror-y asymmetry) originally recomputed characteristic_speeds_...
  // at the arithmetic-averaged primitive state via an EOS call. That cost
  // ~4 Togashi-trilinear lookups per interface per step; on non-equilibrium
  // 3D tables it dominated HLL's wallclock. The symmetric-bound property
  // (invariance under v_n -> -v_n at a shear interface) only requires the
  // AVERAGED (v, c_s^2), so we now:
  //   * package the per-side sound-speed-squared alongside the light-speed
  //     bounds (already computed in dg_package_data; formerly discarded);
  //   * average v and c_s^2 arithmetically at the interface;
  //   * rebuild lambda_+/- via the closed-form
  //         lambda_pm = alpha/(1 - v^2 c_s^2)
  //                     * (v_n (1 - c_s^2) +/- c_s sqrt((1-v^2)(1 - v^2 c_s^2
  //                                                       - v_n^2 (1 -
  //                                                       c_s^2))))
  //                     - beta^n
  //     from the same closed-form dg_package_data uses. Zero EOS calls.
  // Applies only to the hydro sub-branch (both packaged c_s^2 > 0, which
  // implies |B| < threshold and rho > cutoff on both sides). MHD-fast bounds
  // require the fast-magnetosonic dispersion at the averaged state; the
  // current KHI use case keeps the old characteristic_speeds_mhd path there
  // via a fallback to the light bounds when either packaged c_s^2 is zero
  // (which the MHD sub-branch signals). A follow-up can package c_fast^2
  // per side too for the same optimisation on the MHD side.
  if (max(get(metric_flatness_int)) <= 1.0e-12 and
      max(get(metric_flatness_ext)) <= 1.0e-12 and
      min(get(sound_speed_squared_int)) > 0.0 and
      min(get(sound_speed_squared_ext)) > 0.0) {
    const ScopedFpeState fpe(false);
    const DataVector cs2_avg =
        0.5 * (get(sound_speed_squared_int) + get(sound_speed_squared_ext));
    DataVector v_sq_avg{num_points, 0.0};
    DataVector v_n_avg{num_points, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      const DataVector vi_avg =
          0.5 * (spatial_velocity_int.get(i) + spatial_velocity_ext.get(i));
      v_sq_avg += vi_avg * vi_avg;
      v_n_avg += vi_avg * interface_unit_normal_int.get(i);
    }
    v_sq_avg = clamp(v_sq_avg, 0.0, 1.0 - 1.0e-10);
    const DataVector one_minus_cs2 = 1.0 - cs2_avg;
    const DataVector one_minus_v2_cs2 = 1.0 - v_sq_avg * cs2_avg;
    // Discriminant clamped at 0 for numerical safety near the sonic point.
    const DataVector disc =
        sqrt(clamp(cs2_avg * (1.0 - v_sq_avg) *
                       (one_minus_v2_cs2 - v_n_avg * v_n_avg * one_minus_cs2),
                   0.0, 1.0));
    // In flat space lapse = 1, shift = 0 (implied by both metric_flatnesses
    // <= 1e-12), so the lapse/shift terms drop out.
    const DataVector lambda_plus =
        (v_n_avg * one_minus_cs2 + disc) / one_minus_v2_cs2;
    const DataVector lambda_minus =
        (v_n_avg * one_minus_cs2 - disc) / one_minus_v2_cs2;
    fast_max = max(0.0, lambda_plus);
    fast_min = min(0.0, lambda_minus);
  }

  // HLL flux for one conserved component given the bounds l_max, l_min.
  const auto hll = [&weak, &num_points](
                       const DataVector& l_max, const DataVector& l_min,
                       const DataVector& u_int, const DataVector& nf_int,
                       const DataVector& u_ext,
                       const DataVector& nf_ext) -> DataVector {
    DataVector dl = l_max - l_min;
    for (size_t pt = 0; pt < num_points; ++pt) {
      if (dl[pt] < 1.0e-30) {
        dl[pt] = 1.0e-30;
      }
    }
    const DataVector lprod = l_max * l_min;
    if (weak) {
      return (l_max * nf_int + l_min * nf_ext + lprod * (u_ext - u_int)) / dl;
    }
    return (l_min * (nf_int + nf_ext) + lprod * (u_ext - u_int)) / dl;
  };

  // Fluid scalars: pure MHD -> fast bounds.
  get(*boundary_correction_tilde_d) = hll(
      fast_max, fast_min, get(tilde_d_int), get(normal_dot_flux_tilde_d_int),
      get(tilde_d_ext), get(normal_dot_flux_tilde_d_ext));
  get(*boundary_correction_tilde_ye) = hll(
      fast_max, fast_min, get(tilde_ye_int), get(normal_dot_flux_tilde_ye_int),
      get(tilde_ye_ext), get(normal_dot_flux_tilde_ye_ext));
  get(*boundary_correction_tilde_tau) =
      hll(fast_max, fast_min, get(tilde_tau_int),
          get(normal_dot_flux_tilde_tau_int), get(tilde_tau_ext),
          get(normal_dot_flux_tilde_tau_ext));
  // Divergence-cleaning scalar Phi: light speed.
  get(*boundary_correction_tilde_phi) =
      hll(lambda_max, lambda_min, get(tilde_phi_int),
          get(normal_dot_flux_tilde_phi_int), get(tilde_phi_ext),
          get(normal_dot_flux_tilde_phi_ext));
  // Momentum: pure MHD -> fast bounds.
  for (size_t i = 0; i < 3; ++i) {
    boundary_correction_tilde_s->get(i) =
        hll(fast_max, fast_min, tilde_s_int.get(i),
            normal_dot_flux_tilde_s_int.get(i), tilde_s_ext.get(i),
            normal_dot_flux_tilde_s_ext.get(i));
  }
  // Magnetic field: the NORMAL component is part of the GLM subsystem (light
  // speed), the TANGENTIAL component is MHD (fast bounds). Decompose along the
  // interface normal, treat each part with its own bounds, and recombine
  // G(B^i) = G(B_n) n^i + G(B_t^i). (In flat space n is a unit covector and the
  // metric is the identity, so raising/lowering the normal is trivial.)
  {
    const auto& n = interface_unit_normal_int;
    DataVector bn_int{num_points, 0.0};
    DataVector bn_ext{num_points, 0.0};
    DataVector nfbn_int{num_points, 0.0};
    DataVector nfbn_ext{num_points, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      bn_int += tilde_b_int.get(i) * n.get(i);
      bn_ext += tilde_b_ext.get(i) * n.get(i);
      nfbn_int += normal_dot_flux_tilde_b_int.get(i) * n.get(i);
      nfbn_ext += normal_dot_flux_tilde_b_ext.get(i) * n.get(i);
    }
    const DataVector g_bn =
        hll(lambda_max, lambda_min, bn_int, nfbn_int, bn_ext, nfbn_ext);
    // The decomposition uses n as both covector and (raised) vector, which is
    // only valid in flat space; where the background is curved fall back to the
    // plain (light-speed) HLL flux for B, which keeps the scheme conservative.
    for (size_t i = 0; i < 3; ++i) {
      const DataVector bt_int = tilde_b_int.get(i) - bn_int * n.get(i);
      const DataVector bt_ext = tilde_b_ext.get(i) - bn_ext * n.get(i);
      const DataVector nfbt_int =
          normal_dot_flux_tilde_b_int.get(i) - nfbn_int * n.get(i);
      const DataVector nfbt_ext =
          normal_dot_flux_tilde_b_ext.get(i) - nfbn_ext * n.get(i);
      const DataVector g_bt =
          hll(fast_max, fast_min, bt_int, nfbt_int, bt_ext, nfbt_ext);
      const DataVector g_split = g_bn * n.get(i) + g_bt;
      const DataVector g_plain =
          hll(lambda_max, lambda_min, tilde_b_int.get(i),
              normal_dot_flux_tilde_b_int.get(i), tilde_b_ext.get(i),
              normal_dot_flux_tilde_b_ext.get(i));
      for (size_t pt = 0; pt < num_points; ++pt) {
        boundary_correction_tilde_b->get(i)[pt] =
            (get(metric_flatness_int)[pt] > 1.0e-12 or
             get(metric_flatness_ext)[pt] > 1.0e-12)
                ? g_plain[pt]
                : g_split[pt];
      }
    }
  }
}

bool operator==(const Hll& lhs, const Hll& rhs) {
  return lhs.magnetic_field_magnitude_for_hydro_ ==
             rhs.magnetic_field_magnitude_for_hydro_ and
         lhs.light_speed_density_cutoff_ == rhs.light_speed_density_cutoff_;
}
bool operator!=(const Hll& lhs, const Hll& rhs) { return not(lhs == rhs); }

// NOLINTNEXTLINE
PUP::able::PUP_ID Hll::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
