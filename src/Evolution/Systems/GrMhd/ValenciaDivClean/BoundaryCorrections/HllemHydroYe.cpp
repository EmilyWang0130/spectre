// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/HllemHydroYe.hpp"

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
HllemHydroYe::HllemHydroYe(const double magnetic_field_magnitude_for_hydro,
                           const double light_speed_density_cutoff,
                           const bool restore_middle_block,
                           const bool use_physical_zeta)
    : magnetic_field_magnitude_for_hydro_(magnetic_field_magnitude_for_hydro),
      light_speed_density_cutoff_(light_speed_density_cutoff),
      restore_middle_block_(restore_middle_block),
      use_physical_zeta_(use_physical_zeta) {}

HllemHydroYe::HllemHydroYe(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> HllemHydroYe::get_clone() const {
  return std::make_unique<HllemHydroYe>(*this);
}

void HllemHydroYe::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | magnetic_field_magnitude_for_hydro_;
  p | light_speed_density_cutoff_;
  p | restore_middle_block_;
  p | use_physical_zeta_;
}

double HllemHydroYe::dg_package_data(
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
    const gsl::not_null<Scalar<DataVector>*> packaged_temperature,
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
    // Default packaged sound speed to zero -- sentinel used by
    // dg_boundary_terms's fast-bounds fallback (see Hll.cpp for the
    // rationale). Populated below in the hydro / above-atmosphere branch.
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
  // bounds in dg_boundary_terms (as in Hll), plus Y_e and Temperature which
  // will feed the hydro+Y_e eigensystem construction in phase 2.
  *packaged_rest_mass_density = rest_mass_density;
  *packaged_electron_fraction = electron_fraction;
  *packaged_temperature = temperature;
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

void HllemHydroYe::dg_boundary_terms(
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
    const Scalar<DataVector>& rest_mass_density_int,
    const Scalar<DataVector>& electron_fraction_int,
    const Scalar<DataVector>& sound_speed_squared_int,
    const Scalar<DataVector>& /*temperature_int*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_int,
    const Scalar<DataVector>& /*pressure_int*/,
    const Scalar<DataVector>& /*lorentz_factor_int*/,
    const Scalar<DataVector>& specific_internal_energy_int,
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
    const Scalar<DataVector>& rest_mass_density_ext,
    const Scalar<DataVector>& electron_fraction_ext,
    const Scalar<DataVector>& sound_speed_squared_ext,
    const Scalar<DataVector>& /*temperature_ext*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_ext,
    const Scalar<DataVector>& /*pressure_ext*/,
    const Scalar<DataVector>& /*lorentz_factor_ext*/,
    const Scalar<DataVector>& specific_internal_energy_ext,
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
  // Middle-block projector storage (populated only when we take the flat +
  // hydro branch AND restore_middle_block_ is true).
  bool have_middle_block = false;
  // characteristic_eigenvectors_hydro uses tnsr::ij for the right-eigenvector
  // "modes" and tnsr::IJ for the left-eigenvector "projectors"; keep the same
  // convention here.
  tnsr::ij<DataVector, 6> hydro_right{num_points, 0.0};
  tnsr::IJ<DataVector, 6> hydro_left{num_points, 0.0};
  DataVector lambda_mid{num_points, 0.0};
  // Flat + hydro (both packaged c_s^2 > 0) => outer fluid bounds from the
  // closed-form lambda_pm(v_avg, c_s^2_avg), using packaged per-side data
  // only. Zero EOS calls on the fast-bounds path -- the previous
  // "average primitives, EOS-lookup for p/h, then characteristic_speeds_..."
  // recipe cost ~4 Togashi-trilinear lookups per interface per step. See
  // Hll.cpp for the full rationale (this class inherits the same design
  // decision).
  //
  // If restore_middle_block_ = true we still need the eigensystem, which in
  // turn needs the on-EOS (p_avg, h_avg) at the averaged (rho, eps, Y_e). We
  // fall into a second block that does that lookup once, only when middle-
  // block anti-diffusion is requested.
  const bool flat = max(get(metric_flatness_int)) <= 1.0e-12 and
                    max(get(metric_flatness_ext)) <= 1.0e-12 and
                    min(get(sound_speed_squared_int)) > 0.0 and
                    min(get(sound_speed_squared_ext)) > 0.0;
  tnsr::I<DataVector, 3, Frame::Inertial> v_avg{num_points, 0.0};
  DataVector v_sq_avg{num_points, 0.0};
  DataVector v_n_avg{num_points, 0.0};
  if (flat) {
    const ScopedFpeState fpe(false);
    for (size_t i = 0; i < 3; ++i) {
      v_avg.get(i) =
          0.5 * (spatial_velocity_int.get(i) + spatial_velocity_ext.get(i));
      v_sq_avg += v_avg.get(i) * v_avg.get(i);
      v_n_avg += v_avg.get(i) * interface_unit_normal_int.get(i);
    }
    v_sq_avg = clamp(v_sq_avg, 0.0, 1.0 - 1.0e-10);
    const DataVector cs2_avg =
        0.5 * (get(sound_speed_squared_int) + get(sound_speed_squared_ext));
    const DataVector one_minus_cs2 = 1.0 - cs2_avg;
    const DataVector one_minus_v2_cs2 = 1.0 - v_sq_avg * cs2_avg;
    const DataVector disc =
        sqrt(clamp(cs2_avg * (1.0 - v_sq_avg) *
                       (one_minus_v2_cs2 - v_n_avg * v_n_avg * one_minus_cs2),
                   0.0, 1.0));
    // In flat space lapse = 1 and shift = 0, so the alpha/beta terms drop.
    const DataVector lambda_plus =
        (v_n_avg * one_minus_cs2 + disc) / one_minus_v2_cs2;
    const DataVector lambda_minus =
        (v_n_avg * one_minus_cs2 - disc) / one_minus_v2_cs2;
    fast_max = max(0.0, lambda_plus);
    fast_min = min(0.0, lambda_minus);
  }

  // Middle-block eigensystem (only when the anti-diffusion is enabled).
  // This still costs one EOS lookup per interface (for the on-EOS
  // (p_avg, h_avg)); it's the price of the design's thermodynamic
  // consistency in the eigenvector construction. Kept gated so
  // RestoreMiddleBlock = false remains cheap.
  if (flat and restore_middle_block_) {
    const ScopedFpeState fpe(false);
    const Scalar<DataVector> rho_avg{
        0.5 * (get(rest_mass_density_int) + get(rest_mass_density_ext))};
    const Scalar<DataVector> eps_avg{0.5 * (get(specific_internal_energy_int) +
                                            get(specific_internal_energy_ext))};
    const Scalar<DataVector> ye_avg{
        0.5 * (get(electron_fraction_int) + get(electron_fraction_ext))};
    const Scalar<DataVector> w_avg{1.0 / sqrt(1.0 - v_sq_avg)};
    const Scalar<DataVector> p_avg_eos =
        equation_of_state.pressure_from_density_and_energy(rho_avg, eps_avg,
                                                           ye_avg);
    const Scalar<DataVector> h_avg_eos =
        hydro::relativistic_specific_enthalpy(rho_avg, eps_avg, p_avg_eos);
    tnsr::ii<DataVector, 3, Frame::Inertial> flat_metric{num_points, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      flat_metric.get(i, i) = 1.0;
    }
    characteristic_eigenvectors_hydro(
        make_not_null(&hydro_right), make_not_null(&hydro_left), v_avg, rho_avg,
        eps_avg, h_avg_eos, ye_avg, w_avg, interface_unit_normal_int,
        flat_metric, equation_of_state);
    lambda_mid = v_n_avg;  // v_avg . n = HydroSpeed::NormalDotVelocity
    have_middle_block = true;
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

  // Middle-block anti-diffusion (design v2 sec 1, 7). Only active in the
  // flat + hydro branch where we built the eigensystem above.
  //
  //   F_new = F_HLL - (S_L S_R / (S_R - S_L)) * delta_mid * P_mid * (U_R - U_L)
  //
  // with P_mid = I - P_+ - P_-, P_+/- = R_+/- (x) L_+/- / (L_+/- . R_+/-).
  // Ordering of the 6 slots follows characteristic_eigenvectors_hydro:
  //   [D, S_x, S_y, S_z, tau, DYe].
  if (have_middle_block) {
    const std::array<const DataVector*, 6> u_int_c{
        {&get(tilde_d_int), &get<0>(tilde_s_int), &get<1>(tilde_s_int),
         &get<2>(tilde_s_int), &get(tilde_tau_int), &get(tilde_ye_int)}};
    const std::array<const DataVector*, 6> u_ext_c{
        {&get(tilde_d_ext), &get<0>(tilde_s_ext), &get<1>(tilde_s_ext),
         &get<2>(tilde_s_ext), &get(tilde_tau_ext), &get(tilde_ye_ext)}};
    const std::array<DataVector*, 6> corr_c{
        {&get(*boundary_correction_tilde_d),
         &get<0>(*boundary_correction_tilde_s),
         &get<1>(*boundary_correction_tilde_s),
         &get<2>(*boundary_correction_tilde_s),
         &get(*boundary_correction_tilde_tau),
         &get(*boundary_correction_tilde_ye)}};
    constexpr size_t plus_idx = HydroVectorR::Rplus;
    constexpr size_t minus_idx = HydroVectorR::Rminus;
    for (size_t pt = 0; pt < num_points; ++pt) {
      const double s_l = fast_min[pt];
      const double s_r = fast_max[pt];
      const double denom = s_r - s_l;
      // Same guard as the HLL flux; skip anti-diffusion when there is no
      // HLL diffusion to remove (both bounds ~ 0).
      if (denom < 1.0e-30) {
        continue;
      }
      const double lam = lambda_mid[pt];
      const double delta_mid =
          1.0 - std::min(0.0, lam) / (s_l - 1.0e-30) -
          std::max(0.0, lam) / (s_r + 1.0e-30);
      const double coeff = -s_l * s_r / denom * delta_mid;
      // Biorthogonal (not biorthonormal) diagonals.
      double diag_plus = 0.0;
      double diag_minus = 0.0;
      for (size_t n = 0; n < 6; ++n) {
        diag_plus += hydro_left.get(plus_idx, n)[pt] *
                     hydro_right.get(plus_idx, n)[pt];
        diag_minus += hydro_left.get(minus_idx, n)[pt] *
                      hydro_right.get(minus_idx, n)[pt];
      }
      // Skip acoustic projectors that are ill-conditioned at this point
      // (would only be true at strong degeneracies -- shouldn't happen for
      // R+, R- which are non-degenerate by construction, but guard anyway).
      const double diag_tol = 1.0e-12;
      const bool plus_ok = std::abs(diag_plus) > diag_tol;
      const bool minus_ok = std::abs(diag_minus) > diag_tol;
      if (not plus_ok and not minus_ok) {
        continue;
      }
      // L_+/- . (U_R - U_L) at this point.
      double alpha_plus = 0.0;
      double alpha_minus = 0.0;
      std::array<double, 6> du{};
      for (size_t n = 0; n < 6; ++n) {
        du[n] = (*gsl::at(u_ext_c, n))[pt] - (*gsl::at(u_int_c, n))[pt];
        if (plus_ok) {
          alpha_plus += hydro_left.get(plus_idx, n)[pt] * du[n];
        }
        if (minus_ok) {
          alpha_minus += hydro_left.get(minus_idx, n)[pt] * du[n];
        }
      }
      if (plus_ok) {
        alpha_plus /= diag_plus;
      }
      if (minus_ok) {
        alpha_minus /= diag_minus;
      }
      // ΔU_mid = ΔU - alpha_+ R_+ - alpha_- R_-, and add to correction.
      for (size_t n = 0; n < 6; ++n) {
        double du_mid = du[n];
        if (plus_ok) {
          du_mid -= alpha_plus * hydro_right.get(plus_idx, n)[pt];
        }
        if (minus_ok) {
          du_mid -= alpha_minus * hydro_right.get(minus_idx, n)[pt];
        }
        (*gsl::at(corr_c, n))[pt] += coeff * du_mid;
      }
    }
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

bool operator==(const HllemHydroYe& lhs, const HllemHydroYe& rhs) {
  return lhs.magnetic_field_magnitude_for_hydro_ ==
             rhs.magnetic_field_magnitude_for_hydro_ and
         lhs.light_speed_density_cutoff_ == rhs.light_speed_density_cutoff_ and
         lhs.restore_middle_block_ == rhs.restore_middle_block_ and
         lhs.use_physical_zeta_ == rhs.use_physical_zeta_;
}
bool operator!=(const HllemHydroYe& lhs, const HllemHydroYe& rhs) {
  return not(lhs == rhs);
}

// NOLINTNEXTLINE
PUP::able::PUP_ID HllemHydroYe::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
