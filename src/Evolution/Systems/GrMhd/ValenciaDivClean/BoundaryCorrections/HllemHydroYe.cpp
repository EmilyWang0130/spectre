// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/HllemHydroYe.hpp"

#include <cmath>
#include <pup.h>

#include <memory>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/HatTransform.hpp"
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
                           const bool restore_middle_block)
    : magnetic_field_magnitude_for_hydro_(magnetic_field_magnitude_for_hydro),
      light_speed_density_cutoff_(light_speed_density_cutoff),
      restore_middle_block_(restore_middle_block) {}

HllemHydroYe::HllemHydroYe(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> HllemHydroYe::get_clone() const {
  return std::make_unique<HllemHydroYe>(*this);
}

void HllemHydroYe::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | magnetic_field_magnitude_for_hydro_;
  p | light_speed_density_cutoff_;
  p | restore_middle_block_;
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
    const gsl::not_null<Scalar<DataVector>*> packaged_lapse_at_interface,
    const gsl::not_null<Scalar<DataVector>*> packaged_shift_dot_normal,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        packaged_spatial_metric,
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
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,

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

      // Only the sound speed enters the characteristic speeds below. Earlier
      // revisions also evaluated the specific internal energy, the pressure
      // and the specific enthalpy here, but nothing ever read them: those
      // locals SHADOWED the same-named function parameters, and it is the
      // parameters that `*packaged_pressure` and
      // `*packaged_specific_internal_energy` are assigned from below. On a
      // tabulated 3D EOS the chain dominated this function's cost; Hll.cpp
      // deleted the same three declarations in 136fc413c.
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

    // Correct for mesh velocity. The same n . v_mesh is folded into the
    // packaged beta^n so that lambda = alpha nu - beta^n_eff holds for the
    // packaged speeds; `dg_boundary_terms` inverts exactly that relation to
    // map the Eulerian-frame eigenvalues into the coordinate frame.
    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_largest_outgoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
      get(*packaged_largest_ingoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
      get(shift_dot_normal) += get(*normal_dot_mesh_velocity);
    }

    // 3+1 geometry for the interface frame (see HatTransform.hpp). Packaged
    // raw and per-side; combined into one frame in `dg_boundary_terms`,
    // because building the frame per side would hand the Riemann solver two
    // states in different frames.
    *packaged_lapse_at_interface = lapse;
    *packaged_shift_dot_normal = shift_dot_normal;
  }

  // Package the interface unit normal and metric. These are needed to build
  // the averaged interface state for the MIDDLE-BLOCK eigensystem, not for the
  // outer bounds: dg_boundary_terms takes S_L, S_R from the per-side speeds
  // packaged just above (Recipe A), as Hll does.
  //
  // NOTE: an earlier revision computed the outer bounds at the averaged state
  // too, to fix a top/bottom asymmetry the per-side, sign-of-v_n-dependent
  // bounds seeded in the Kelvin-Helmholtz test, and argued that sharing the
  // state gave S_L <= lambda_mid <= S_R by construction. That recipe
  // under-bounds the fan and was removed; the delta_mid range now rests on a
  // sound-speed margin instead. See the class documentation.
  *packaged_interface_unit_normal = normal_covector;
  *packaged_spatial_metric = spatial_metric;

  // Package the primitives needed to build the hydro+Y_e eigensystem in
  // dg_boundary_terms. Note that Pressure, LorentzFactor and Temperature are
  // packaged but not currently read there: the eigensystem is built from an
  // on-EOS (p, h) at the AVERAGED (rho, eps, Y_e) rather than from averaged
  // per-side values.
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
    const Scalar<DataVector>& lapse_at_interface_int,
    const Scalar<DataVector>& shift_dot_normal_int,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric_int,
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
    const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_ext,
    const Scalar<DataVector>& lapse_at_interface_ext,
    const Scalar<DataVector>& shift_dot_normal_ext,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric_ext,
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
  // Fast-magnetosonic HLL bounds: used for the MHD variables (D, Ye, Tau, S),
  // which travel slower than light.
  //
  // NOTE: an earlier revision evaluated these at the ARITHMETIC-AVERAGE
  // interface state, to make the bounds invariant under v_n -> -v_n (a
  // top/bottom asymmetry in the Kelvin-Helmholtz test) and to guarantee
  // S_L <= lambda_mid <= S_R by construction. That is not what HLLEM
  // prescribes, and it is where the 2026-09-15 flat-space bug lived. The
  // wider per-side interval still contains lambda_mid, so delta_mid in [0, 1]
  // survives -- see below.
  //
  // Fluid-variable bounds: the per-side max/min combination (Recipe A,
  // Davis 1988), identical to Hll.cpp:132-133. HLLEM requires lambda_L and
  // lambda_R to be an UPPER BOUND on the true wave speeds, estimated from the
  // left and right INPUT states -- see Mattia & Mignone 2021 (arXiv:2111.09369)
  // sec. "HLL Formulation": the averaged (*) state enters ONLY the
  // anti-diffusion term, via R*, L* and lambda_{m,*}. Evaluating the outer
  // bounds at the averaged state instead ("Recipe B") under-bounds the fan at
  // strongly-asymmetric interfaces and violates the Harten-Lax-van Leer
  // premise; Hll.cpp:124-131 records that it was tried there and reverted.
  const DataVector& fast_max = lambda_max;
  const DataVector& fast_min = lambda_min;
  // Middle-block projector storage (populated only when we take the hydro
  // branch AND restore_middle_block_ is true).
  bool have_middle_block = false;
  // characteristic_eigenvectors_hydro uses tnsr::ij for the right-eigenvector
  // "modes" and tnsr::IJ for the left-eigenvector "projectors"; keep the same
  // convention here.
  tnsr::ij<DataVector, 6> hydro_right{num_points, 0.0};
  tnsr::IJ<DataVector, 6> hydro_left{num_points, 0.0};
  DataVector lambda_mid{num_points, 0.0};
  // Hydro branch: both sides packaged a positive sound speed, i.e. both sides
  // are above the atmosphere cutoff and free of a magnetic field above
  // MagneticFieldMagnitudeForHydro. The outer bounds now cost nothing extra --
  // they are the packaged per-side speeds, as in Hll.
  //
  // The averaged interface state below is built ONLY when
  // restore_middle_block_ is true, because only the anti-diffusion term needs
  // it (R*, L*, lambda_{m,*}). A consequence worth keeping: with
  // RestoreMiddleBlock = false this class is now bit-identical to Hll.
  //
  // Unlike the earlier flat-space-only version, this branch has no
  // metric-flatness gate: the eigenvectors of the coordinate-frame Jacobian
  // A_coord = alpha A_Eulerian - beta^n_eff I are the Eulerian-frame ones, so
  // characteristic_eigenvectors_hydro can be used as is on a curved
  // background, with the eigenvalues mapped by HatTransform::inverse_speed.
  // See the class documentation in HllemHydroYe.hpp.
  const bool hydro_branch = min(get(sound_speed_squared_int)) > 0.0 and
                            min(get(sound_speed_squared_ext)) > 0.0;
  // One-frame interface geometry, built from the two packaged sides
  // (HatTransform.hpp; HllcGr plan sec 3). Scalars and the spatial metric are
  // averaged symmetrically; quantities linear in the normal (n_i itself and
  // beta^n_eff) antisymmetrically, which is what conservation,
  // G(int, ext) = -G(ext, int), requires on slightly mismatched faces.
  tnsr::I<DataVector, 3, Frame::Inertial> v_avg{num_points, 0.0};
  tnsr::i<DataVector, 3, Frame::Inertial> n_avg{num_points, 0.0};
  tnsr::ii<DataVector, 3, Frame::Inertial> gamma_avg{num_points, 0.0};
  DataVector alpha_avg{num_points, 1.0};
  DataVector beta_n_avg{num_points, 0.0};
  DataVector v_sq_avg{num_points, 0.0};
  DataVector v_n_avg{num_points, 0.0};
  if (hydro_branch and restore_middle_block_) {
    const ScopedFpeState fpe(false);
    alpha_avg = interface_lapse(get(lapse_at_interface_int),
                                get(lapse_at_interface_ext));
    beta_n_avg = interface_shift_dot_normal(get(shift_dot_normal_int),
                                            get(shift_dot_normal_ext));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        gamma_avg.get(i, j) =
            0.5 * (spatial_metric_int.get(i, j) + spatial_metric_ext.get(i, j));
      }
      n_avg.get(i) = 0.5 * (interface_unit_normal_int.get(i) -
                            interface_unit_normal_ext.get(i));
      v_avg.get(i) =
          0.5 * (spatial_velocity_int.get(i) + spatial_velocity_ext.get(i));
    }
    const auto inv_gamma_avg = determinant_and_inverse(gamma_avg).second;
    // Averaging does not preserve gamma^ij n_i n_j = 1, and each side's
    // normal was normalized in its own metric, so renormalize in the averaged
    // metric before handing the normal to the eigensystem (which assumes a
    // unit normal).
    DataVector normal_norm_squared{num_points, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        normal_norm_squared +=
            inv_gamma_avg.get(i, j) * n_avg.get(i) * n_avg.get(j);
      }
    }
    const DataVector inv_normal_norm = 1.0 / sqrt(normal_norm_squared);
    for (size_t i = 0; i < 3; ++i) {
      n_avg.get(i) *= inv_normal_norm;
    }
    // v^2 = gamma_ij v^i v^j and v_n = v^i n_i at the averaged state.
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        v_sq_avg += gamma_avg.get(i, j) * v_avg.get(i) * v_avg.get(j);
      }
      v_n_avg += v_avg.get(i) * n_avg.get(i);
    }
    v_sq_avg = clamp(v_sq_avg, 0.0, 1.0 - 1.0e-10);
  }

  // Middle-block eigensystem (only when the anti-diffusion is enabled).
  // This still costs one EOS lookup per interface (for the on-EOS
  // (p_avg, h_avg)); it's the price of the design's thermodynamic
  // consistency in the eigenvector construction. Kept gated so
  // RestoreMiddleBlock = false remains cheap.
  if (hydro_branch and restore_middle_block_) {
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
    characteristic_eigenvectors_hydro(
        make_not_null(&hydro_right), make_not_null(&hydro_left), v_avg, rho_avg,
        eps_avg, h_avg_eos, ye_avg, w_avg, n_avg, gamma_avg, equation_of_state);
    // Eulerian-frame degenerate eigenvalue nu_mid = v . n (=
    // HydroSpeed::NormalDotVelocity), mapped to the coordinate frame. Using
    // the Eulerian value here instead would put the middle wave at the wrong
    // place in the fan whenever alpha != 1 or beta^n != 0.
    lambda_mid = coordinate_frame_speed(alpha_avg, beta_n_avg, v_n_avg);
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
  // hydro branch where we built the eigensystem above. Curved backgrounds
  // are handled by taking S_L, S_R and lambda_mid in the COORDINATE frame
  // (the projectors are frame-independent); see the class documentation for
  // why the flat-space formula is exact there rather than approximate.
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
      const double delta_mid = 1.0 - std::min(0.0, lam) / (s_l - 1.0e-30) -
                               std::max(0.0, lam) / (s_r + 1.0e-30);
      const double coeff = -s_l * s_r / denom * delta_mid;
      // Biorthogonal (not biorthonormal) diagonals.
      double diag_plus = 0.0;
      double diag_minus = 0.0;
      for (size_t n = 0; n < 6; ++n) {
        diag_plus +=
            hydro_left.get(plus_idx, n)[pt] * hydro_right.get(plus_idx, n)[pt];
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

  // Magnetic field: plain HLL with light-speed bounds, matching Hll.cpp
  // (develop's pre-f5b73d016 behaviour). The GLM / MHD normal-tangential
  // split that used to live here was gated on flat space and, where it was
  // active, numerically vacuous: the split only differs from plain HLL when
  // the fast and light bounds differ, which requires the hydro branch, which
  // in turn requires |B| < MagneticFieldMagnitudeForHydro (1e-30 by default).
  for (size_t i = 0; i < 3; ++i) {
    boundary_correction_tilde_b->get(i) =
        hll(lambda_max, lambda_min, tilde_b_int.get(i),
            normal_dot_flux_tilde_b_int.get(i), tilde_b_ext.get(i),
            normal_dot_flux_tilde_b_ext.get(i));
  }
}

bool operator==(const HllemHydroYe& lhs, const HllemHydroYe& rhs) {
  return lhs.magnetic_field_magnitude_for_hydro_ ==
             rhs.magnetic_field_magnitude_for_hydro_ and
         lhs.light_speed_density_cutoff_ == rhs.light_speed_density_cutoff_ and
         lhs.restore_middle_block_ == rhs.restore_middle_block_;
}
bool operator!=(const HllemHydroYe& lhs, const HllemHydroYe& rhs) {
  return not(lhs == rhs);
}

// NOLINTNEXTLINE
PUP::able::PUP_ID HllemHydroYe::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
