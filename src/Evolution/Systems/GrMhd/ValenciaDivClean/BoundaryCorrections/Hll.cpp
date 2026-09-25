// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"

#include <array>
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
    const gsl::not_null<Scalar<DataVector>*> packaged_fast_outgoing_char_speed,
    const gsl::not_null<Scalar<DataVector>*> packaged_fast_ingoing_char_speed,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_interface_unit_normal,
    const gsl::not_null<Scalar<DataVector>*> packaged_metric_flatness,

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
    const tnsr::ii<DataVector, 3, Frame::Inertial>& /*spatial_metric*/,

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

      const Scalar<DataVector> sound_speed_squared{
          clamp(get(equation_of_state
                        .sound_speed_squared_from_density_and_temperature(
                            rest_mass_density, temperature, electron_fraction)),
                0.0, 1.0)};

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

  *packaged_interface_unit_normal = normal_covector;
  get(*packaged_metric_flatness) = abs(get(lapse) - 1.0) + abs(get<0>(shift)) +
                                   abs(get<1>(shift)) + abs(get<2>(shift));

  // Fast-magnetosonic speeds of this side. In flat space the divergence
  // cleaning pair decouples and keeps the light-speed bounds; all other
  // variables use these speeds instead of the light speed, combined with the
  // neighbor's by the same two-sided min/max in dg_boundary_terms. The
  // estimate a^2 = c_s^2 + v_A^2 (1 - c_s^2) drops the dependence on B.n and is
  // an upper bound on the fast speed. On a curved background they are the
  // largest speeds.
  *packaged_fast_outgoing_char_speed = *packaged_largest_outgoing_char_speed;
  *packaged_fast_ingoing_char_speed = *packaged_largest_ingoing_char_speed;
  if (max(get(*packaged_metric_flatness)) <= 1.0e-12) {
    const ScopedFpeState fpe(false);
    const size_t num_points = get(rest_mass_density).size();
    const Scalar<DataVector> specific_enthalpy =
        hydro::relativistic_specific_enthalpy(
            rest_mass_density, specific_internal_energy, pressure);
    tnsr::ii<DataVector, 3, Frame::Inertial> flat_metric{num_points, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      flat_metric.get(i, i) = 1.0;
    }
    // In flat space TildeB^i = B^i.
    std::array<DataVector, 9> fast_speeds{};
    characteristic_speeds_approximate_mhd(
        make_not_null(&fast_speeds), rest_mass_density, electron_fraction,
        specific_internal_energy, specific_enthalpy, spatial_velocity,
        lorentz_factor, tilde_b, lapse, shift, flat_metric, normal_covector,
        equation_of_state);
    // Indices 1 and 7 are the ingoing and outgoing fast speeds.
    get(*packaged_fast_outgoing_char_speed) = fast_speeds[7];
    get(*packaged_fast_ingoing_char_speed) = fast_speeds[1];
    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_fast_outgoing_char_speed) -= get(*normal_dot_mesh_velocity);
      get(*packaged_fast_ingoing_char_speed) -= get(*normal_dot_mesh_velocity);
    }
    // Keep the light speed in the atmosphere, as the largest speeds do.
    for (size_t pt = 0; pt < num_points; ++pt) {
      if (get(rest_mass_density)[pt] <= light_speed_density_cutoff_) {
        get(*packaged_fast_outgoing_char_speed)[pt] =
            get(*packaged_largest_outgoing_char_speed)[pt];
        get(*packaged_fast_ingoing_char_speed)[pt] =
            get(*packaged_largest_ingoing_char_speed)[pt];
      }
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
    const Scalar<DataVector>& fast_outgoing_char_speed_int,
    const Scalar<DataVector>& fast_ingoing_char_speed_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_int,
    const Scalar<DataVector>& metric_flatness_int,
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
    const Scalar<DataVector>& fast_outgoing_char_speed_ext,
    const Scalar<DataVector>& fast_ingoing_char_speed_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_ext,
    const Scalar<DataVector>& metric_flatness_ext,
    const dg::Formulation dg_formulation) {
  const size_t num_points = get(tilde_d_int).size();
  const bool weak = dg_formulation == dg::Formulation::WeakInertial;

  Variables<tmpl::list<
      ::Tags::TempScalar<0>, ::Tags::TempScalar<1>, ::Tags::TempScalar<2>,
      ::Tags::TempScalar<3>, ::Tags::TempScalar<4>, ::Tags::TempScalar<5>,
      ::Tags::TempScalar<6>, ::Tags::TempScalar<7>, ::Tags::TempScalar<8>,
      ::Tags::TempScalar<9>, ::Tags::TempScalar<10>, ::Tags::TempScalar<11>,
      ::Tags::TempScalar<12>, ::Tags::TempScalar<13>, ::Tags::TempScalar<14>,
      ::Tags::TempScalar<15>, ::Tags::TempScalar<16>, ::Tags::TempScalar<17>,
      ::Tags::Tempi<0, 3, Frame::Inertial>>>
      temp_buffer{num_points};
  auto& lambda_max = get(get<::Tags::TempScalar<0>>(temp_buffer));
  auto& lambda_min = get(get<::Tags::TempScalar<1>>(temp_buffer));
  auto& lambda_dl = get(get<::Tags::TempScalar<2>>(temp_buffer));
  auto& lambda_lprod = get(get<::Tags::TempScalar<3>>(temp_buffer));
  auto& fast_max = get(get<::Tags::TempScalar<4>>(temp_buffer));
  auto& fast_min = get(get<::Tags::TempScalar<5>>(temp_buffer));
  auto& fast_dl = get(get<::Tags::TempScalar<6>>(temp_buffer));
  auto& fast_lprod = get(get<::Tags::TempScalar<7>>(temp_buffer));

  // Light-speed bounds for the divergence cleaning pair (Phi and the normal
  // magnetic field).
  lambda_max = max(0., get(largest_outgoing_char_speed_int),
                   -get(largest_ingoing_char_speed_ext));
  lambda_min = min(0., get(largest_ingoing_char_speed_int),
                   -get(largest_outgoing_char_speed_ext));
  // Fast-magnetosonic bounds for the other variables, from the two sides' own
  // speeds. The exterior packaged its speeds against its own outward normal,
  // hence the signs. On a curved background and in the atmosphere these are
  // the light speeds.
  fast_max = max(0., get(fast_outgoing_char_speed_int),
                 -get(fast_ingoing_char_speed_ext));
  fast_min = min(0., get(fast_ingoing_char_speed_int),
                 -get(fast_outgoing_char_speed_ext));
  lambda_dl = max(lambda_max - lambda_min, 1.0e-30);
  lambda_lprod = lambda_max * lambda_min;
  fast_dl = max(fast_max - fast_min, 1.0e-30);
  fast_lprod = fast_max * fast_min;

  // The HLL flux of one component for the bounds l_max, l_min, with
  // dl = l_max - l_min and lprod = l_max l_min.
  const auto hll = [&weak](const gsl::not_null<DataVector*> result,
                           const DataVector& l_max, const DataVector& l_min,
                           const DataVector& dl, const DataVector& lprod,
                           const DataVector& u_int, const DataVector& nf_int,
                           const DataVector& u_ext, const DataVector& nf_ext) {
    if (weak) {
      *result =
          (l_max * nf_int + l_min * nf_ext + lprod * (u_ext - u_int)) / dl;
    } else {
      *result = (l_min * (nf_int + nf_ext) + lprod * (u_ext - u_int)) / dl;
    }
  };

  hll(make_not_null(&get(*boundary_correction_tilde_d)), fast_max, fast_min,
      fast_dl, fast_lprod, get(tilde_d_int), get(normal_dot_flux_tilde_d_int),
      get(tilde_d_ext), get(normal_dot_flux_tilde_d_ext));
  hll(make_not_null(&get(*boundary_correction_tilde_ye)), fast_max, fast_min,
      fast_dl, fast_lprod, get(tilde_ye_int), get(normal_dot_flux_tilde_ye_int),
      get(tilde_ye_ext), get(normal_dot_flux_tilde_ye_ext));
  hll(make_not_null(&get(*boundary_correction_tilde_tau)), fast_max, fast_min,
      fast_dl, fast_lprod, get(tilde_tau_int),
      get(normal_dot_flux_tilde_tau_int), get(tilde_tau_ext),
      get(normal_dot_flux_tilde_tau_ext));
  hll(make_not_null(&get(*boundary_correction_tilde_phi)), lambda_max,
      lambda_min, lambda_dl, lambda_lprod, get(tilde_phi_int),
      get(normal_dot_flux_tilde_phi_int), get(tilde_phi_ext),
      get(normal_dot_flux_tilde_phi_ext));
  for (size_t i = 0; i < 3; ++i) {
    hll(make_not_null(&boundary_correction_tilde_s->get(i)), fast_max, fast_min,
        fast_dl, fast_lprod, tilde_s_int.get(i),
        normal_dot_flux_tilde_s_int.get(i), tilde_s_ext.get(i),
        normal_dot_flux_tilde_s_ext.get(i));
  }
  // On a curved background the whole magnetic field takes the light-speed
  // HLL flux. A face is flat or curved as a whole.
  if (max(get(metric_flatness_int)) > 1.0e-12 or
      max(get(metric_flatness_ext)) > 1.0e-12) {
    for (size_t i = 0; i < 3; ++i) {
      hll(make_not_null(&boundary_correction_tilde_b->get(i)), lambda_max,
          lambda_min, lambda_dl, lambda_lprod, tilde_b_int.get(i),
          normal_dot_flux_tilde_b_int.get(i), tilde_b_ext.get(i),
          normal_dot_flux_tilde_b_ext.get(i));
    }
    return;
  }
  // The normal part of the magnetic field uses the light-speed bounds, the
  // tangential part the fast bounds: G(B^i) = G(B_n) n^i + G(B_t^i). The two
  // sides' normals are opposite up to interpolation, so the split is along
  // their average, which makes the correction change sign exactly under the
  // int<->ext swap. In flat space raising the normal is trivial.
  auto& n = get<::Tags::Tempi<0, 3, Frame::Inertial>>(temp_buffer);
  auto& n_magnitude = get(get<::Tags::TempScalar<8>>(temp_buffer));
  for (size_t i = 0; i < 3; ++i) {
    n.get(i) = 0.5 * (interface_unit_normal_int.get(i) -
                      interface_unit_normal_ext.get(i));
  }
  n_magnitude =
      sqrt(n.get(0) * n.get(0) + n.get(1) * n.get(1) + n.get(2) * n.get(2));
  for (size_t i = 0; i < 3; ++i) {
    n.get(i) /= n_magnitude;
  }
  auto& bn_int = get(get<::Tags::TempScalar<9>>(temp_buffer));
  auto& bn_ext = get(get<::Tags::TempScalar<10>>(temp_buffer));
  auto& nfbn_int = get(get<::Tags::TempScalar<11>>(temp_buffer));
  auto& nfbn_ext = get(get<::Tags::TempScalar<12>>(temp_buffer));
  auto& g_bn = get(get<::Tags::TempScalar<13>>(temp_buffer));
  auto& bt_int = get(get<::Tags::TempScalar<14>>(temp_buffer));
  auto& bt_ext = get(get<::Tags::TempScalar<15>>(temp_buffer));
  auto& nfbt_int = get(get<::Tags::TempScalar<16>>(temp_buffer));
  auto& nfbt_ext = get(get<::Tags::TempScalar<17>>(temp_buffer));
  bn_int = 0.0;
  bn_ext = 0.0;
  nfbn_int = 0.0;
  nfbn_ext = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    bn_int += tilde_b_int.get(i) * n.get(i);
    bn_ext += tilde_b_ext.get(i) * n.get(i);
    nfbn_int += normal_dot_flux_tilde_b_int.get(i) * n.get(i);
    nfbn_ext += normal_dot_flux_tilde_b_ext.get(i) * n.get(i);
  }
  hll(make_not_null(&g_bn), lambda_max, lambda_min, lambda_dl, lambda_lprod,
      bn_int, nfbn_int, bn_ext, nfbn_ext);
  for (size_t i = 0; i < 3; ++i) {
    bt_int = tilde_b_int.get(i) - bn_int * n.get(i);
    bt_ext = tilde_b_ext.get(i) - bn_ext * n.get(i);
    nfbt_int = normal_dot_flux_tilde_b_int.get(i) - nfbn_int * n.get(i);
    nfbt_ext = normal_dot_flux_tilde_b_ext.get(i) - nfbn_ext * n.get(i);
    hll(make_not_null(&boundary_correction_tilde_b->get(i)), fast_max, fast_min,
        fast_dl, fast_lprod, bt_int, nfbt_int, bt_ext, nfbt_ext);
    boundary_correction_tilde_b->get(i) += g_bn * n.get(i);
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
