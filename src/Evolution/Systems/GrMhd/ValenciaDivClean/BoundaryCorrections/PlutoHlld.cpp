// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoHlld.hpp"

#include <string>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Parallel/Printf/Printf.hpp"

#include <algorithm>
#include <cstdlib>
#include <array>
#include <cmath>
#include <pup.h>

#include <memory>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoFaceFrame.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/pluto/pluto_hlld_shim.h"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {

namespace {
// Orthonormal triad (n, t1, t2) from a unit normal, in flat (Euclidean) space.
void make_triad(const std::array<double, 3>& n, std::array<double, 3>* t1,
                std::array<double, 3>* t2) {
  const std::array<double, 3> helper =
      std::abs(n[0]) < 0.9 ? std::array<double, 3>{1.0, 0.0, 0.0}
                           : std::array<double, 3>{0.0, 1.0, 0.0};
  const double hdotn = helper[0] * n[0] + helper[1] * n[1] + helper[2] * n[2];
  std::array<double, 3> a{helper[0] - hdotn * n[0], helper[1] - hdotn * n[1],
                          helper[2] - hdotn * n[2]};
  const double inv = 1.0 / std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  *t1 = {a[0] * inv, a[1] * inv, a[2] * inv};
  // t2 = n x t1
  *t2 = {n[1] * (*t1)[2] - n[2] * (*t1)[1], n[2] * (*t1)[0] - n[0] * (*t1)[2],
         n[0] * (*t1)[1] - n[1] * (*t1)[0]};
}
}  // namespace

PlutoHlld::PlutoHlld(const double magnetic_field_magnitude_for_hydro,
                     const double light_speed_density_cutoff)
    : magnetic_field_magnitude_for_hydro_(magnetic_field_magnitude_for_hydro),
      light_speed_density_cutoff_(light_speed_density_cutoff) {}

PlutoHlld::PlutoHlld(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> PlutoHlld::get_clone() const {
  return std::make_unique<PlutoHlld>(*this);
}

void PlutoHlld::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | magnetic_field_magnitude_for_hydro_;
  p | light_speed_density_cutoff_;
}

double PlutoHlld::dg_package_data(
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
    const gsl::not_null<Scalar<DataVector>*> packaged_interface_lapse,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_interface_shift_dot_normal,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        packaged_interface_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> packaged_rest_mass_density,
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
    get(*packaged_largest_outgoing_char_speed) =
        get(lapse) - get(shift_dot_normal);
    get(*packaged_largest_ingoing_char_speed) =
        -get(lapse) - get(shift_dot_normal);

    if (const bool has_b_field =
            max(get(magnitude(tilde_b))) > magnetic_field_magnitude_for_hydro_;
        not has_b_field and
        (max(get(rest_mass_density)) > light_speed_density_cutoff_)) {
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
      const Scalar<DataVector> sound_speed_squared{clamp(
          get(equation_of_state
                  .sound_speed_squared_from_density_and_temperature(
                      rest_mass_density, temperature,
                      Scalar<DataVector>{get(rest_mass_density).size(), 0.0})),
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
    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_largest_outgoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
      get(*packaged_largest_ingoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
    }
  }

  *packaged_tilde_d = tilde_d;
  *packaged_tilde_ye = tilde_ye;
  *packaged_tilde_tau = tilde_tau;
  *packaged_tilde_s = tilde_s;
  *packaged_tilde_b = tilde_b;
  *packaged_tilde_phi = tilde_phi;
  *packaged_interface_unit_normal = normal_covector;
  // Flatness indicator: the five-wave fan below assumes flat space, so measure
  // the metric's departure from Minkowski using the (analytic,un-reconstructed)
  // background lapse and shift -- zero for the Minkowski backgrounds of the
  // relativistic M&M tests, nonzero otherwise (where we fall back to HLL).
  // We deliberately do NOT use sqrt(det gamma)=TildeD/(rho W): at reconstructed
  // FD faces TildeD and rho,W are reconstructed independently so that ratio is
  // not 1 even in flat space. Since the GR face frame (PlutoFaceFrame.hpp) the
  // spatial metric and the mesh motion count too: both add exact zeros on a
  // static Minkowski face, so the flat path is selected exactly as before,
  // and every other face is solved in the face frame.
  get(*packaged_metric_flatness) = abs(get(lapse) - 1.0) + abs(get<0>(shift)) +
                                   abs(get<1>(shift)) + abs(get<2>(shift));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      get(*packaged_metric_flatness) +=
          abs(spatial_metric.get(i, j) - (i == j ? 1.0 : 0.0));
    }
  }
  if (normal_dot_mesh_velocity.has_value()) {
    get(*packaged_metric_flatness) += abs(get(*normal_dot_mesh_velocity));
  }
  // This side's geometry for the face frame. The normal shift carries the
  // mesh velocity: the face moves through the normal observer's frame at
  // (beta^n + n.v_g)/alpha, because the fluxes reaching dg_package_data are
  // already shifted by -v_g U.
  *packaged_interface_lapse = lapse;
  dot_product(packaged_interface_shift_dot_normal, shift, normal_covector);
  if (normal_dot_mesh_velocity.has_value()) {
    get(*packaged_interface_shift_dot_normal) += get(*normal_dot_mesh_velocity);
  }
  *packaged_interface_spatial_metric = spatial_metric;
  *packaged_rest_mass_density = rest_mass_density;
  *packaged_spatial_velocity = spatial_velocity;
  *packaged_pressure = pressure;
  *packaged_lorentz_factor = lorentz_factor;
  *packaged_specific_internal_energy = specific_internal_energy;

  // --- Fast-magnetosonic signal speeds for the MHD part, PER SIDE -----------
  //
  // In flat space the divergence-cleaning (Phi, B_n) subsystem decouples and
  // travels at the light speed -- that is what the Largest speeds above are --
  // while the MHD variables travel at the slower fast-magnetosonic speed.
  // THIS side's fast speeds are packaged here and `dg_boundary_terms` combines
  // them with the neighbour's by the standard two-sided min/max, exactly as
  // `Hll` already does.
  //
  // What this replaces, and why. Until now the fast bounds were built by
  // evaluating `characteristic_speeds_mhd` ONCE, at the arithmetic average of
  // the two face states, with rho, eps and p averaged INDEPENDENTLY. That
  // triple satisfies no equation of state: c_s^2 is formed as
  // (chi + kappa p/rho^2)/h with chi and kappa at (rho_avg, eps_avg) but h
  // built from p_avg, which for an ideal fluid tends to Gamma (Gamma - 1) as
  // p_avg/rho_avg -> 0 -- above 1 for every Gamma past the golden ratio
  // 1.618. A superluminal c_s^2 pushes the magnetosonic quartic's extremal
  // roots outside the light cone, which voids the precondition of the +/-1
  // Newton seeds in `find_magnetosonic_speed_from_quartic`, so both seeds can
  // land on the SAME root and the deflation by the "fast pair" is no longer a
  // factorization. That is what aborted a Del Zanna et al. (2003) jet run with
  // a reduced-quadratic discriminant of -7.8677, a value outside
  // the [-4, +4] band the quantity can occupy at all when the deflation is
  // valid. The clamp in `characteristic_speeds_mhd` stops the abort; taking
  // the bound per side is the cure, because each side IS a state (the FD
  // reconstruction reconstructs rho and T and derives eps and p from the
  // equation of state), so c_s^2 < 1 structurally and the failure mode cannot
  // occur.
  //
  // It also moves us TOWARD the reference rather than away from it. No
  // reference code builds a signal speed at an averaged interface state:
  // PLUTO's HLL_Speed calls MaxSignalSpeed(stateL) and MaxSignalSpeed(stateR)
  // and then MIN/MAX, under the comment "DAVIS Estimate"
  // (Src/RMHD/hll_speed.c:44-58); AthenaK does the same
  // (hlle_srmhd.hpp:95-96, Mignone & Bodo Eq. 55), as do Fields, Wong & Stone
  // (their Eq. 8) and Del Zanna, Bucciantini & Londrillo (their Eq. 43). And
  // the averaged state never reached PLUTO's own HLLD in any case: the
  // wrapped pluto/hlld.c opens with its own two-sided HLL_Speed call, so these
  // bounds feed only the HLL baseline layer and the GLM/MHD split built around
  // it here. The old in-code justification for averaging -- "a single
  // interface eigensystem, so the bounds are the same on mirror-image faces"
  // -- did not require it either: min/max over the two sides is equally
  // symmetric under swapping interior and exterior.
  //
  // The speed itself is the direction-independent estimate
  //   a^2 = c_s^2 + v_A^2 (1 - c_s^2)
  // pushed through the relativistic dispersion relation, i.e. exactly what
  // `characteristic_speeds_approximate_mhd` computes and what `Hll` calls.
  // Dropping the B.n dependence makes it an UPPER bound on the true fast
  // speed, so it is a safe (slightly dissipative) HLL bound rather than the
  // exact quartic root; that bias is shared with AthenaK and is far smaller
  // than the averaged-state defect it replaces. It is also strictly less work
  // than before: this solver only ever read the fast pair out of
  // `characteristic_speeds_mhd` (indices 1 and 7) and discarded the slow
  // roots, so the entire quartic solve was dead work.
  *packaged_fast_outgoing_char_speed = *packaged_largest_outgoing_char_speed;
  *packaged_fast_ingoing_char_speed = *packaged_largest_ingoing_char_speed;
  // A flat face keeps exactly the computation it always had (the identity
  // metric, TildeB = B); any other face computes the same per-side speeds
  // with its own metric, below. A face with flat and curved points takes the
  // second branch, which gives its flat points the same speeds.
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
    // In flat space TildeB = sqrt(gamma) B^i = B^i.
    std::array<DataVector, 9> fast_speeds{};
    characteristic_speeds_approximate_mhd(
        make_not_null(&fast_speeds), rest_mass_density, electron_fraction,
        specific_internal_energy, specific_enthalpy, spatial_velocity,
        lorentz_factor, tilde_b, lapse, shift, flat_metric, normal_covector,
        equation_of_state);
    // Indices 1 and 7 are the ingoing and outgoing fast-magnetosonic speeds.
    get(*packaged_fast_outgoing_char_speed) = fast_speeds[7];
    get(*packaged_fast_ingoing_char_speed) = fast_speeds[1];
    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_fast_outgoing_char_speed) -= get(*normal_dot_mesh_velocity);
      get(*packaged_fast_ingoing_char_speed) -= get(*normal_dot_mesh_velocity);
    }
    // In the atmosphere keep the light speed, exactly as the Largest speeds do
    // (they already carry the mesh-velocity correction).
    for (size_t pt = 0; pt < num_points; ++pt) {
      if (get(rest_mass_density)[pt] <= light_speed_density_cutoff_) {
        get(*packaged_fast_outgoing_char_speed)[pt] =
            get(*packaged_largest_outgoing_char_speed)[pt];
        get(*packaged_fast_ingoing_char_speed)[pt] =
            get(*packaged_largest_ingoing_char_speed)[pt];
      }
    }
  } else {
    // The GR face frame: the same per-side fast-magnetosonic speeds with this
    // side's metric, lapse and shift (characteristic_speeds_approximate_mhd
    // returns alpha lambda_hat - beta^n) and the undensitized field
    // B^i = TildeB^i / sqrt(gamma). They serve the MHD part of the HLL
    // baseline only -- D, Ye, Tau, S and the tangential field, as on a flat
    // face; the GLM pair (Phi, B_n) keeps the Largest (light-cone) speeds.
    // Points where they come out non-finite or inverted (incoming above
    // outgoing, which inadmissible states such as v^2 c_s^2 > 1 produce) keep
    // the Largest speeds: with inverted bounds both clamped HLL bounds can
    // vanish, and the clamped formula is then not weak/strong consistent.
    const ScopedFpeState fpe(false);
    const size_t num_points = get(rest_mass_density).size();
    const Scalar<DataVector> specific_enthalpy =
        hydro::relativistic_specific_enthalpy(
            rest_mass_density, specific_internal_energy, pressure);
    const DataVector sqrt_det_spatial_metric =
        sqrt(get(determinant(spatial_metric)));
    tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field = tilde_b;
    for (size_t i = 0; i < 3; ++i) {
      magnetic_field.get(i) /= sqrt_det_spatial_metric;
    }
    std::array<DataVector, 9> fast_speeds{};
    characteristic_speeds_approximate_mhd(
        make_not_null(&fast_speeds), rest_mass_density, electron_fraction,
        specific_internal_energy, specific_enthalpy, spatial_velocity,
        lorentz_factor, magnetic_field, lapse, shift, spatial_metric,
        normal_covector, equation_of_state);
    for (size_t pt = 0; pt < num_points; ++pt) {
      double outgoing = fast_speeds[7][pt];
      double ingoing = fast_speeds[1][pt];
      if (normal_dot_mesh_velocity.has_value()) {
        outgoing -= get(*normal_dot_mesh_velocity)[pt];
        ingoing -= get(*normal_dot_mesh_velocity)[pt];
      }
      if (get(rest_mass_density)[pt] > light_speed_density_cutoff_ and
          std::isfinite(outgoing) and std::isfinite(ingoing) and
          ingoing <= outgoing) {
        get(*packaged_fast_outgoing_char_speed)[pt] = outgoing;
        get(*packaged_fast_ingoing_char_speed)[pt] = ingoing;
      }
    }
  }

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

namespace detail {
void pluto_dg_boundary_terms(
    const PlutoFluxFunction pluto_solver,
    const PlutoMovingFaceFluxFunction moving_face_solver,
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
    const Scalar<DataVector>& interface_lapse_int,
    const Scalar<DataVector>& interface_shift_dot_normal_int,
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        interface_spatial_metric_int,
    const Scalar<DataVector>& rest_mass_density_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_int,
    const Scalar<DataVector>& pressure_int,
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
    const Scalar<DataVector>& fast_outgoing_char_speed_ext,
    const Scalar<DataVector>& fast_ingoing_char_speed_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& iface_normal_ext,
    const Scalar<DataVector>& metric_flatness_ext,
    const Scalar<DataVector>& interface_lapse_ext,
    const Scalar<DataVector>& interface_shift_dot_normal_ext,
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        interface_spatial_metric_ext,
    const Scalar<DataVector>& rest_mass_density_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_ext,
    const Scalar<DataVector>& pressure_ext,
    const Scalar<DataVector>& /*lorentz_factor_ext*/,
    const Scalar<DataVector>& specific_internal_energy_ext,
    const dg::Formulation dg_formulation,
    const double light_speed_density_cutoff,
    const bool force_common_normal_field, const bool force_glm_energy_pair,
    const bool force_mass_consistent_ye) {
  const size_t num_points = get(tilde_d_int).size();
  const bool weak = dg_formulation == dg::Formulation::WeakInertial;

  // --- HLL baseline (used for TildeYe, TildePhi, and as a fallback) ---
  //
  // SCALAR / MHD SPLIT (the same one HLL and HLLEM use; Teukolsky MHD Eq.4.35).
  // In flat space the GLM subsystem decouples: Phi and the NORMAL mag field
  // propagate at the light speed, everything else (D, Ye, Tau, S, and the
  // TANGENTIAL magnetic field) at the fast-magnetosonic speed. Using +/-c for
  // the MHD variables makes this baseline far more dissipative than it needs to
  // be, so the HLLD *logic* is applied to the MHD part and the GLM part is
  // carried separately at +/-c.
  //
  // Note this baseline is not merely cosmetic for HLLD: it supplies TildePhi
  // outright, seeds TildeYe/D/S/B/Tau, and is what a point falls back to if the
  // fan solve returns something non-finite. Legacy callers retain this TildeYe
  // flux; D2-R requests passive-scalar transport by the accepted mass flux.
  DataVector lambda_max = max(0.0, get(largest_outgoing_char_speed_int),
                              -get(largest_ingoing_char_speed_ext));
  DataVector lambda_min = min(0.0, get(largest_ingoing_char_speed_int),
                              -get(largest_outgoing_char_speed_ext));

  // Fast-magnetosonic bounds for the MHD variables (D, Ye, Tau, S and the
  // tangential magnetic field), which travel slower than light.
  //
  // Built from the two sides' OWN fast speeds -- packaged per side in
  // dg_package_data, see the long comment there -- by the same min/max as the
  // light bounds above. This is the standard (Davis) construction and the one
  // PLUTO, AthenaK, Del Zanna et al. and Fields, Wong & Stone all use; it
  // replaces an eigensystem evaluated once at an arithmetic-averaged interface
  // state whose (rho, eps, p) triple satisfied no equation of state and could
  // be superluminal.
  //
  // The minus signs on the exterior speeds are there for the same reason as in
  // the light bounds: the neighbour packaged its speeds against its own
  // outward normal, which is -n. Written this way the pair satisfies
  // fast_max(ext, int) = -fast_min and fast_min(ext, int) = -fast_max, i.e.
  // the flux is antisymmetric under the int<->ext swap, which is what
  // conservation across the face requires -- the same symmetry the averaged
  // state was introduced to get, obtained without manufacturing a non-state.
  //
  // Where the background is curved, or in the atmosphere, dg_package_data left
  // these equal to the light speeds, so the scheme is the standard HLL flux
  // there.
  const DataVector fast_max = max(0.0, get(fast_outgoing_char_speed_int),
                                  -get(fast_ingoing_char_speed_ext));
  const DataVector fast_min = min(0.0, get(fast_ingoing_char_speed_int),
                                  -get(fast_outgoing_char_speed_ext));
  // Each point takes one of three paths. FLAT (both sides' flatness within
  // 1e-12): the special-relativistic fan exactly as before the GR face frame.
  // CURVED (finite flatness): the fan solved in the face frame
  // (PlutoFaceFrame.hpp) and sampled on the ray of the moving face; where the
  // frame cannot be built the point keeps the HLL baseline. UNSUPPORTED
  // (infinite flatness, PlutoD2R's gate): the HLL baseline only.
  enum class PointPath : char { Flat, Curved, Unsupported };
  std::vector<PointPath> path(num_points, PointPath::Flat);
  std::vector<std::optional<FaceFrame>> frames(num_points);
  FaceFrameCounts tally{};
  tally.calls = 1;
  // An experiment, off by default: honour LightSpeedDensityCutoff in the PLUTO
  // path as Hll and HllcGr do in an atmosphere (their bounds there are the
  // light speeds), by keeping the HLL flux wherever a side's density is at or
  // below it. Read once.
  static const bool honor_density_cutoff = [] {
    const char* e = std::getenv("SPECTRE_PLUTO_DENSITY_CUTOFF_HLL");
    return e != nullptr and std::atoi(e) != 0;
  }();
  const auto below_density_cutoff = [&](const size_t pt) {
    return honor_density_cutoff and
           (get(rest_mass_density_int)[pt] <= light_speed_density_cutoff or
            get(rest_mass_density_ext)[pt] <= light_speed_density_cutoff);
  };
  const auto symmetric_components =
      [](const tnsr::ii<DataVector, 3, Frame::Inertial>& metric,
         const size_t pt) {
        return std::array<double, 6>{
            {get<0, 0>(metric)[pt], get<0, 1>(metric)[pt],
             get<0, 2>(metric)[pt], get<1, 1>(metric)[pt],
             get<1, 2>(metric)[pt], get<2, 2>(metric)[pt]}};
      };
  const auto covector_at = [](const tnsr::i<DataVector, 3, Frame::Inertial>& v,
                              const size_t pt) {
    return std::array<double, 3>{{get<0>(v)[pt], get<1>(v)[pt], get<2>(v)[pt]}};
  };
  for (size_t pt = 0; pt < num_points; ++pt) {
    const double flatness_int = get(metric_flatness_int)[pt];
    const double flatness_ext = get(metric_flatness_ext)[pt];
    if (flatness_int <= 1.0e-12 and flatness_ext <= 1.0e-12) {
      ++tally.flat_points;
    } else if (not(std::isfinite(flatness_int) and
                   std::isfinite(flatness_ext))) {
      path[pt] = PointPath::Unsupported;
      ++tally.unsupported_points;
    } else {
      path[pt] = PointPath::Curved;
      ++tally.curved_points;
      frames[pt] = make_face_frame(
          get(interface_lapse_int)[pt], get(interface_lapse_ext)[pt],
          get(interface_shift_dot_normal_int)[pt],
          get(interface_shift_dot_normal_ext)[pt],
          symmetric_components(interface_spatial_metric_int, pt),
          symmetric_components(interface_spatial_metric_ext, pt),
          covector_at(interface_unit_normal_int, pt),
          covector_at(iface_normal_ext, pt));
      if (not frames[pt].has_value()) {
        ++tally.fallback_frame;
      }
    }
  }

  // HLL flux for one component given explicit bounds.
  const auto hll_with = [&weak, &num_points](
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
  // MHD variables -> fast bounds; kept as a lambda with old signature so the
  // call sites below are unchanged.
  auto hll = [&](const Scalar<DataVector>& nf_int,
                 const Scalar<DataVector>& u_int,
                 const Scalar<DataVector>& nf_ext,
                 const Scalar<DataVector>& u_ext) -> DataVector {
    return hll_with(fast_max, fast_min, get(u_int), get(nf_int), get(u_ext),
                    get(nf_ext));
  };
  get(*boundary_correction_tilde_ye) =
      hll(normal_dot_flux_tilde_ye_int, tilde_ye_int,
          normal_dot_flux_tilde_ye_ext, tilde_ye_ext);
  // Divergence-cleaning scalar Phi belongs to the GLM subsystem: light speed.
  get(*boundary_correction_tilde_phi) =
      hll_with(lambda_max, lambda_min, get(tilde_phi_int),
               get(normal_dot_flux_tilde_phi_int), get(tilde_phi_ext),
               get(normal_dot_flux_tilde_phi_ext));
  // HLLD overrides TildeD/S/B/Tau below; seed with HLL so any skipped point is
  // still filled.
  get(*boundary_correction_tilde_d) =
      hll(normal_dot_flux_tilde_d_int, tilde_d_int, normal_dot_flux_tilde_d_ext,
          tilde_d_ext);
  get(*boundary_correction_tilde_tau) =
      hll(normal_dot_flux_tilde_tau_int, tilde_tau_int,
          normal_dot_flux_tilde_tau_ext, tilde_tau_ext);
  // Momentum: pure MHD -> fast bounds.
  for (size_t k = 0; k < 3; ++k) {
    boundary_correction_tilde_s->get(k) =
        hll_with(fast_max, fast_min, tilde_s_int.get(k),
                 normal_dot_flux_tilde_s_int.get(k), tilde_s_ext.get(k),
                 normal_dot_flux_tilde_s_ext.get(k));
  }
  // Raw (weak-form) GLM flux for the normal field, filled by the block below
  // and reused by the PLUTO scatter.
  DataVector glm_normal_b_flux{num_points, 0.0};
  // The GLM face STATE of the normal field (the HLL intermediate state), kept for the
  // energy pairing below.
  DataVector glm_normal_b_star{num_points, 0.0};
  // Magnetic field: the NORMAL component belongs to the GLM subsystem (light
  // speed), the TANGENTIAL component is MHD (fast bounds). Split along the
  // interface normal, use the appropriate bounds for each part, and recombine
  // G(B^i) = G(B_n) n^i + G(B_t^i). The decomposition below treats n as both
  // covector and raised vector, which holds only in flat space; curved points
  // with a face frame are redone after this block with the frame's normal
  // covector (projection) and vector (recombination), and the other non-flat
  // points keep the plain light-speed HLL flux (which stays conservative).
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
        hll_with(lambda_max, lambda_min, bn_int, nfbn_int, bn_ext, nfbn_ext);
    // The PLUTO scatter below needs this as a RAW flux G (it applies its own
    // strong-form subtraction), so recompute it in weak form regardless of
    // `weak`.  See the comment at the scatter for why PLUTO cannot supply it.
    {
      DataVector dl = lambda_max - lambda_min;
      for (size_t pt = 0; pt < num_points; ++pt) {
        if (dl[pt] < 1.0e-30) {
          dl[pt] = 1.0e-30;
        }
      }
      glm_normal_b_flux = (lambda_max * nfbn_int + lambda_min * nfbn_ext +
                           lambda_max * lambda_min * (bn_ext - bn_int)) /
                          dl;
      // The HLL intermediate state is U* = (SR UR - SL UL + F_L - F_R)/(SR - SL)
      // with both fluxes along the INTERIOR normal. The packaged exterior
      // normal flux carries the exterior normal, nfbn_ext = -F_R . n_int (that
      // is why the HLL flux above takes lambda_min * nfbn_ext), so the state
      // needs the SUM nfbn_int + nfbn_ext. In flat space F_Bn . n = Phi, so this
      // is PLUTO's GLM_Solve value Bm = (Bn_L + Bn_R)/2 - (Phi_R - Phi_L)/(2 c_h)
      // for c_h = 1. Until 2026-10-15 this read nfbn_int - nfbn_ext, i.e. the
      // sign of Phi_R was wrong (found by the task-3 session's unequal-Bn,
      // nonzero-Phi test; the error is 2 Phi_R/(SR - SL), zero wherever Phi = 0,
      // so every 1-D tube is unaffected and only 2-D GLM runs since 72620d6e2
      // saw it). The same expression now serves every PLUTO-backed solver;
      // `force_common_normal_field` only forces its use below.
      glm_normal_b_star = (lambda_max * bn_ext - lambda_min * bn_int +
                           nfbn_int + nfbn_ext) /
                          dl;
    }
    for (size_t i = 0; i < 3; ++i) {
      const DataVector bt_int = tilde_b_int.get(i) - bn_int * n.get(i);
      const DataVector bt_ext = tilde_b_ext.get(i) - bn_ext * n.get(i);
      const DataVector nfbt_int =
          normal_dot_flux_tilde_b_int.get(i) - nfbn_int * n.get(i);
      const DataVector nfbt_ext =
          normal_dot_flux_tilde_b_ext.get(i) - nfbn_ext * n.get(i);
      const DataVector g_bt =
          hll_with(fast_max, fast_min, bt_int, nfbt_int, bt_ext, nfbt_ext);
      const DataVector g_split = g_bn * n.get(i) + g_bt;
      const DataVector g_plain =
          hll_with(lambda_max, lambda_min, tilde_b_int.get(i),
                   normal_dot_flux_tilde_b_int.get(i), tilde_b_ext.get(i),
                   normal_dot_flux_tilde_b_ext.get(i));
      for (size_t pt = 0; pt < num_points; ++pt) {
        boundary_correction_tilde_b->get(i)[pt] =
            path[pt] == PointPath::Flat ? g_split[pt] : g_plain[pt];
      }
    }
  }

  // The GLM pair and the field split at the curved points with a face frame.
  // The pair (Phi, B_n) keeps the light-cone bounds -- the Largest speeds,
  // +/-alpha - beta^n - n.v_g, as on a flat face -- and the tangential field
  // the fast bounds. In coordinates the HLL state is the normal observer's
  // HLL state and the HLL flux maps by F = alpha F_hat - beta^n U, so this
  // coordinate-frame HLL is the light-speed fan sampled on the ray of the
  // moving face. Two quantities need more care than on a flat face:
  //  * the common normal field handed to the MHD solver is the GLM fan's state
  //    ON THE FACE'S RAY: its MIDDLE state (bounds not clamped to include 0)
  //    while the face is inside the light cone -- the MHD fan lies inside it,
  //    where B_n = B_n* -- and the upwind side's own B_n outside it
  //    (|w| >= 1, inside a horizon), where the face takes the upwind state of
  //    the whole fan and the MHD solver must see that state unchanged. (On a
  //    flat face the light bounds are never clamped and the face is always
  //    inside the cone, so this is the flat path's middle state.)
  //  * the cleaning part of the normal-field flux, alpha Phi_sel (densitized),
  //    for the energy pairing: the normal-field flux is
  //    sqrt(gamma) (alpha Phi - beta^n B_n), and its -beta^n B_n part is the
  //    common translation every variable gets, not cleaning.
  DataVector glm_phi_sel{num_points, 0.0};
  for (size_t pt = 0; pt < num_points; ++pt) {
    if (not frames[pt].has_value()) {
      continue;
    }
    const FaceFrame& frame = *frames[pt];
    const auto& n_lo = frame.duals[0];
    const auto& n_up = frame.legs[0];
    double bn_i = 0.0, bn_e = 0.0, nfbn_i = 0.0, nfbn_e = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      bn_i += n_lo[i] * tilde_b_int.get(i)[pt];
      bn_e += n_lo[i] * tilde_b_ext.get(i)[pt];
      nfbn_i += n_lo[i] * normal_dot_flux_tilde_b_int.get(i)[pt];
      nfbn_e += n_lo[i] * normal_dot_flux_tilde_b_ext.get(i)[pt];
    }
    const double l_max = lambda_max[pt];
    const double l_min = lambda_min[pt];
    const double dl = std::max(l_max - l_min, 1.0e-30);
    const auto hll_point = [&dl, &weak](const double lp, const double lm,
                                        const double u_i, const double nf_i,
                                        const double u_e, const double nf_e) {
      if (weak) {
        return (lp * nf_i + lm * nf_e + lp * lm * (u_e - u_i)) / dl;
      }
      return (lm * (nf_i + nf_e) + lp * lm * (u_e - u_i)) / dl;
    };
    const double g_bn = hll_point(l_max, l_min, bn_i, nfbn_i, bn_e, nfbn_e);
    glm_normal_b_flux[pt] =
        (l_max * nfbn_i + l_min * nfbn_e + l_max * l_min * (bn_e - bn_i)) / dl;
    // The unclamped light cone and the GLM fan's states on it.
    const double cone_max = std::max(get(largest_outgoing_char_speed_int)[pt],
                                     -get(largest_ingoing_char_speed_ext)[pt]);
    const double cone_min = std::min(get(largest_ingoing_char_speed_int)[pt],
                                     -get(largest_outgoing_char_speed_ext)[pt]);
    const double cone_width = std::max(cone_max - cone_min, 1.0e-30);
    const double bn_middle =
        (cone_max * bn_e - cone_min * bn_i + nfbn_i + nfbn_e) / cone_width;
    glm_normal_b_star[pt] = cone_min >= 0.0   ? bn_i
                            : cone_max <= 0.0 ? bn_e
                                              : bn_middle;
    const double phi_star =
        (cone_max * get(tilde_phi_ext)[pt] - cone_min * get(tilde_phi_int)[pt] +
         get(normal_dot_flux_tilde_phi_int)[pt] +
         get(normal_dot_flux_tilde_phi_ext)[pt]) /
        cone_width;
    const double phi_on_ray = cone_min >= 0.0   ? get(tilde_phi_int)[pt]
                              : cone_max <= 0.0 ? get(tilde_phi_ext)[pt]
                                                : phi_star;
    glm_phi_sel[pt] = frame.lapse * phi_on_ray;
    for (size_t i = 0; i < 3; ++i) {
      const double bt_i = tilde_b_int.get(i)[pt] - bn_i * n_up[i];
      const double bt_e = tilde_b_ext.get(i)[pt] - bn_e * n_up[i];
      const double nfbt_i =
          normal_dot_flux_tilde_b_int.get(i)[pt] - nfbn_i * n_up[i];
      const double nfbt_e =
          normal_dot_flux_tilde_b_ext.get(i)[pt] - nfbn_e * n_up[i];
      const double dl_fast = std::max(fast_max[pt] - fast_min[pt], 1.0e-30);
      const double g_bt =
          (weak ? (fast_max[pt] * nfbt_i + fast_min[pt] * nfbt_e +
                   fast_max[pt] * fast_min[pt] * (bt_e - bt_i))
                : (fast_min[pt] * (nfbt_i + nfbt_e) +
                   fast_max[pt] * fast_min[pt] * (bt_e - bt_i))) /
          dl_fast;
      boundary_correction_tilde_b->get(i)[pt] = g_bn * n_up[i] + g_bt;
    }
  }

  // --- The curved points: the fan in the face frame --------------------------
  // (HLLx campaign, gr_tetrad_adapter.md.) The states are the packaged
  // primitives in the frame's triad, undensitized, with the GLM state on the
  // face's ray as the common normal field. The solver samples its fan on the
  // ray of the moving face, w = frame.face_speed, and returns the flux through
  // the moving face in the triad, f, with the normal total pressure P apart;
  // then
  //   G(D)   = alpha sqrt(gamma) f[RHO],   G(Tau) = alpha sqrt(gamma) f[ENG],
  //   G(S_j) = alpha sqrt(gamma) [(f[MX1] + P) n_j + f[MX2] e^(1)_j
  //                               + f[MX3] e^(2)_j]                (covectors)
  //   G(B^j) = G_GLM(B_n) n^j
  //            + alpha sqrt(gamma) [f[BX2] e_(1)^j + f[BX3] e_(2)^j] (vectors).
  // Everything here runs before the flat batch so that the latter keeps its
  // early return exactly as it always had.
  {
    const ScopedFpeState curved_fpe_scope(false);
    std::vector<size_t> curved_index;
    std::vector<double> curved_vl, curved_vr, curved_speed, curved_gamma;
    for (size_t pt = 0; pt < num_points; ++pt) {
      if (not frames[pt].has_value()) {
        continue;
      }
      const FaceFrame& frame = *frames[pt];
      // Both sides are checked, and the adiabatic index is the two sides'
      // mean, so that swapping interior and exterior gives the mirror image
      // of the same problem (conservation across the face). On a flat face
      // the interior's index is used as before; for an ideal gas the two
      // agree to round-off at FD faces.
      const double rho_i = get(rest_mass_density_int)[pt];
      const double eps_i = get(specific_internal_energy_int)[pt];
      const double p_i = get(pressure_int)[pt];
      const double rho_e = get(rest_mass_density_ext)[pt];
      const double eps_e = get(specific_internal_energy_ext)[pt];
      const double p_e = get(pressure_ext)[pt];
      if (not(rho_i > 0.0 and eps_i > 0.0 and p_i > 0.0 and rho_e > 0.0 and
              eps_e > 0.0 and p_e > 0.0)) {
        ++tally.fallback_state;
        continue;  // keep HLL where the ideal-gas relation is undefined
      }
      if (below_density_cutoff(pt)) {
        ++tally.density_cutoff;
        continue;  // the experiment: keep the light-speed HLL flux
      }
      if (moving_face_solver == nullptr and frame.face_speed != 0.0) {
        ++tally.fallback_ray;
        continue;  // a ray-0 kernel cannot sample a moving face
      }
      const double inv_sqrt_det = 1.0 / frame.sqrt_det_spatial_metric;
      const double common_normal_field = glm_normal_b_star[pt] * inv_sqrt_det;
      const auto push = [&](std::vector<double>* out,
                            const Scalar<DataVector>& rho,
                            const tnsr::I<DataVector, 3, Frame::Inertial>& vel,
                            const tnsr::I<DataVector, 3, Frame::Inertial>& bt,
                            const Scalar<DataVector>& press) {
        const auto v = frame.components(
            {{get<0>(vel)[pt], get<1>(vel)[pt], get<2>(vel)[pt]}});
        const auto b = frame.components(
            {{get<0>(bt)[pt] * inv_sqrt_det, get<1>(bt)[pt] * inv_sqrt_det,
              get<2>(bt)[pt] * inv_sqrt_det}});
        out->push_back(get(rho)[pt]);
        out->push_back(v[0]);
        out->push_back(v[1]);
        out->push_back(v[2]);
        out->push_back(common_normal_field);
        out->push_back(b[1]);
        out->push_back(b[2]);
        out->push_back(get(press)[pt]);
      };
      push(&curved_vl, rest_mass_density_int, spatial_velocity_int, tilde_b_int,
           pressure_int);
      push(&curved_vr, rest_mass_density_ext, spatial_velocity_ext, tilde_b_ext,
           pressure_ext);
      curved_index.push_back(pt);
      curved_speed.push_back(frame.face_speed);
      curved_gamma.push_back(
          1.0 + 0.5 * (p_i / (rho_i * eps_i) + p_e / (rho_e * eps_e)));
    }

    if (not curved_index.empty()) {
      const size_t n_solve = curved_index.size();
      std::vector<double> flux(8 * n_solve, 0.0);
      std::vector<double> press(n_solve, 0.0);
      const double gamma_min =
          *std::min_element(curved_gamma.begin(), curved_gamma.end());
      const double gamma_max =
          *std::max_element(curved_gamma.begin(), curved_gamma.end());
      const bool uniform_gamma =
          gamma_max - gamma_min <= 1.0e-12 * std::max(1.0, gamma_max);
      const auto solve = [&](const size_t offset, const size_t count,
                             const double gamma) {
        if (moving_face_solver != nullptr) {
          moving_face_solver(static_cast<int>(count),
                             curved_vl.data() + 8 * offset,
                             curved_vr.data() + 8 * offset, gamma,
                             curved_speed.data() + offset,
                             flux.data() + 8 * offset, press.data() + offset);
        } else {
          pluto_solver(static_cast<int>(count), curved_vl.data() + 8 * offset,
                       curved_vr.data() + 8 * offset, gamma,
                       flux.data() + 8 * offset, press.data() + offset);
        }
      };
      if (uniform_gamma) {
        solve(0, n_solve, gamma_min);
      } else {
        for (size_t k = 0; k < n_solve; ++k) {
          solve(k, 1, curved_gamma[k]);
        }
      }

      for (size_t k = 0; k < n_solve; ++k) {
        const double* f = flux.data() + 8 * k;
        bool all_finite = std::isfinite(press[k]);
        for (size_t q = 0; q < 8; ++q) {
          all_finite = all_finite and std::isfinite(f[q]);
        }
        if (not all_finite) {
          ++tally.fallback_solver;
          continue;  // keep HLL if PLUTO returned a non-finite result
        }
        const size_t pt = curved_index[k];
        const FaceFrame& frame = *frames[pt];
        const double scale = frame.lapse * frame.sqrt_det_spatial_metric;
        const double g_d = scale * f[0];
        double g_ye = 0.0;
        if (force_mass_consistent_ye) {
          // As on a flat face: the upwind concentration with the accepted mass
          // flux, upwind by the sign of the coordinate mass flux.
          const double upwind_d =
              g_d >= 0.0 ? get(tilde_d_int)[pt] : get(tilde_d_ext)[pt];
          const double upwind_dye =
              g_d >= 0.0 ? get(tilde_ye_int)[pt] : get(tilde_ye_ext)[pt];
          if (not std::isfinite(upwind_d) or upwind_d <= 0.0 or
              not std::isfinite(upwind_dye)) {
            ++tally.fallback_solver;
            continue;  // retain the complete HLL baseline
          }
          g_ye = upwind_dye / upwind_d * g_d;
          if (not std::isfinite(g_ye)) {
            ++tally.fallback_solver;
            continue;
          }
        }
        const auto g_s = frame.covector(
            {{scale * (f[1] + press[k]), scale * f[2], scale * f[3]}});
        const auto g_bt = frame.vector({{0.0, scale * f[5], scale * f[6]}});
        const double g_tau = scale * f[7];
        get(*boundary_correction_tilde_d)[pt] =
            g_d - (weak ? 0.0 : get(normal_dot_flux_tilde_d_int)[pt]);
        if (force_mass_consistent_ye) {
          get(*boundary_correction_tilde_ye)[pt] =
              g_ye - (weak ? 0.0 : get(normal_dot_flux_tilde_ye_int)[pt]);
        }
        get(*boundary_correction_tilde_tau)[pt] =
            g_tau - (weak ? 0.0 : get(normal_dot_flux_tilde_tau_int)[pt]);
        for (size_t q = 0; q < 3; ++q) {
          boundary_correction_tilde_s->get(q)[pt] =
              g_s[q] - (weak ? 0.0 : normal_dot_flux_tilde_s_int.get(q)[pt]);
          boundary_correction_tilde_b->get(q)[pt] =
              glm_normal_b_flux[pt] * frame.legs[0][q] + g_bt[q] -
              (weak ? 0.0 : normal_dot_flux_tilde_b_int.get(q)[pt]);
        }
        ++tally.curved_solved;
        if (std::abs(frame.face_speed) >= 1.0) {
          ++tally.outside_light_cone;
        }
      }
    }

    // The GLM energy pairing in GR form (same switches as on a flat face):
    //   G(Tau) += alpha Phi~_sel [(1 - v^2) B_n* + (v.B) v^n],
    // alpha Phi~_sel = glm_phi_sel (the cleaning part of the normal-field
    // flux), B_n* = glm_normal_b_star / sqrt(gamma), and v, B the two sides'
    // means in the triad. With alpha = sqrt(gamma) = 1 and w = 0 this is the
    // flat-face term below.
    static const int glm_pair_curved = [] {
      const char* e = std::getenv("SPECTRE_GLM_PAIR");
      return e == nullptr ? 0 : std::atoi(e);
    }();
    if (glm_pair_curved != 0 or force_glm_energy_pair) {
      for (size_t pt = 0; pt < num_points; ++pt) {
        if (not frames[pt].has_value()) {
          continue;
        }
        const FaceFrame& frame = *frames[pt];
        const double inv_sqrt_det = 1.0 / frame.sqrt_det_spatial_metric;
        const auto v_i = frame.components({{get<0>(spatial_velocity_int)[pt],
                                            get<1>(spatial_velocity_int)[pt],
                                            get<2>(spatial_velocity_int)[pt]}});
        const auto v_e = frame.components({{get<0>(spatial_velocity_ext)[pt],
                                            get<1>(spatial_velocity_ext)[pt],
                                            get<2>(spatial_velocity_ext)[pt]}});
        const auto b_i =
            frame.components({{get<0>(tilde_b_int)[pt] * inv_sqrt_det,
                               get<1>(tilde_b_int)[pt] * inv_sqrt_det,
                               get<2>(tilde_b_int)[pt] * inv_sqrt_det}});
        const auto b_e =
            frame.components({{get<0>(tilde_b_ext)[pt] * inv_sqrt_det,
                               get<1>(tilde_b_ext)[pt] * inv_sqrt_det,
                               get<2>(tilde_b_ext)[pt] * inv_sqrt_det}});
        double v2 = 0.0, vb = 0.0;
        for (size_t a = 0; a < 3; ++a) {
          const double va = 0.5 * (v_i[a] + v_e[a]);
          v2 += va * va;
          vb += va * 0.5 * (b_i[a] + b_e[a]);
        }
        const double vn = 0.5 * (v_i[0] + v_e[0]);
        get(*boundary_correction_tilde_tau)[pt] +=
            glm_phi_sel[pt] *
            ((1.0 - v2) * glm_normal_b_star[pt] * inv_sqrt_det + vb * vn);
      }
    }
    add_face_frame_counts(tally);
  }

  // --- MHD flux from PLUTO's own HLLD ---------------------------------------
  // Nothing about the five-wave fan is implemented here. This function only
  //   (a) rotates each interface into a normal-aligned frame,
  //   (b) converts to PLUTO's RMHD primitive layout,
  //   (c) hands the batch to PLUTO's HLLD_Solver via the C shim, and
  //   (d) rotates the returned flux back into the inertial frame.
  // See pluto/README.md for the vendored sources and the list of local edits.
  //
  // PLUTO can produce inf/nan for unphysical trial states inside its own root
  // find (it falls back to HLL internally), so FP-exception trapping is off for
  // the duration of the call, exactly as for the hand-ported solver.
  const ScopedFpeState pluto_fpe_scope(false);

  // Gather the interfaces PLUTO can handle; the rest keep the HLL flux that has
  // already been written above.
  std::vector<size_t> pt_index;
  size_t flat_density_cutoff = 0;
  std::vector<std::array<double, 3>> triad_n, triad_t1, triad_t2;
  std::vector<double> pluto_vl, pluto_vr, pt_gamma;
  pt_index.reserve(num_points);
  triad_n.reserve(num_points);
  triad_t1.reserve(num_points);
  triad_t2.reserve(num_points);
  pluto_vl.reserve(8 * num_points);
  pluto_vr.reserve(8 * num_points);
  pt_gamma.reserve(num_points);

  for (size_t i = 0; i < num_points; ++i) {
    if (path[i] != PointPath::Flat) {
      continue;  // solved in the face frame above, or kept on HLL
    }
    if (below_density_cutoff(i)) {
      ++flat_density_cutoff;
      continue;  // the experiment: keep the light-speed HLL flux
    }
    const double rho_i = get(rest_mass_density_int)[i];
    const double eps_i = get(specific_internal_energy_int)[i];
    const double p_i = get(pressure_int)[i];
    if (rho_i <= 0.0 or eps_i <= 0.0 or p_i <= 0.0) {
      continue;  // keep HLL where the ideal-gas relation is undefined
    }

    const std::array<double, 3> n{get<0>(interface_unit_normal_int)[i],
                                  get<1>(interface_unit_normal_int)[i],
                                  get<2>(interface_unit_normal_int)[i]};
    std::array<double, 3> t1{}, t2{};
    make_triad(n, &t1, &t2);
    auto proj = [&](const tnsr::I<DataVector, 3, Frame::Inertial>& v,
                    size_t comp) {
      const std::array<double, 3> vv{v.get(0)[i], v.get(1)[i], v.get(2)[i]};
      const std::array<double, 3>& e = comp == 0 ? n : (comp == 1 ? t1 : t2);
      return vv[0] * e[0] + vv[1] * e[1] + vv[2] * e[2];
    };
    // PLUTO's RMHD primitive vector is
    //   {RHO, VX1, VX2, VX3, BX1, BX2, BX3, PRS}
    // with VX1/BX1 along the interface normal. NOTE the velocity is the
    // 3-VELOCITY, not W*v: PLUTO forms its Lorentz factor internally as
    // g2 = 1/(1 - v.v) (mappers.c PrimToCons). The hand-ported solver took
    // W*v here, so this is a genuine difference in what gets passed.
    // PLUTO's GLM_Solve replaces BOTH sides' normal field (and psi) by the
    // face values of the 2x2 GLM Riemann problem before any Riemann solver
    // runs, so hlld.c / hllx.c always see one B_n. Handing them the two sides'
    // own B_n (the behaviour until 2026-10-13) gives an inconsistent fan
    // wherever [B_n] != 0; HLLD's diffusion tolerated it, the seven-wave
    // Newton did not (a cell-scale instability in the periodic slab's slow
    // rarefaction, with 100x PLUTO's divergence error). glm_normal_b_star is
    // the light-speed HLL state of B_n, identical to PLUTO's Bm for c_h = 1.
    // SPECTRE_GLM_BN_STAR=0 restores the old behaviour for comparison.
    static const bool use_glm_bn_star = [] {
      const char* e = std::getenv("SPECTRE_GLM_BN_STAR");
      return e == nullptr or std::atoi(e) != 0;
    }();
    auto push = [&](std::vector<double>* out, const Scalar<DataVector>& rho,
                    const tnsr::I<DataVector, 3, Frame::Inertial>& vel,
                    const tnsr::I<DataVector, 3, Frame::Inertial>& bfield,
                    const Scalar<DataVector>& press) {
      out->push_back(get(rho)[i]);
      out->push_back(proj(vel, 0));
      out->push_back(proj(vel, 1));
      out->push_back(proj(vel, 2));
      out->push_back((use_glm_bn_star or force_common_normal_field)
                         ? glm_normal_b_star[i]
                         : proj(bfield, 0));
      out->push_back(proj(bfield, 1));
      out->push_back(proj(bfield, 2));
      out->push_back(get(press)[i]);
    };
    push(&pluto_vl, rest_mass_density_int, spatial_velocity_int, tilde_b_int,
         pressure_int);
    push(&pluto_vr, rest_mass_density_ext, spatial_velocity_ext, tilde_b_ext,
         pressure_ext);
    pt_index.push_back(i);
    triad_n.push_back(n);
    triad_t1.push_back(t1);
    triad_t2.push_back(t2);
    // ideal-gas index from the interior primitives (constant for an ideal EoS)
    pt_gamma.push_back(1.0 + p_i / (rho_i * eps_i));
  }

  if (flat_density_cutoff > 0) {
    FaceFrameCounts flat_tally{};
    flat_tally.density_cutoff = flat_density_cutoff;
    add_face_frame_counts(flat_tally);
  }
  if (pt_index.empty()) {
    return;
  }

  // PLUTO takes ONE adiabatic index per call (its EoS is a global). For an
  // ideal EoS every point shares it, so the whole face goes in a single call;
  // if that ever stops holding we solve point by point rather than silently
  // applying one point's gamma to another's state.
  const double gamma_min = *std::min_element(pt_gamma.begin(), pt_gamma.end());
  const double gamma_max = *std::max_element(pt_gamma.begin(), pt_gamma.end());
  const bool uniform_gamma =
      gamma_max - gamma_min <= 1.0e-12 * std::max(1.0, gamma_max);

  const size_t n_solve = pt_index.size();
  std::vector<double> pluto_flux(8 * n_solve, 0.0);
  std::vector<double> pluto_press(n_solve, 0.0);
  if (uniform_gamma) {
    pluto_solver(static_cast<int>(n_solve), pluto_vl.data(), pluto_vr.data(),
                 gamma_min, pluto_flux.data(), pluto_press.data());
  } else {
    for (size_t k = 0; k < n_solve; ++k) {
      pluto_solver(1, pluto_vl.data() + 8 * k, pluto_vr.data() + 8 * k,
                   pt_gamma[k], pluto_flux.data() + 8 * k,
                   pluto_press.data() + k);
    }
  }

  // Scatter back. PLUTO's conserved order is
  //   {RHO, MX1, MX2, MX3, BX1, BX2, BX3, ENG}
  // with ENG the REDUCED energy E - D, which is exactly SpECTRE's TildeTau.
  // PLUTO also carries the normal total pressure OUTSIDE the momentum flux
  // (sweep->press), so it must be added back onto the normal component.
  for (size_t k = 0; k < n_solve; ++k) {
    const double* f = pluto_flux.data() + 8 * k;
    bool all_finite = true;
    for (size_t q = 0; q < 8; ++q) {
      all_finite = all_finite and std::isfinite(f[q]);
    }
    if (not all_finite or not std::isfinite(pluto_press[k])) {
      continue;  // keep HLL if PLUTO returned a non-finite result
    }
    const size_t i = pt_index[k];
    double g_ye = 0.0;
    if (force_mass_consistent_ye) {
      // DYe is a passive conserved density: advect the upwind concentration
      // with the SAME accepted numerical mass flux. In particular, constant
      // Ye must obey G(DYe) = Ye G(D). Retaining an independent HLL DYe flux
      // after replacing G(D) would violate that identity. No change is made
      // to any field until this concentration and its flux are usable.
      const double upwind_d =
          f[0] >= 0.0 ? get(tilde_d_int)[i] : get(tilde_d_ext)[i];
      const double upwind_dye =
          f[0] >= 0.0 ? get(tilde_ye_int)[i] : get(tilde_ye_ext)[i];
      if (not std::isfinite(upwind_d) or upwind_d <= 0.0 or
          not std::isfinite(upwind_dye)) {
        continue;  // retain the complete HLL baseline
      }
      const double upwind_ye = upwind_dye / upwind_d;
      g_ye = upwind_ye * f[0];
      if (not std::isfinite(upwind_ye) or not std::isfinite(g_ye)) {
        continue;
      }
    }
    const std::array<double, 3>& n = triad_n[k];
    const std::array<double, 3>& t1 = triad_t1[k];
    const std::array<double, 3>& t2 = triad_t2[k];

    const double f_mn = f[1] + pluto_press[k];  // normal momentum + p_tot
    const double f_mt1 = f[2];
    const double f_mt2 = f[3];
    // PLUTO returns fx[BXn] = 0.0 (fluxes.c:64) because with DIVB_CONTROL=NO
    // it has no evolution equation for the NORMAL field -- with GLM enabled it
    // would return v[PSI_GLM] there instead (fluxes.c:76). SpECTRE's normal-B
    // flux belongs to the GLM subsystem and is NOT zero, so taking PLUTO's
    // value would silently disable divergence cleaning at every interface this
    // solver handles. That is invisible in 1D (B_n is uniform, no div-B is
    // generated, psi stays 0 -- which is why this matched PLUTO to 0.1% on the
    // whole ST1 ladder) and destructive in 2D, where it produced grid-scale
    // magnetic noise growing with |B|: KHI enstrophy 1202 at sigma=0.1 versus
    // 37 at sigma=0.01, against 32 for HLLEM-All. Take the normal part from the
    // GLM split computed above and only the TANGENTIAL field from PLUTO.
    std::array<double, 3> g_s{}, g_b{};
    for (size_t q = 0; q < 3; ++q) {
      g_s[q] = f_mn * n[q] + f_mt1 * t1[q] + f_mt2 * t2[q];
      g_b[q] = glm_normal_b_flux[i] * n[q] + f[5] * t1[q] + f[6] * t2[q];
    }
    const double g_d = f[0];
    const double g_tau = f[7];

    // strong form returns G - F_int; weak form returns G
    const double sub_d = weak ? 0.0 : get(normal_dot_flux_tilde_d_int)[i];
    const double sub_tau = weak ? 0.0 : get(normal_dot_flux_tilde_tau_int)[i];
    get(*boundary_correction_tilde_d)[i] = g_d - sub_d;
    if (force_mass_consistent_ye) {
      get(*boundary_correction_tilde_ye)[i] =
          g_ye - (weak ? 0.0 : get(normal_dot_flux_tilde_ye_int)[i]);
    }
    get(*boundary_correction_tilde_tau)[i] = g_tau - sub_tau;
    for (size_t q = 0; q < 3; ++q) {
      boundary_correction_tilde_s->get(q)[i] =
          g_s[q] - (weak ? 0.0 : normal_dot_flux_tilde_s_int.get(q)[i]);
      boundary_correction_tilde_b->get(q)[i] =
          g_b[q] - (weak ? 0.0 : normal_dot_flux_tilde_b_int.get(q)[i]);
    }
  }

  // GLM ENERGY PAIRING (2026-10-08; environment SPECTRE_GLM_PAIR=1, default off while it is
  // tested; flat space only, like the GLM split above). ValenciaDivClean transports the
  // normal field with the cleaning flux Phi and Phi with B_n, but its TildeTau flux carries no
  // matching term, so the non-solenoidal part of the field update changes magnetic energy at
  // fixed total energy and the gas pays for it -- at beta ~ 1e-3 the seven-wave solver's 2-D
  // contact entropy loss (task 2 of the HLLx campaign, 2026-10-07/08; PLUTO's RMHD GLM has the
  // same omission and the same term, GLM_PAIR, removes the loss there). The relativistic
  // entropy variables pair a field change dB at fixed (D, m, E) with dE = (b/W).dB,
  // b/W = (1 - v^2) B + (v.B) v; in flux form on every face, with the GLM face values
  // Phi* (= the normal-field flux) and B_n*:
  //   F_tau += Phi* [(1 - v^2) B_n* + (v.B) v_n],   v, B the face means.
  // It is added to the numerical flux G in both DG formulations (the system's volume flux has
  // no such term; on FD-subcell faces, which the tube runs use, the update is the face-flux
  // difference and this is exactly PLUTO's scheme).
  {
    static const int glm_pair = [] {
      const char* e = std::getenv("SPECTRE_GLM_PAIR");
      return e == nullptr ? 0 : std::atoi(e);
    }();
    if (glm_pair != 0 or force_glm_energy_pair) {
      const auto& n = interface_unit_normal_int;
      for (size_t i = 0; i < num_points; ++i) {
        // flat points only; the curved points got the GR form above
        if (path[i] != PointPath::Flat) {
          continue;
        }
        double v2 = 0.0, vb = 0.0, vn = 0.0;
        for (size_t q = 0; q < 3; ++q) {
          const double vq = 0.5 * (spatial_velocity_int.get(q)[i] +
                                   spatial_velocity_ext.get(q)[i]);
          const double bq =
              0.5 * (tilde_b_int.get(q)[i] + tilde_b_ext.get(q)[i]);
          v2 += vq * vq;
          vb += vq * bq;
          vn += vq * n.get(q)[i];
        }
        get(*boundary_correction_tilde_tau)[i] +=
            glm_normal_b_flux[i] *
            ((1.0 - v2) * glm_normal_b_star[i] + vb * vn);
      }
    }
  }
}
}  // namespace detail

void PlutoHlld::dg_boundary_terms(
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
    const Scalar<DataVector>& interface_lapse_int,
    const Scalar<DataVector>& interface_shift_dot_normal_int,
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        interface_spatial_metric_int,
    const Scalar<DataVector>& rest_mass_density_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_int,
    const Scalar<DataVector>& pressure_int,
    const Scalar<DataVector>& lorentz_factor_int,
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
    const Scalar<DataVector>& fast_outgoing_char_speed_ext,
    const Scalar<DataVector>& fast_ingoing_char_speed_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& iface_normal_ext,
    const Scalar<DataVector>& metric_flatness_ext,
    const Scalar<DataVector>& interface_lapse_ext,
    const Scalar<DataVector>& interface_shift_dot_normal_ext,
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        interface_spatial_metric_ext,
    const Scalar<DataVector>& rest_mass_density_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_ext,
    const Scalar<DataVector>& pressure_ext,
    const Scalar<DataVector>& lorentz_factor_ext,
    const Scalar<DataVector>& specific_internal_energy_ext,
    const dg::Formulation dg_formulation) const {
  detail::pluto_dg_boundary_terms(
      &pluto_hlld_flux, &pluto_hlld_flux_moving, boundary_correction_tilde_d,
      boundary_correction_tilde_ye, boundary_correction_tilde_tau,
      boundary_correction_tilde_s, boundary_correction_tilde_b,
      boundary_correction_tilde_phi, tilde_d_int, tilde_ye_int, tilde_tau_int,
      tilde_s_int, tilde_b_int, tilde_phi_int, normal_dot_flux_tilde_d_int,
      normal_dot_flux_tilde_ye_int, normal_dot_flux_tilde_tau_int,
      normal_dot_flux_tilde_s_int, normal_dot_flux_tilde_b_int,
      normal_dot_flux_tilde_phi_int, largest_outgoing_char_speed_int,
      largest_ingoing_char_speed_int, fast_outgoing_char_speed_int,
      fast_ingoing_char_speed_int, interface_unit_normal_int,
      metric_flatness_int, interface_lapse_int, interface_shift_dot_normal_int,
      interface_spatial_metric_int, rest_mass_density_int, spatial_velocity_int,
      pressure_int, lorentz_factor_int, specific_internal_energy_int,
      tilde_d_ext, tilde_ye_ext, tilde_tau_ext, tilde_s_ext, tilde_b_ext,
      tilde_phi_ext, normal_dot_flux_tilde_d_ext, normal_dot_flux_tilde_ye_ext,
      normal_dot_flux_tilde_tau_ext, normal_dot_flux_tilde_s_ext,
      normal_dot_flux_tilde_b_ext, normal_dot_flux_tilde_phi_ext,
      largest_outgoing_char_speed_ext, largest_ingoing_char_speed_ext,
      fast_outgoing_char_speed_ext, fast_ingoing_char_speed_ext,
      iface_normal_ext, metric_flatness_ext, interface_lapse_ext,
      interface_shift_dot_normal_ext, interface_spatial_metric_ext,
      rest_mass_density_ext, spatial_velocity_ext, pressure_ext,
      lorentz_factor_ext, specific_internal_energy_ext, dg_formulation,
      light_speed_density_cutoff_);
}

bool operator==(const PlutoHlld& lhs, const PlutoHlld& rhs) {
  return lhs.magnetic_field_magnitude_for_hydro_ ==
             rhs.magnetic_field_magnitude_for_hydro_ and
         lhs.light_speed_density_cutoff_ == rhs.light_speed_density_cutoff_;
}
bool operator!=(const PlutoHlld& lhs, const PlutoHlld& rhs) {
  return not(lhs == rhs);
}

// NOLINTNEXTLINE
PUP::able::PUP_ID PlutoHlld::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
