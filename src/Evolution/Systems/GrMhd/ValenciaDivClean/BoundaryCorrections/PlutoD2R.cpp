// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoD2R.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/D2R/Flux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {

PlutoD2R::PlutoD2R(const double magnetic_field_magnitude_for_hydro,
                   const double light_speed_density_cutoff)
    : magnetic_field_magnitude_for_hydro_(magnetic_field_magnitude_for_hydro),
      light_speed_density_cutoff_(light_speed_density_cutoff) {}

PlutoD2R::PlutoD2R(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> PlutoD2R::get_clone() const {
  return std::make_unique<PlutoD2R>(*this);
}

void PlutoD2R::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | magnetic_field_magnitude_for_hydro_;
  p | light_speed_density_cutoff_;
}

// Use the shared packaging on supported SR faces. Unsupported input is
// packaged directly with physical light bounds before any flat-space
// characteristic evaluation can occur.
double PlutoD2R::dg_package_data(
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
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) const {
  const size_t num_points = get(rest_mass_density).size();
  const bool ideal_eos =
      dynamic_cast<const EquationsOfState::Equilibrium3D<
          EquationsOfState::IdealFluid<true>>*>(&equation_of_state) != nullptr;
  std::vector<bool> supported(num_points, ideal_eos);
  DataVector mesh_normal_speed(num_points, 0.0);
  bool all_supported = true;
  for (size_t pt = 0; pt < num_points; ++pt) {
    if (normal_dot_mesh_velocity.has_value()) {
      mesh_normal_speed[pt] = get(*normal_dot_mesh_velocity)[pt];
    } else if (mesh_velocity.has_value()) {
      for (size_t d = 0; d < 3; ++d) {
        mesh_normal_speed[pt] +=
            mesh_velocity->get(d)[pt] * normal_covector.get(d)[pt];
      }
    }
    // The metric is no longer part of the gate: curved points are solved in
    // the shared adapter's face frame (PlutoFaceFrame.hpp) at ray 0, and keep
    // its HLL baseline where the face moves (w != 0), because D2-R's kernel
    // samples only x/t = 0. The ideal EoS, a static mesh and admissible
    // primitives are still required.
    supported[pt] = supported[pt] and mesh_normal_speed[pt] == 0.0;
    const double rho = get(rest_mass_density)[pt];
    const double eps = get(specific_internal_energy)[pt];
    const double p = get(pressure)[pt];
    if (not(std::isfinite(rho) and std::isfinite(eps) and std::isfinite(p) and
            rho > 0.0 and eps > 0.0 and p > 0.0)) {
      supported[pt] = false;
    } else {
      const double gamma = 1.0 + p / (rho * eps);
      // Gamma is recomputed from p, rho, eps, so Gamma = 2 (Balsara 1) can land
      // an ulp above 2; a face would then silently drop to light-speed HLL.
      // Allow a relative 1e-12, the same slack the shared adapter uses when it
      // checks that the batch's Gamma is uniform.
      supported[pt] = supported[pt] and std::isfinite(gamma) and gamma > 1.0 and
                      gamma <= 2.0 * (1.0 + 1.0e-12);
    }
    all_supported = all_supported and supported[pt];
  }
  if (not all_supported) {
    *packaged_tilde_d = tilde_d;
    *packaged_tilde_ye = tilde_ye;
    *packaged_tilde_tau = tilde_tau;
    *packaged_tilde_s = tilde_s;
    *packaged_tilde_b = tilde_b;
    *packaged_tilde_phi = tilde_phi;
    *packaged_interface_unit_normal = normal_covector;
    *packaged_interface_lapse = lapse;
    dot_product(packaged_interface_shift_dot_normal, shift, normal_covector);
    get(*packaged_interface_shift_dot_normal) += mesh_normal_speed;
    *packaged_interface_spatial_metric = spatial_metric;
    *packaged_rest_mass_density = rest_mass_density;
    *packaged_spatial_velocity = spatial_velocity;
    *packaged_pressure = pressure;
    *packaged_lorentz_factor = lorentz_factor;
    *packaged_specific_internal_energy = specific_internal_energy;
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
    double largest_speed = 0.0;
    for (size_t pt = 0; pt < num_points; ++pt) {
      double shift_n = 0.0;
      for (size_t d = 0; d < 3; ++d) {
        shift_n += shift.get(d)[pt] * normal_covector.get(d)[pt];
      }
      const double plus = get(lapse)[pt] - shift_n - mesh_normal_speed[pt];
      const double minus = -get(lapse)[pt] - shift_n - mesh_normal_speed[pt];
      get(*packaged_largest_outgoing_char_speed)[pt] = plus;
      get(*packaged_largest_ingoing_char_speed)[pt] = minus;
      get(*packaged_fast_outgoing_char_speed)[pt] = plus;
      get(*packaged_fast_ingoing_char_speed)[pt] = minus;
      // A supported point carries PlutoHlld's flatness indicator, so that a
      // curved one takes the face frame (zero exactly on a static Minkowski
      // face, as before); infinity marks the unsupported points (HLL only).
      double flatness = std::abs(get(lapse)[pt] - 1.0);
      for (size_t i = 0; i < 3; ++i) {
        flatness += std::abs(shift.get(i)[pt]);
      }
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          flatness +=
              std::abs(spatial_metric.get(i, j)[pt] - (i == j ? 1.0 : 0.0));
        }
      }
      get(*packaged_metric_flatness)[pt] =
          supported[pt] ? flatness : std::numeric_limits<double>::infinity();
      largest_speed =
          std::max(largest_speed, std::max(std::abs(plus), std::abs(minus)));
    }
    return largest_speed;
  }
  return PlutoHlld{magnetic_field_magnitude_for_hydro_,
                   light_speed_density_cutoff_}
      .dg_package_data(
          packaged_tilde_d, packaged_tilde_ye, packaged_tilde_tau,
          packaged_tilde_s, packaged_tilde_b, packaged_tilde_phi,
          packaged_normal_dot_flux_tilde_d, packaged_normal_dot_flux_tilde_ye,
          packaged_normal_dot_flux_tilde_tau, packaged_normal_dot_flux_tilde_s,
          packaged_normal_dot_flux_tilde_b, packaged_normal_dot_flux_tilde_phi,
          packaged_largest_outgoing_char_speed,
          packaged_largest_ingoing_char_speed,
          packaged_fast_outgoing_char_speed, packaged_fast_ingoing_char_speed,
          packaged_interface_unit_normal, packaged_metric_flatness,
          packaged_interface_lapse, packaged_interface_shift_dot_normal,
          packaged_interface_spatial_metric, packaged_rest_mass_density,
          packaged_spatial_velocity, packaged_pressure, packaged_lorentz_factor,
          packaged_specific_internal_energy, tilde_d, tilde_ye, tilde_tau,
          tilde_s, tilde_b, tilde_phi, flux_tilde_d, flux_tilde_ye,
          flux_tilde_tau, flux_tilde_s, flux_tilde_b, flux_tilde_phi, lapse,
          shift, spatial_velocity_one_form, spatial_metric, rest_mass_density,
          electron_fraction, temperature, spatial_velocity,
          specific_internal_energy, pressure, lorentz_factor, normal_covector,
          normal_vector, mesh_velocity, normal_dot_mesh_velocity,
          equation_of_state);
}

// The common adapter handles both DG forms, rotation, Ye and GLM. D2-R forces
// common normal field and GLM energy pairing without global environment edits.
void PlutoD2R::dg_boundary_terms(
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
      &pluto_d2r_flux, nullptr, boundary_correction_tilde_d,
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
      light_speed_density_cutoff_, true, true, true);
}

bool operator==(const PlutoD2R& lhs, const PlutoD2R& rhs) {
  return lhs.magnetic_field_magnitude_for_hydro_ ==
             rhs.magnetic_field_magnitude_for_hydro_ and
         lhs.light_speed_density_cutoff_ == rhs.light_speed_density_cutoff_;
}
bool operator!=(const PlutoD2R& lhs, const PlutoD2R& rhs) {
  return not(lhs == rhs);
}

// NOLINTNEXTLINE
PUP::able::PUP_ID PlutoD2R::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
