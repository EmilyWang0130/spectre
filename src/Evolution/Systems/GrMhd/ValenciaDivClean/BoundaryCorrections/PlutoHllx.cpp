// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoHllx.hpp"

#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/pluto/pluto_hlld_shim.h"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {

PlutoHllx::PlutoHllx(const double magnetic_field_magnitude_for_hydro,
                     const double light_speed_density_cutoff)
    : magnetic_field_magnitude_for_hydro_(magnetic_field_magnitude_for_hydro),
      light_speed_density_cutoff_(light_speed_density_cutoff) {}

PlutoHllx::PlutoHllx(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> PlutoHllx::get_clone() const {
  return std::make_unique<PlutoHllx>(*this);
}

void PlutoHllx::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | magnetic_field_magnitude_for_hydro_;
  p | light_speed_density_cutoff_;
}

// The packaged data is exactly PlutoHlld's: both solvers take the same
// primitive state on each side and the same per-side fast and light bounds.
double PlutoHllx::dg_package_data(
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

// PlutoHlld's boundary terms, with the seven-wave solver (hllx.c) in place
// of its HLLD.
void PlutoHllx::dg_boundary_terms(
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
      &pluto_hllx_flux, &pluto_hllx_flux_moving, boundary_correction_tilde_d,
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

bool operator==(const PlutoHllx& lhs, const PlutoHllx& rhs) {
  return lhs.magnetic_field_magnitude_for_hydro_ ==
             rhs.magnetic_field_magnitude_for_hydro_ and
         lhs.light_speed_density_cutoff_ == rhs.light_speed_density_cutoff_;
}
bool operator!=(const PlutoHllx& lhs, const PlutoHllx& rhs) {
  return not(lhs == rhs);
}

// NOLINTNEXTLINE
PUP::able::PUP_ID PlutoHllx::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
