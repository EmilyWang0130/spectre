// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/MarquinaCpm.hpp"

#include <memory>
#include <optional>
#include <pup.h>

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
MarquinaCpm::MarquinaCpm(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> MarquinaCpm::get_clone() const {
  return std::make_unique<MarquinaCpm>(*this);
}

void MarquinaCpm::pup(PUP::er& p) { BoundaryCorrection::pup(p); }

double MarquinaCpm::dg_package_data(
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
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::NoFrame>*>
        packaged_characteristic_speeds,
    const gsl::not_null<tnsr::iJ<DataVector, 6, Frame::NoFrame>*>
        packaged_left_characteristic_fields,
    const gsl::not_null<tnsr::ij<DataVector, 6, Frame::NoFrame>*>
        packaged_right_characteristic_fields,
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
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) {
  return Marquina::dg_package_data_impl(
      true, packaged_tilde_d, packaged_tilde_ye, packaged_tilde_tau,
      packaged_tilde_s, packaged_tilde_b, packaged_tilde_phi,
      packaged_normal_dot_flux_tilde_d, packaged_normal_dot_flux_tilde_ye,
      packaged_normal_dot_flux_tilde_tau, packaged_normal_dot_flux_tilde_s,
      packaged_normal_dot_flux_tilde_b, packaged_normal_dot_flux_tilde_phi,
      packaged_characteristic_speeds, packaged_left_characteristic_fields,
      packaged_right_characteristic_fields, tilde_d, tilde_ye, tilde_tau,
      tilde_s, tilde_b, tilde_phi, flux_tilde_d, flux_tilde_ye, flux_tilde_tau,
      flux_tilde_s, flux_tilde_b, flux_tilde_phi, lapse, shift,
      spatial_velocity_one_form, spatial_metric, rest_mass_density,
      electron_fraction, temperature, spatial_velocity,
      specific_internal_energy, pressure, lorentz_factor, normal_covector,
      normal_vector, mesh_velocity, normal_dot_mesh_velocity,
      equation_of_state);
}

void MarquinaCpm::dg_boundary_terms(
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
    const tnsr::i<DataVector, 3, Frame::NoFrame>& characteristic_speeds_int,
    const tnsr::iJ<DataVector, 6, Frame::NoFrame>&
        left_characteristic_fields_int,
    const tnsr::ij<DataVector, 6, Frame::NoFrame>&
        right_characteristic_fields_int,
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
    const tnsr::i<DataVector, 3, Frame::NoFrame>& characteristic_speeds_ext,
    const tnsr::iJ<DataVector, 6, Frame::NoFrame>&
        left_characteristic_fields_ext,
    const tnsr::ij<DataVector, 6, Frame::NoFrame>&
        right_characteristic_fields_ext,
    const dg::Formulation dg_formulation) {
  Marquina::dg_boundary_terms_impl(
      true, boundary_correction_tilde_d, boundary_correction_tilde_ye,
      boundary_correction_tilde_tau, boundary_correction_tilde_s,
      boundary_correction_tilde_b, boundary_correction_tilde_phi, tilde_d_int,
      tilde_ye_int, tilde_tau_int, tilde_s_int, tilde_b_int, tilde_phi_int,
      normal_dot_flux_tilde_d_int, normal_dot_flux_tilde_ye_int,
      normal_dot_flux_tilde_tau_int, normal_dot_flux_tilde_s_int,
      normal_dot_flux_tilde_b_int, normal_dot_flux_tilde_phi_int,
      characteristic_speeds_int, left_characteristic_fields_int,
      right_characteristic_fields_int, tilde_d_ext, tilde_ye_ext,
      tilde_tau_ext, tilde_s_ext, tilde_b_ext, tilde_phi_ext,
      normal_dot_flux_tilde_d_ext, normal_dot_flux_tilde_ye_ext,
      normal_dot_flux_tilde_tau_ext, normal_dot_flux_tilde_s_ext,
      normal_dot_flux_tilde_b_ext, normal_dot_flux_tilde_phi_ext,
      characteristic_speeds_ext, left_characteristic_fields_ext,
      right_characteristic_fields_ext, dg_formulation);
}

bool operator==(const MarquinaCpm& /*lhs*/, const MarquinaCpm& /*rhs*/) {
  return true;
}

bool operator!=(const MarquinaCpm& lhs, const MarquinaCpm& rhs) {
  return not(lhs == rhs);
}

// NOLINTNEXTLINE
PUP::able::PUP_ID MarquinaCpm::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
