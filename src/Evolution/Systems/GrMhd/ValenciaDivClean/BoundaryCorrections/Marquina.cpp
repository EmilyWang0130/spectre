// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Marquina.hpp"

#include <algorithm>
#include <cmath>
#include <pup.h>

#include <memory>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
Marquina::Marquina(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> Marquina::get_clone() const {
  return std::make_unique<Marquina>(*this);
}

void Marquina::pup(PUP::er& p) { BoundaryCorrection::pup(p); }

double Marquina::dg_package_data(
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
        packaged_left_eigenvectors,
    const gsl::not_null<tnsr::ij<DataVector, 6, Frame::NoFrame>*>
        packaged_right_eigenvectors,
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
  return dg_package_data_impl(
      false, packaged_tilde_d, packaged_tilde_ye, packaged_tilde_tau,
      packaged_tilde_s, packaged_tilde_b, packaged_tilde_phi,
      packaged_normal_dot_flux_tilde_d, packaged_normal_dot_flux_tilde_ye,
      packaged_normal_dot_flux_tilde_tau, packaged_normal_dot_flux_tilde_s,
      packaged_normal_dot_flux_tilde_b, packaged_normal_dot_flux_tilde_phi,
      packaged_characteristic_speeds, packaged_left_eigenvectors,
      packaged_right_eigenvectors, tilde_d, tilde_ye, tilde_tau, tilde_s,
      tilde_b, tilde_phi, flux_tilde_d, flux_tilde_ye, flux_tilde_tau,
      flux_tilde_s, flux_tilde_b, flux_tilde_phi, lapse, shift,
      spatial_velocity_one_form, spatial_metric, rest_mass_density,
      electron_fraction, temperature, spatial_velocity,
      specific_internal_energy, pressure, lorentz_factor, normal_covector,
      normal_vector, mesh_velocity, normal_dot_mesh_velocity,
      equation_of_state);
}

double Marquina::dg_package_data_impl(
    const bool use_cpm_degenerate_block,
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
        packaged_left_eigenvectors,
    const gsl::not_null<tnsr::ij<DataVector, 6, Frame::NoFrame>*>
        packaged_right_eigenvectors,

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

    const Scalar<DataVector>& /*lapse*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*shift*/,
    const tnsr::i<DataVector, 3,
                  Frame::Inertial>& /*spatial_velocity_one_form*/,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,

    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& /*temperature*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& /*pressure*/,
    const Scalar<DataVector>& lorentz_factor,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) {
  const size_t num_points = get(tilde_d).size();
  const Scalar<DataVector> consistent_pressure =
      equation_of_state.pressure_from_density_and_energy(
          rest_mass_density, specific_internal_energy, electron_fraction);
  Scalar<DataVector> specific_enthalpy{num_points};
  get(specific_enthalpy) = 1.0 + get(specific_internal_energy) +
                           get(consistent_pressure) / get(rest_mass_density);
  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto normal_covector_mag =
      magnitude(normal_covector, inv_spatial_metric);
  tnsr::i<DataVector, 3, Frame::Inertial> unit_normal_covector{num_points};
  for (size_t i = 0; i < 3; ++i) {
    unit_normal_covector.get(i) =
        normal_covector.get(i) / get(normal_covector_mag);
  }

  // Compute characteristic decomposition
  // TO-DO: update functions to stop using std::array
  std::array<DataVector, 3> tmp_char_speeds = {DataVector(num_points, 0.0),
                                               DataVector(num_points, 0.0),
                                               DataVector(num_points, 0.0)};
  std::array<tnsr::I<DataVector, 6>, 6> tmp_left_eigenvectors =
      make_array<6>(make_with_value<tnsr::I<DataVector, 6>>(tilde_d, 0.0));
  std::array<tnsr::i<DataVector, 6>, 6> tmp_right_eigenvectors =
      make_array<6>(make_with_value<tnsr::i<DataVector, 6>>(tilde_d, 0.0));
  characteristic_speeds_hydro(
      make_not_null(&tmp_char_speeds), spatial_velocity, rest_mass_density,
      specific_internal_energy, specific_enthalpy, electron_fraction,
      lorentz_factor, unit_normal_covector, spatial_metric, equation_of_state);
  if (use_cpm_degenerate_block) {
    // CPM: only the two acoustic eigenvectors are needed. Leave the
    // degenerate-block rows (0..3) as zero; the CPM branch of
    // dg_boundary_terms never reads them.
    std::array<tnsr::I<DataVector, 6>, 2> tmp_left_acoustic =
        make_array<2>(make_with_value<tnsr::I<DataVector, 6>>(tilde_d, 0.0));
    std::array<tnsr::i<DataVector, 6>, 2> tmp_right_acoustic =
        make_array<2>(make_with_value<tnsr::i<DataVector, 6>>(tilde_d, 0.0));
    acoustic_eigenvectors_hydro(
        make_not_null(&tmp_right_acoustic), make_not_null(&tmp_left_acoustic),
        spatial_velocity, rest_mass_density, specific_internal_energy,
        specific_enthalpy, electron_fraction, lorentz_factor,
        unit_normal_covector, spatial_metric, equation_of_state);
    for (size_t j = 0; j < 6; ++j) {
      tmp_left_eigenvectors[HydroVectorL::Lplus].get(j) =
          tmp_left_acoustic[AcousticHydroVector::AcousticPlus].get(j);
      tmp_left_eigenvectors[HydroVectorL::Lminus].get(j) =
          tmp_left_acoustic[AcousticHydroVector::AcousticMinus].get(j);
      tmp_right_eigenvectors[HydroVectorR::Rplus].get(j) =
          tmp_right_acoustic[AcousticHydroVector::AcousticPlus].get(j);
      tmp_right_eigenvectors[HydroVectorR::Rminus].get(j) =
          tmp_right_acoustic[AcousticHydroVector::AcousticMinus].get(j);
    }
  } else {
    eigenvectors_hydro(make_not_null(&tmp_right_eigenvectors),
                       make_not_null(&tmp_left_eigenvectors), spatial_velocity,
                       rest_mass_density, specific_internal_energy,
                       specific_enthalpy, electron_fraction, lorentz_factor,
                       unit_normal_covector, spatial_metric, equation_of_state);
  }
  // Copy from std::array to tnsr
  for (size_t i = 0; i < 3; ++i) {
    packaged_characteristic_speeds->get(i) = tmp_char_speeds[i];
  }
  for (size_t i = 0; i < 6; ++i) {
    for (size_t j = 0; j < 6; ++j) {
      packaged_left_eigenvectors->get(i, j) = tmp_left_eigenvectors[i].get(j);
      packaged_right_eigenvectors->get(i, j) = tmp_right_eigenvectors[i].get(j);
    }
  }

  // Package conservative variables
  *packaged_tilde_d = tilde_d;
  *packaged_tilde_ye = tilde_ye;
  *packaged_tilde_tau = tilde_tau;
  *packaged_tilde_s = tilde_s;
  *packaged_tilde_b = tilde_b;
  *packaged_tilde_phi = tilde_phi;

  // Package conservative fluxes dotted with normal
  normal_dot_flux(packaged_normal_dot_flux_tilde_d, unit_normal_covector,
                  flux_tilde_d);
  normal_dot_flux(packaged_normal_dot_flux_tilde_ye, unit_normal_covector,
                  flux_tilde_ye);
  normal_dot_flux(packaged_normal_dot_flux_tilde_tau, unit_normal_covector,
                  flux_tilde_tau);
  normal_dot_flux(packaged_normal_dot_flux_tilde_s, unit_normal_covector,
                  flux_tilde_s);
  normal_dot_flux(packaged_normal_dot_flux_tilde_b, unit_normal_covector,
                  flux_tilde_b);
  normal_dot_flux(packaged_normal_dot_flux_tilde_phi, unit_normal_covector,
                  flux_tilde_phi);

  // Return the maximum absolute characteristic speed so that time step doesn't
  // violate CFL condition.
  using std::max;
  double max_abs_char_speed = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    max_abs_char_speed =
        max(max_abs_char_speed, max((*packaged_characteristic_speeds)[i]));
  }
  return max_abs_char_speed;
}

void Marquina::dg_boundary_terms(
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
  dg_boundary_terms_impl(
      false, boundary_correction_tilde_d, boundary_correction_tilde_ye,
      boundary_correction_tilde_tau, boundary_correction_tilde_s,
      boundary_correction_tilde_b, boundary_correction_tilde_phi, tilde_d_int,
      tilde_ye_int, tilde_tau_int, tilde_s_int, tilde_b_int, tilde_phi_int,
      normal_dot_flux_tilde_d_int, normal_dot_flux_tilde_ye_int,
      normal_dot_flux_tilde_tau_int, normal_dot_flux_tilde_s_int,
      normal_dot_flux_tilde_b_int, normal_dot_flux_tilde_phi_int,
      characteristic_speeds_int, left_characteristic_fields_int,
      right_characteristic_fields_int, tilde_d_ext, tilde_ye_ext, tilde_tau_ext,
      tilde_s_ext, tilde_b_ext, tilde_phi_ext, normal_dot_flux_tilde_d_ext,
      normal_dot_flux_tilde_ye_ext, normal_dot_flux_tilde_tau_ext,
      normal_dot_flux_tilde_s_ext, normal_dot_flux_tilde_b_ext,
      normal_dot_flux_tilde_phi_ext, characteristic_speeds_ext,
      left_characteristic_fields_ext, right_characteristic_fields_ext,
      dg_formulation);
}

void Marquina::dg_boundary_terms_impl(
    const bool use_cpm_degenerate_block,
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
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*tilde_b_int*/,
    const Scalar<DataVector>& /*tilde_phi_int*/,
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
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*tilde_b_ext*/,
    const Scalar<DataVector>& /*tilde_phi_ext*/,
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
  auto aligned_characteristic_speeds_ext = characteristic_speeds_ext;
  aligned_characteristic_speeds_ext.get(
      grmhd::ValenciaDivClean::HydroSpeed::NormalDotVelocity) =
      -characteristic_speeds_ext.get(
          grmhd::ValenciaDivClean::HydroSpeed::NormalDotVelocity);
  aligned_characteristic_speeds_ext.get(
      grmhd::ValenciaDivClean::HydroSpeed::LambdaPlus) =
      -characteristic_speeds_ext.get(
          grmhd::ValenciaDivClean::HydroSpeed::LambdaMinus);
  aligned_characteristic_speeds_ext.get(
      grmhd::ValenciaDivClean::HydroSpeed::LambdaMinus) =
      -characteristic_speeds_ext.get(
          grmhd::ValenciaDivClean::HydroSpeed::LambdaPlus);

  auto aligned_left_characteristic_fields_ext = left_characteristic_fields_ext;
  auto aligned_right_characteristic_fields_ext =
      right_characteristic_fields_ext;
  for (size_t j = 0; j < 6; ++j) {
    aligned_left_characteristic_fields_ext.get(
        grmhd::ValenciaDivClean::HydroVectorR::Rplus, j) =
        left_characteristic_fields_ext.get(
            grmhd::ValenciaDivClean::HydroVectorR::Rminus, j);
    aligned_left_characteristic_fields_ext.get(
        grmhd::ValenciaDivClean::HydroVectorR::Rminus, j) =
        left_characteristic_fields_ext.get(
            grmhd::ValenciaDivClean::HydroVectorR::Rplus, j);
    aligned_right_characteristic_fields_ext.get(
        grmhd::ValenciaDivClean::HydroVectorR::Rplus, j) =
        right_characteristic_fields_ext.get(
            grmhd::ValenciaDivClean::HydroVectorR::Rminus, j);
    aligned_right_characteristic_fields_ext.get(
        grmhd::ValenciaDivClean::HydroVectorR::Rminus, j) =
        right_characteristic_fields_ext.get(
            grmhd::ValenciaDivClean::HydroVectorR::Rplus, j);
  }
  // Initialize boundary corrections to zero, as we'll compute them by adding
  // the contributions from each characteristic field
  const size_t num_points = get(tilde_d_int).size();
  for (size_t point = 0; point < get(tilde_d_int).size(); ++point) {
    get(*boundary_correction_tilde_d) = 0.0;
    get<0>(*boundary_correction_tilde_s) = 0.0;
    get<1>(*boundary_correction_tilde_s) = 0.0;
    get<2>(*boundary_correction_tilde_s) = 0.0;
    get(*boundary_correction_tilde_tau) = 0.0;
    get(*boundary_correction_tilde_ye) = 0.0;

    // Not yet implemented for magnetic field and divergence cleaning field, so
    // set to zero
    get<0>(*boundary_correction_tilde_b) = 0.0;
    get<1>(*boundary_correction_tilde_b) = 0.0;
    get<2>(*boundary_correction_tilde_b) = 0.0;
    get(*boundary_correction_tilde_phi) = 0.0;
  }

  // Fallback flux for unmodeled B and Phi fields to satisfy DG contracts
  for (size_t j = 0; j < 3; ++j) {
    boundary_correction_tilde_b->get(j) =
        0.5 * (normal_dot_flux_tilde_b_int.get(j) -
               normal_dot_flux_tilde_b_ext.get(j));
  }
  get(*boundary_correction_tilde_phi) =
      0.5 *
      (get(normal_dot_flux_tilde_phi_int) - get(normal_dot_flux_tilde_phi_ext));

  // Temporary variables
  Scalar<DataVector> omega_i_int{num_points};
  Scalar<DataVector> omega_i_ext{num_points};
  Scalar<DataVector> phi_i_int{num_points};
  Scalar<DataVector> phi_i_ext{num_points};
  Scalar<DataVector> phi_i_plus{num_points};
  Scalar<DataVector> phi_i_minus{num_points};

  // Loop over characteristic fields.
  // In CPM mode, skip the four degenerate modes (i < Rplus) — the degenerate
  // block is handled after the loop via the spectral projector complement.
  const size_t i_start =
      use_cpm_degenerate_block
          ? static_cast<size_t>(grmhd::ValenciaDivClean::HydroVectorR::Rplus)
          : 0;
  for (size_t i = i_start; i < 6; ++i) {
    // Project conservative variables and normal fluxes onto "characteristic
    // basis"
    get(omega_i_int) =
        left_characteristic_fields_int.get(i, 0) * get(tilde_d_int) +
        left_characteristic_fields_int.get(i, 1) * get<0>(tilde_s_int) +
        left_characteristic_fields_int.get(i, 2) * get<1>(tilde_s_int) +
        left_characteristic_fields_int.get(i, 3) * get<2>(tilde_s_int) +
        left_characteristic_fields_int.get(i, 4) * get(tilde_tau_int) +
        left_characteristic_fields_int.get(i, 5) * get(tilde_ye_int);
    get(omega_i_ext) =
        aligned_left_characteristic_fields_ext.get(i, 0) * get(tilde_d_ext) +
        aligned_left_characteristic_fields_ext.get(i, 1) * get<0>(tilde_s_ext) +
        aligned_left_characteristic_fields_ext.get(i, 2) * get<1>(tilde_s_ext) +
        aligned_left_characteristic_fields_ext.get(i, 3) * get<2>(tilde_s_ext) +
        aligned_left_characteristic_fields_ext.get(i, 4) * get(tilde_tau_ext) +
        aligned_left_characteristic_fields_ext.get(i, 5) * get(tilde_ye_ext);
    get(phi_i_int) = left_characteristic_fields_int.get(i, 0) *
                         get(normal_dot_flux_tilde_d_int) +
                     left_characteristic_fields_int.get(i, 1) *
                         get<0>(normal_dot_flux_tilde_s_int) +
                     left_characteristic_fields_int.get(i, 2) *
                         get<1>(normal_dot_flux_tilde_s_int) +
                     left_characteristic_fields_int.get(i, 3) *
                         get<2>(normal_dot_flux_tilde_s_int) +
                     left_characteristic_fields_int.get(i, 4) *
                         get(normal_dot_flux_tilde_tau_int) +
                     left_characteristic_fields_int.get(i, 5) *
                         get(normal_dot_flux_tilde_ye_int);
    get(phi_i_ext) = aligned_left_characteristic_fields_ext.get(i, 0) *
                         (-get(normal_dot_flux_tilde_d_ext)) +
                     aligned_left_characteristic_fields_ext.get(i, 1) *
                         (-get<0>(normal_dot_flux_tilde_s_ext)) +
                     aligned_left_characteristic_fields_ext.get(i, 2) *
                         (-get<1>(normal_dot_flux_tilde_s_ext)) +
                     aligned_left_characteristic_fields_ext.get(i, 3) *
                         (-get<2>(normal_dot_flux_tilde_s_ext)) +
                     aligned_left_characteristic_fields_ext.get(i, 4) *
                         (-get(normal_dot_flux_tilde_tau_ext)) +
                     aligned_left_characteristic_fields_ext.get(i, 5) *
                         (-get(normal_dot_flux_tilde_ye_ext));

    // TO-DO: improve how we handle the indices of characteristic speeds
    size_t hydro_speed_index;
    switch (i) {
      case grmhd::ValenciaDivClean::HydroVectorR::R1:
      case grmhd::ValenciaDivClean::HydroVectorR::R2:
      case grmhd::ValenciaDivClean::HydroVectorR::R3:
      case grmhd::ValenciaDivClean::HydroVectorR::R4:
        hydro_speed_index =
            grmhd::ValenciaDivClean::HydroSpeed::NormalDotVelocity;
        break;
      case grmhd::ValenciaDivClean::HydroVectorR::Rplus:
        hydro_speed_index = grmhd::ValenciaDivClean::HydroSpeed::LambdaPlus;
        break;
      case grmhd::ValenciaDivClean::HydroVectorR::Rminus:
        hydro_speed_index = grmhd::ValenciaDivClean::HydroSpeed::LambdaMinus;
        break;
      default:
        ERROR("Unhandled index value in switch statement.");
    }

    // Compute Marquina fluxes in "characteristic basis"
    const DataVector& lambda_i_int =
        characteristic_speeds_int.get(hydro_speed_index);
    const DataVector& lambda_i_ext =
        aligned_characteristic_speeds_ext.get(hydro_speed_index);
    for (size_t point = 0; point < num_points; ++point) {
      if (lambda_i_int[point] >= 0.0 and lambda_i_ext[point] >= 0.0) {
        get(phi_i_plus)[point] = get(phi_i_int)[point];
        get(phi_i_minus)[point] = 0.0;
      } else if (lambda_i_int[point] <= 0.0 and lambda_i_ext[point] <= 0.0) {
        get(phi_i_plus)[point] = 0.0;
        get(phi_i_minus)[point] = get(phi_i_ext)[point];
      } else {
        double alpha = std::max(std::abs(lambda_i_int[point]),
                                std::abs(lambda_i_ext[point]));
        get(phi_i_plus)[point] =
            0.5 * (get(phi_i_int)[point] + alpha * get(omega_i_int)[point]);
        get(phi_i_minus)[point] =
            0.5 * (get(phi_i_ext)[point] - alpha * get(omega_i_ext)[point]);
      }
    }

    // Reconstruct Marquina fluxes in "conserved basis"
    // TO-DO: handle dg_formulation (strong/weak)
    get(*boundary_correction_tilde_d) +=
        get(phi_i_plus) * right_characteristic_fields_int.get(i, 0) +
        get(phi_i_minus) * aligned_right_characteristic_fields_ext.get(i, 0);
    get<0>(*boundary_correction_tilde_s) +=
        get(phi_i_plus) * right_characteristic_fields_int.get(i, 1) +
        get(phi_i_minus) * aligned_right_characteristic_fields_ext.get(i, 1);
    get<1>(*boundary_correction_tilde_s) +=
        get(phi_i_plus) * right_characteristic_fields_int.get(i, 2) +
        get(phi_i_minus) * aligned_right_characteristic_fields_ext.get(i, 2);
    get<2>(*boundary_correction_tilde_s) +=
        get(phi_i_plus) * right_characteristic_fields_int.get(i, 3) +
        get(phi_i_minus) * aligned_right_characteristic_fields_ext.get(i, 3);
    get(*boundary_correction_tilde_tau) +=
        get(phi_i_plus) * right_characteristic_fields_int.get(i, 4) +
        get(phi_i_minus) * aligned_right_characteristic_fields_ext.get(i, 4);
    get(*boundary_correction_tilde_ye) +=
        get(phi_i_plus) * right_characteristic_fields_int.get(i, 5) +
        get(phi_i_minus) * aligned_right_characteristic_fields_ext.get(i, 5);
  }

  if (use_cpm_degenerate_block) {
    // Fedkiw, Merriman & Osher 1997 §4.3, eqs (6) and (47). The
    // degenerate-block flux is the spectral projector complement
    // (I - R_+ L_+ - R_- L_-) F acted upon by a single-speed Marquina
    // upwind at lambda_0 = v_n. The four non-unique degenerate
    // eigenvectors never enter.
    using grmhd::ValenciaDivClean::HydroSpeed;
    using grmhd::ValenciaDivClean::HydroVectorL;
    using grmhd::ValenciaDivClean::HydroVectorR;

    Scalar<DataVector> omega_plus_int{num_points};
    Scalar<DataVector> omega_minus_int{num_points};
    Scalar<DataVector> omega_plus_ext{num_points};
    Scalar<DataVector> omega_minus_ext{num_points};
    Scalar<DataVector> phi_plus_int{num_points};
    Scalar<DataVector> phi_minus_int{num_points};
    Scalar<DataVector> phi_plus_ext{num_points};
    Scalar<DataVector> phi_minus_ext{num_points};

    const auto project = [&](Scalar<DataVector>* out, const auto& L_matrix,
                             size_t row, const auto& v_d, const auto& v_s,
                             const auto& v_tau, const auto& v_ye, double sign) {
      get(*out) = L_matrix.get(row, 0) * (sign * get(v_d)) +
                  L_matrix.get(row, 1) * (sign * get<0>(v_s)) +
                  L_matrix.get(row, 2) * (sign * get<1>(v_s)) +
                  L_matrix.get(row, 3) * (sign * get<2>(v_s)) +
                  L_matrix.get(row, 4) * (sign * get(v_tau)) +
                  L_matrix.get(row, 5) * (sign * get(v_ye));
    };

    project(&omega_plus_int, left_characteristic_fields_int,
            HydroVectorL::Lplus, tilde_d_int, tilde_s_int, tilde_tau_int,
            tilde_ye_int, 1.0);
    project(&omega_minus_int, left_characteristic_fields_int,
            HydroVectorL::Lminus, tilde_d_int, tilde_s_int, tilde_tau_int,
            tilde_ye_int, 1.0);
    project(&omega_plus_ext, aligned_left_characteristic_fields_ext,
            HydroVectorL::Lplus, tilde_d_ext, tilde_s_ext, tilde_tau_ext,
            tilde_ye_ext, 1.0);
    project(&omega_minus_ext, aligned_left_characteristic_fields_ext,
            HydroVectorL::Lminus, tilde_d_ext, tilde_s_ext, tilde_tau_ext,
            tilde_ye_ext, 1.0);
    project(&phi_plus_int, left_characteristic_fields_int, HydroVectorL::Lplus,
            normal_dot_flux_tilde_d_int, normal_dot_flux_tilde_s_int,
            normal_dot_flux_tilde_tau_int, normal_dot_flux_tilde_ye_int, 1.0);
    project(&phi_minus_int, left_characteristic_fields_int,
            HydroVectorL::Lminus, normal_dot_flux_tilde_d_int,
            normal_dot_flux_tilde_s_int, normal_dot_flux_tilde_tau_int,
            normal_dot_flux_tilde_ye_int, 1.0);
    // Exterior fluxes flip sign under the aligned-normal convention.
    project(&phi_plus_ext, aligned_left_characteristic_fields_ext,
            HydroVectorL::Lplus, normal_dot_flux_tilde_d_ext,
            normal_dot_flux_tilde_s_ext, normal_dot_flux_tilde_tau_ext,
            normal_dot_flux_tilde_ye_ext, -1.0);
    project(&phi_minus_ext, aligned_left_characteristic_fields_ext,
            HydroVectorL::Lminus, normal_dot_flux_tilde_d_ext,
            normal_dot_flux_tilde_s_ext, normal_dot_flux_tilde_tau_ext,
            normal_dot_flux_tilde_ye_ext, -1.0);

    const DataVector& lambda_0_int =
        characteristic_speeds_int.get(HydroSpeed::NormalDotVelocity);
    const DataVector& lambda_0_ext =
        aligned_characteristic_speeds_ext.get(HydroSpeed::NormalDotVelocity);

    const auto cpm_add_component = [&](const gsl::not_null<DataVector*> bc_j,
                                       const DataVector& u_int_j,
                                       const DataVector& u_ext_j,
                                       const DataVector& f_int_j,
                                       const DataVector& f_ext_j,
                                       const size_t j) {
      const DataVector f_deg_int_j =
          f_int_j -
          right_characteristic_fields_int.get(HydroVectorR::Rplus, j) *
              get(phi_plus_int) -
          right_characteristic_fields_int.get(HydroVectorR::Rminus, j) *
              get(phi_minus_int);
      const DataVector f_deg_ext_j =
          (-f_ext_j) -
          aligned_right_characteristic_fields_ext.get(HydroVectorR::Rplus, j) *
              get(phi_plus_ext) -
          aligned_right_characteristic_fields_ext.get(HydroVectorR::Rminus, j) *
              get(phi_minus_ext);
      const DataVector u_deg_int_j =
          u_int_j -
          right_characteristic_fields_int.get(HydroVectorR::Rplus, j) *
              get(omega_plus_int) -
          right_characteristic_fields_int.get(HydroVectorR::Rminus, j) *
              get(omega_minus_int);
      const DataVector u_deg_ext_j =
          u_ext_j -
          aligned_right_characteristic_fields_ext.get(HydroVectorR::Rplus, j) *
              get(omega_plus_ext) -
          aligned_right_characteristic_fields_ext.get(HydroVectorR::Rminus, j) *
              get(omega_minus_ext);

      for (size_t point = 0; point < num_points; ++point) {
        double f_deg_plus;
        double f_deg_minus;
        if (lambda_0_int[point] >= 0.0 and lambda_0_ext[point] >= 0.0) {
          f_deg_plus = f_deg_int_j[point];
          f_deg_minus = 0.0;
        } else if (lambda_0_int[point] <= 0.0 and lambda_0_ext[point] <= 0.0) {
          f_deg_plus = 0.0;
          f_deg_minus = f_deg_ext_j[point];
        } else {
          const double alpha = std::max(std::abs(lambda_0_int[point]),
                                        std::abs(lambda_0_ext[point]));
          f_deg_plus = 0.5 * (f_deg_int_j[point] + alpha * u_deg_int_j[point]);
          f_deg_minus = 0.5 * (f_deg_ext_j[point] - alpha * u_deg_ext_j[point]);
        }
        (*bc_j)[point] += f_deg_plus + f_deg_minus;
      }
    };

    cpm_add_component(make_not_null(&get(*boundary_correction_tilde_d)),
                      get(tilde_d_int), get(tilde_d_ext),
                      get(normal_dot_flux_tilde_d_int),
                      get(normal_dot_flux_tilde_d_ext), 0);
    cpm_add_component(make_not_null(&get<0>(*boundary_correction_tilde_s)),
                      get<0>(tilde_s_int), get<0>(tilde_s_ext),
                      get<0>(normal_dot_flux_tilde_s_int),
                      get<0>(normal_dot_flux_tilde_s_ext), 1);
    cpm_add_component(make_not_null(&get<1>(*boundary_correction_tilde_s)),
                      get<1>(tilde_s_int), get<1>(tilde_s_ext),
                      get<1>(normal_dot_flux_tilde_s_int),
                      get<1>(normal_dot_flux_tilde_s_ext), 2);
    cpm_add_component(make_not_null(&get<2>(*boundary_correction_tilde_s)),
                      get<2>(tilde_s_int), get<2>(tilde_s_ext),
                      get<2>(normal_dot_flux_tilde_s_int),
                      get<2>(normal_dot_flux_tilde_s_ext), 3);
    cpm_add_component(make_not_null(&get(*boundary_correction_tilde_tau)),
                      get(tilde_tau_int), get(tilde_tau_ext),
                      get(normal_dot_flux_tilde_tau_int),
                      get(normal_dot_flux_tilde_tau_ext), 4);
    cpm_add_component(make_not_null(&get(*boundary_correction_tilde_ye)),
                      get(tilde_ye_int), get(tilde_ye_ext),
                      get(normal_dot_flux_tilde_ye_int),
                      get(normal_dot_flux_tilde_ye_ext), 5);
  }

  if (dg_formulation == dg::Formulation::StrongInertial) {
    get(*boundary_correction_tilde_d) -= get(normal_dot_flux_tilde_d_int);
    get<0>(*boundary_correction_tilde_s) -= get<0>(normal_dot_flux_tilde_s_int);
    get<1>(*boundary_correction_tilde_s) -= get<1>(normal_dot_flux_tilde_s_int);
    get<2>(*boundary_correction_tilde_s) -= get<2>(normal_dot_flux_tilde_s_int);
    get(*boundary_correction_tilde_tau) -= get(normal_dot_flux_tilde_tau_int);
    get(*boundary_correction_tilde_ye) -= get(normal_dot_flux_tilde_ye_int);
    get<0>(*boundary_correction_tilde_b) -= get<0>(normal_dot_flux_tilde_b_int);
    get<1>(*boundary_correction_tilde_b) -= get<1>(normal_dot_flux_tilde_b_int);
    get<2>(*boundary_correction_tilde_b) -= get<2>(normal_dot_flux_tilde_b_int);
    get(*boundary_correction_tilde_phi) -= get(normal_dot_flux_tilde_phi_int);
  }
}

bool operator==(const Marquina& /*lhs*/, const Marquina& /*rhs*/) {
  return true;
}

bool operator!=(const Marquina& lhs, const Marquina& rhs) {
  return not(lhs == rhs);
}

// NOLINTNEXTLINE
PUP::able::PUP_ID Marquina::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
