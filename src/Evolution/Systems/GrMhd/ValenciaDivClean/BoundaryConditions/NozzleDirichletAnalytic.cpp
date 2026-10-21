// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/NozzleDirichletAnalytic.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/HydroFreeOutflow.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Reflective.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Options.hpp"
#include "Options/ParseError.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
namespace {
// Overwrite the points of `target` where `inside` is true with those of
// `source`, component by component.
template <typename TensorType>
void overwrite_inside(const gsl::not_null<TensorType*> target,
                      const TensorType& source,
                      const std::vector<bool>& inside) {
  for (size_t component = 0; component < target->size(); ++component) {
    DataVector& target_component = (*target)[component];
    const DataVector& source_component = source[component];
    for (size_t point = 0; point < inside.size(); ++point) {
      if (inside[point]) {
        target_component[point] = source_component[point];
      }
    }
  }
}
}  // namespace

std::ostream& operator<<(std::ostream& os, const NozzleOutside outside) {
  switch (outside) {
    case NozzleOutside::Extrapolation:
      return os << "Extrapolation";
    case NozzleOutside::FreeOutflow:
      return os << "FreeOutflow";
    case NozzleOutside::Reflective:
      return os << "Reflective";
    case NozzleOutside::ReflectiveBoth:
      return os << "ReflectiveBoth";
    default:
      ERROR("Unknown NozzleOutside");
  }
}

NozzleDirichletAnalytic::NozzleDirichletAnalytic(
    const NozzleDirichletAnalytic& rhs)
    : BoundaryCondition{dynamic_cast<const BoundaryCondition&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()),
      nozzle_radius_(rhs.nozzle_radius_),
      outside_(rhs.outside_) {}

NozzleDirichletAnalytic& NozzleDirichletAnalytic::operator=(
    const NozzleDirichletAnalytic& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  nozzle_radius_ = rhs.nozzle_radius_;
  outside_ = rhs.outside_;
  return *this;
}

NozzleDirichletAnalytic::NozzleDirichletAnalytic(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

NozzleDirichletAnalytic::NozzleDirichletAnalytic(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription,
    const double nozzle_radius, const NozzleOutside outside)
    : analytic_prescription_(std::move(analytic_prescription)),
      nozzle_radius_(nozzle_radius),
      outside_(outside) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
NozzleDirichletAnalytic::get_clone() const {
  return std::make_unique<NozzleDirichletAnalytic>(*this);
}

void NozzleDirichletAnalytic::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
  p | analytic_prescription_;
  p | nozzle_radius_;
  p | outside_;
}

std::optional<std::string> NozzleDirichletAnalytic::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_ye_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,

    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        spatial_velocity_one_form,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        inv_spatial_metric,

    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_vector,

    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_electron_fraction,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_lorentz_factor,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_temperature,

    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::II<DataVector, 3, Frame::Inertial>& interior_inv_spatial_metric,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords,

    const double time) const {
  // The condition outside the nozzle, on every point of the face.
  std::optional<std::string> error{};
  if (outside_ == NozzleOutside::Reflective or
      outside_ == NozzleOutside::ReflectiveBoth) {
    error =
        BoundaryConditions::Reflective{outside_ ==
                                       NozzleOutside::ReflectiveBoth}
            .dg_ghost(tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi,
                      tilde_d_flux, tilde_ye_flux, tilde_tau_flux, tilde_s_flux,
                      tilde_b_flux, tilde_phi_flux, lapse, shift,
                      spatial_velocity_one_form, spatial_metric,
                      rest_mass_density, electron_fraction, temperature,
                      spatial_velocity, specific_internal_energy, pressure,
                      lorentz_factor, inv_spatial_metric, face_mesh_velocity,
                      outward_directed_normal_covector,
                      outward_directed_normal_vector,
                      interior_rest_mass_density, interior_electron_fraction,
                      interior_specific_internal_energy,
                      interior_spatial_velocity, interior_magnetic_field,
                      interior_lorentz_factor, interior_pressure,
                      interior_temperature, interior_shift, interior_lapse,
                      interior_inv_spatial_metric);
  } else {
    // FreeOutflow, and Extrapolation (which falls back to it on DG faces).
    error = HydroFreeOutflow::dg_ghost(
        tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi, tilde_d_flux,
        tilde_ye_flux, tilde_tau_flux, tilde_s_flux, tilde_b_flux,
        tilde_phi_flux, lapse, shift, spatial_velocity_one_form, spatial_metric,
        rest_mass_density, electron_fraction, temperature, spatial_velocity,
        specific_internal_energy, pressure, lorentz_factor, inv_spatial_metric,
        face_mesh_velocity, outward_directed_normal_covector,
        outward_directed_normal_vector, interior_rest_mass_density,
        interior_electron_fraction, interior_specific_internal_energy,
        interior_spatial_velocity, interior_magnetic_field,
        interior_lorentz_factor, interior_pressure, interior_temperature,
        interior_shift, interior_lapse, interior_inv_spatial_metric);
  }
  if (error.has_value()) {
    return error;
  }

  // The analytic data inside the nozzle.
  const size_t number_of_points = get<0>(coords).size();
  std::vector<bool> inside(number_of_points, false);
  bool any_inside = false;
  for (size_t point = 0; point < number_of_points; ++point) {
    inside[point] = std::abs(get<0>(coords)[point]) <= nozzle_radius_;
    any_inside = any_inside or inside[point];
  }
  if (not any_inside) {
    return {};
  }
  Scalar<DataVector> d_tilde_d{number_of_points};
  Scalar<DataVector> d_tilde_ye{number_of_points};
  Scalar<DataVector> d_tilde_tau{number_of_points};
  tnsr::i<DataVector, 3, Frame::Inertial> d_tilde_s{number_of_points};
  tnsr::I<DataVector, 3, Frame::Inertial> d_tilde_b{number_of_points};
  Scalar<DataVector> d_tilde_phi{number_of_points};
  tnsr::I<DataVector, 3, Frame::Inertial> d_tilde_d_flux{number_of_points};
  tnsr::I<DataVector, 3, Frame::Inertial> d_tilde_ye_flux{number_of_points};
  tnsr::I<DataVector, 3, Frame::Inertial> d_tilde_tau_flux{number_of_points};
  tnsr::Ij<DataVector, 3, Frame::Inertial> d_tilde_s_flux{number_of_points};
  tnsr::IJ<DataVector, 3, Frame::Inertial> d_tilde_b_flux{number_of_points};
  tnsr::I<DataVector, 3, Frame::Inertial> d_tilde_phi_flux{number_of_points};
  Scalar<DataVector> d_lapse{number_of_points};
  tnsr::I<DataVector, 3, Frame::Inertial> d_shift{number_of_points};
  tnsr::i<DataVector, 3, Frame::Inertial> d_spatial_velocity_one_form{
      number_of_points};
  tnsr::ii<DataVector, 3, Frame::Inertial> d_spatial_metric{number_of_points};
  Scalar<DataVector> d_rest_mass_density{number_of_points};
  Scalar<DataVector> d_electron_fraction{number_of_points};
  Scalar<DataVector> d_temperature{number_of_points};
  tnsr::I<DataVector, 3, Frame::Inertial> d_spatial_velocity{number_of_points};
  Scalar<DataVector> d_specific_internal_energy{number_of_points};
  Scalar<DataVector> d_pressure{number_of_points};
  Scalar<DataVector> d_lorentz_factor{number_of_points};
  tnsr::II<DataVector, 3, Frame::Inertial> d_inv_spatial_metric{
      number_of_points};
  error = DirichletAnalytic{analytic_prescription_->get_clone()}.dg_ghost(
      make_not_null(&d_tilde_d), make_not_null(&d_tilde_ye),
      make_not_null(&d_tilde_tau), make_not_null(&d_tilde_s),
      make_not_null(&d_tilde_b), make_not_null(&d_tilde_phi),
      make_not_null(&d_tilde_d_flux), make_not_null(&d_tilde_ye_flux),
      make_not_null(&d_tilde_tau_flux), make_not_null(&d_tilde_s_flux),
      make_not_null(&d_tilde_b_flux), make_not_null(&d_tilde_phi_flux),
      make_not_null(&d_lapse), make_not_null(&d_shift),
      make_not_null(&d_spatial_velocity_one_form),
      make_not_null(&d_spatial_metric), make_not_null(&d_rest_mass_density),
      make_not_null(&d_electron_fraction), make_not_null(&d_temperature),
      make_not_null(&d_spatial_velocity),
      make_not_null(&d_specific_internal_energy), make_not_null(&d_pressure),
      make_not_null(&d_lorentz_factor), make_not_null(&d_inv_spatial_metric),
      face_mesh_velocity, outward_directed_normal_covector,
      outward_directed_normal_vector, coords, time);
  if (error.has_value()) {
    return error;
  }
  overwrite_inside(tilde_d, d_tilde_d, inside);
  overwrite_inside(tilde_ye, d_tilde_ye, inside);
  overwrite_inside(tilde_tau, d_tilde_tau, inside);
  overwrite_inside(tilde_s, d_tilde_s, inside);
  overwrite_inside(tilde_b, d_tilde_b, inside);
  overwrite_inside(tilde_phi, d_tilde_phi, inside);
  overwrite_inside(tilde_d_flux, d_tilde_d_flux, inside);
  overwrite_inside(tilde_ye_flux, d_tilde_ye_flux, inside);
  overwrite_inside(tilde_tau_flux, d_tilde_tau_flux, inside);
  overwrite_inside(tilde_s_flux, d_tilde_s_flux, inside);
  overwrite_inside(tilde_b_flux, d_tilde_b_flux, inside);
  overwrite_inside(tilde_phi_flux, d_tilde_phi_flux, inside);
  overwrite_inside(lapse, d_lapse, inside);
  overwrite_inside(shift, d_shift, inside);
  overwrite_inside(spatial_velocity_one_form, d_spatial_velocity_one_form,
                   inside);
  overwrite_inside(spatial_metric, d_spatial_metric, inside);
  overwrite_inside(rest_mass_density, d_rest_mass_density, inside);
  overwrite_inside(electron_fraction, d_electron_fraction, inside);
  overwrite_inside(temperature, d_temperature, inside);
  overwrite_inside(spatial_velocity, d_spatial_velocity, inside);
  overwrite_inside(specific_internal_energy, d_specific_internal_energy,
                   inside);
  overwrite_inside(pressure, d_pressure, inside);
  overwrite_inside(lorentz_factor, d_lorentz_factor, inside);
  overwrite_inside(inv_spatial_metric, d_inv_spatial_metric, inside);
  return {};
}

std::optional<std::string> NozzleDirichletAnalytic::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_ye_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,

    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        spatial_velocity_one_form,
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        inv_spatial_metric,

    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_vector,

    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_electron_fraction,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_lorentz_factor,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_temperature,

    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::II<DataVector, 3, Frame::Inertial>& interior_inv_spatial_metric,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords,

    const double time) const {
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{
      get(interior_lapse).size()};
  return dg_ghost(
      tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi, tilde_d_flux,
      tilde_ye_flux, tilde_tau_flux, tilde_s_flux, tilde_b_flux, tilde_phi_flux,
      lapse, shift, spatial_velocity_one_form, make_not_null(&spatial_metric),
      rest_mass_density, electron_fraction, temperature, spatial_velocity,
      specific_internal_energy, pressure, lorentz_factor, inv_spatial_metric,
      face_mesh_velocity, outward_directed_normal_covector,
      outward_directed_normal_vector, interior_rest_mass_density,
      interior_electron_fraction, interior_specific_internal_energy,
      interior_spatial_velocity, interior_magnetic_field,
      interior_lorentz_factor, interior_pressure, interior_temperature,
      interior_shift, interior_lapse, interior_inv_spatial_metric, coords,
      time);
}

void NozzleDirichletAnalytic::fd_ghost(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        lorentz_factor_times_spatial_velocity,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
    const gsl::not_null<std::optional<Variables<db::wrap_tags_in<
        Flux, typename grmhd::ValenciaDivClean::System::flux_variables>>>*>
        cell_centered_ghost_fluxes,
    const Direction<3>& direction,

    // fd_interior_temporary_tags
    const Mesh<3>& subcell_mesh,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_spatial_metric,

    // fd_interior_primitive_variables_tags
    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_electron_fraction,
    const Scalar<DataVector>& interior_temperature,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const Scalar<DataVector>& interior_lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,

    // fd_gridless_tags
    const double time,
    const std::unordered_map<
        std::string,
        std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const ElementMap<3, Frame::Grid>& logical_to_grid_map,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial_map,
    const fd::Reconstructor& reconstructor) const {
  const size_t ghost_zone_size{reconstructor.ghost_zone_size()};

  // The condition outside the nozzle, on every ghost point.
  if (outside_ == NozzleOutside::Reflective or
      outside_ == NozzleOutside::ReflectiveBoth) {
    BoundaryConditions::Reflective{outside_ == NozzleOutside::ReflectiveBoth}
        .fd_ghost(rest_mass_density, electron_fraction, temperature,
                  lorentz_factor_times_spatial_velocity, magnetic_field,
                  divergence_cleaning_field, cell_centered_ghost_fluxes,
                  direction, subcell_mesh, interior_shift, interior_lapse,
                  interior_spatial_metric, interior_rest_mass_density,
                  interior_electron_fraction, interior_temperature,
                  interior_pressure, interior_specific_internal_energy,
                  interior_lorentz_factor, interior_spatial_velocity,
                  interior_magnetic_field, reconstructor);
  } else {
    if (outside_ == NozzleOutside::Extrapolation and
        cell_centered_ghost_fluxes->has_value()) {
      ERROR(
          "NozzleDirichletAnalytic with Outside: Extrapolation does not "
          "support cell-centered ghost fluxes (FiniteDifferenceDerivativeOrder "
          "above 2).");
    }
    HydroFreeOutflow::fd_ghost(
        rest_mass_density, electron_fraction, temperature,
        lorentz_factor_times_spatial_velocity, magnetic_field,
        divergence_cleaning_field, cell_centered_ghost_fluxes, direction,
        subcell_mesh, interior_shift, interior_lapse, interior_spatial_metric,
        interior_rest_mass_density, interior_electron_fraction,
        interior_temperature, interior_pressure,
        interior_specific_internal_energy, interior_lorentz_factor,
        interior_spatial_velocity, interior_magnetic_field, reconstructor);
    if (outside_ == NozzleOutside::Extrapolation) {
      // HydroFreeOutflow removed the inflowing normal velocity; restore it,
      // so that every ghost slice is the boundary cell's W v^i.
      const size_t normal_dimension = direction.dimension();
      const Index<3>& subcell_extents = subcell_mesh.extents();
      const size_t number_of_face_points =
          subcell_extents.slice_away(normal_dimension).product();
      using NormalMomentum = ::Tags::TempScalar<0>;
      Variables<tmpl::list<NormalMomentum>> boundary_slice{
          number_of_face_points};
      get(get<NormalMomentum>(boundary_slice)) =
          get(evolution::dg::subcell::slice_tensor_for_subcell(
              interior_lorentz_factor, subcell_extents, 1, direction, {})) *
          evolution::dg::subcell::slice_tensor_for_subcell(
              interior_spatial_velocity, subcell_extents, 1, direction, {})
              .get(normal_dimension);
      Index<3> ghost_extents = subcell_extents;
      ghost_extents[normal_dimension] = ghost_zone_size;
      // add_slice_to_data adds into the buffer, so it must start at zero.
      Variables<tmpl::list<NormalMomentum>> ghost_slices{
          number_of_face_points * ghost_zone_size, 0.0};
      for (size_t ghost_index = 0; ghost_index < ghost_zone_size;
           ++ghost_index) {
        add_slice_to_data(make_not_null(&ghost_slices), boundary_slice,
                          ghost_extents, normal_dimension, ghost_index);
      }
      lorentz_factor_times_spatial_velocity->get(normal_dimension) =
          get(get<NormalMomentum>(ghost_slices));
    }
  }

  // The analytic data inside the nozzle.
  const auto ghost_inertial_coords = grid_to_inertial_map(
      logical_to_grid_map(
          evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
              subcell_mesh, ghost_zone_size, direction)),
      time, functions_of_time);
  const size_t number_of_points = get<0>(ghost_inertial_coords).size();
  std::vector<bool> inside(number_of_points, false);
  bool any_inside = false;
  for (size_t point = 0; point < number_of_points; ++point) {
    inside[point] =
        std::abs(get<0>(ghost_inertial_coords)[point]) <= nozzle_radius_;
    any_inside = any_inside or inside[point];
  }
  if (not any_inside) {
    return;
  }
  Scalar<DataVector> d_rest_mass_density{number_of_points};
  Scalar<DataVector> d_electron_fraction{number_of_points};
  Scalar<DataVector> d_temperature{number_of_points};
  tnsr::I<DataVector, 3, Frame::Inertial>
      d_lorentz_factor_times_spatial_velocity{number_of_points};
  tnsr::I<DataVector, 3, Frame::Inertial> d_magnetic_field{number_of_points};
  Scalar<DataVector> d_divergence_cleaning_field{number_of_points};
  std::optional<Variables<db::wrap_tags_in<
      Flux, typename grmhd::ValenciaDivClean::System::flux_variables>>>
      d_cell_centered_ghost_fluxes{};
  if (cell_centered_ghost_fluxes->has_value()) {
    d_cell_centered_ghost_fluxes.emplace(number_of_points);
  }
  DirichletAnalytic{analytic_prescription_->get_clone()}.fd_ghost(
      make_not_null(&d_rest_mass_density), make_not_null(&d_electron_fraction),
      make_not_null(&d_temperature),
      make_not_null(&d_lorentz_factor_times_spatial_velocity),
      make_not_null(&d_magnetic_field),
      make_not_null(&d_divergence_cleaning_field),
      make_not_null(&d_cell_centered_ghost_fluxes), direction, subcell_mesh,
      time, functions_of_time, logical_to_grid_map, grid_to_inertial_map,
      reconstructor);
  overwrite_inside(rest_mass_density, d_rest_mass_density, inside);
  overwrite_inside(electron_fraction, d_electron_fraction, inside);
  overwrite_inside(temperature, d_temperature, inside);
  overwrite_inside(lorentz_factor_times_spatial_velocity,
                   d_lorentz_factor_times_spatial_velocity, inside);
  overwrite_inside(magnetic_field, d_magnetic_field, inside);
  overwrite_inside(divergence_cleaning_field, d_divergence_cleaning_field,
                   inside);
  if (cell_centered_ghost_fluxes->has_value()) {
    auto& fluxes = cell_centered_ghost_fluxes->value();
    const auto& d_fluxes = d_cell_centered_ghost_fluxes.value();
    const size_t number_of_components = fluxes.number_of_independent_components;
    for (size_t component = 0; component < number_of_components; ++component) {
      for (size_t point = 0; point < number_of_points; ++point) {
        if (inside[point]) {
          fluxes.data()[component * number_of_points + point] =
              d_fluxes.data()[component * number_of_points + point];
        }
      }
    }
  }
}

// NOLINTNEXTLINE
PUP::able::PUP_ID NozzleDirichletAnalytic::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions

template <>
grmhd::ValenciaDivClean::BoundaryConditions::NozzleOutside
Options::create_from_yaml<
    grmhd::ValenciaDivClean::BoundaryConditions::NozzleOutside>::
    create<void>(const Options::Option& options) {
  namespace bc = grmhd::ValenciaDivClean::BoundaryConditions;
  const auto type_read = options.parse_as<std::string>();
  if (type_read == "Extrapolation") {
    return bc::NozzleOutside::Extrapolation;
  } else if (type_read == "FreeOutflow") {
    return bc::NozzleOutside::FreeOutflow;
  } else if (type_read == "Reflective") {
    return bc::NozzleOutside::Reflective;
  } else if (type_read == "ReflectiveBoth") {
    return bc::NozzleOutside::ReflectiveBoth;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to NozzleOutside. Must be one of Extrapolation, "
                     "FreeOutflow, Reflective or ReflectiveBoth.");
}
