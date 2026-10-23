// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <pup.h>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace grmhd::ValenciaDivClean::BoundaryConditions {
/// What `NozzleDirichletAnalytic` imposes at ghost points outside the nozzle.
enum class NozzleOutside {
  /// Zero-gradient extrapolation of the primitive variables, inflow allowed
  /// (the "simple extrapolation" of the Del Zanna et al. jet tests).
  Extrapolation,
  /// `HydroFreeOutflow`: extrapolation with the inflowing normal velocity
  /// removed.
  FreeOutflow,
  /// `Reflective` with `ReflectBoth: false`.
  Reflective,
  /// `Reflective` with `ReflectBoth: true`, a reflecting wall.
  ReflectiveBoth
};

std::ostream& operator<<(std::ostream& os, NozzleOutside outside);

/*!
 * \brief `DirichletAnalytic` inside a nozzle radius and another condition
 * outside it, for the inflow face of a jet.
 *
 * Ghost points whose \f$|x^0|\f$ (the cylindrical radius in a `CartoonCylinder`
 * domain, whose axis is \f$x^1\f$) is at most `NozzleRadius` take exactly the
 * values `DirichletAnalytic` gives them. The others take the values of the
 * condition chosen by `Outside`:
 *
 * - `Extrapolation`: the values `HydroFreeOutflow` gives, except that the
 *   normal component of \f$W v^i\f$ is copied from the boundary cell whatever
 *   its sign, so the ghost cells are a zero-gradient copy of the boundary
 *   cell's primitives with \f$\Phi = 0\f$. Cell-centered ghost fluxes
 *   (`FiniteDifferenceDerivativeOrder` above 2) are recomputed from these
 *   primitives, with the boundary cell's pressure, specific internal energy
 *   and metric. On DG faces it falls back to `HydroFreeOutflow`.
 * - `FreeOutflow`: `HydroFreeOutflow`.
 * - `Reflective`, `ReflectiveBoth`: `Reflective` with `ReflectBoth` false or
 *   true.
 *
 * Unlike `DirichletAnalytic`, the FD ghost data of this condition does not
 * pass through the atmosphere treatment that `ReconstructRhoTimesTemperature`
 * applies to `DirichletAnalytic` ghost data, so the analytic data must not
 * vanish inside the nozzle.
 */
class NozzleDirichletAnalytic final : public BoundaryCondition {
 private:
  template <typename T>
  using Flux = ::Tags::Flux<T, tmpl::size_t<3>, Frame::Inertial>;
  using Shift = gr::Tags::Shift<DataVector, 3>;
  using Lapse = gr::Tags::Lapse<DataVector>;
  using SpatialMetric = gr::Tags::SpatialMetric<DataVector, 3>;
  using InvSpatialMetric = gr::Tags::InverseSpatialMetric<DataVector, 3>;

 public:
  struct AnalyticPrescription {
    static constexpr Options::String help =
        "What analytic solution/data to prescribe inside the nozzle.";
    using type = std::unique_ptr<evolution::initial_data::InitialData>;
  };
  struct NozzleRadius {
    using type = double;
    static constexpr Options::String help =
        "Ghost points with |x^0| at most this radius take the analytic data.";
    static type lower_bound() { return 0.0; }
  };
  struct Outside {
    using type = NozzleOutside;
    static constexpr Options::String help =
        "The condition outside the nozzle radius: Extrapolation, FreeOutflow, "
        "Reflective or ReflectiveBoth.";
  };
  using options = tmpl::list<AnalyticPrescription, NozzleRadius, Outside>;
  static constexpr Options::String help{
      "DirichletAnalytic inside a nozzle radius and extrapolation, free "
      "outflow or reflection outside it."};

  NozzleDirichletAnalytic() = default;
  NozzleDirichletAnalytic(NozzleDirichletAnalytic&&) = default;
  NozzleDirichletAnalytic& operator=(NozzleDirichletAnalytic&&) = default;
  NozzleDirichletAnalytic(const NozzleDirichletAnalytic& rhs);
  NozzleDirichletAnalytic& operator=(const NozzleDirichletAnalytic& rhs);
  ~NozzleDirichletAnalytic() override = default;

  explicit NozzleDirichletAnalytic(CkMigrateMessage* msg);

  NozzleDirichletAnalytic(std::unique_ptr<evolution::initial_data::InitialData>
                              analytic_prescription,
                          double nozzle_radius, NozzleOutside outside);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, NozzleDirichletAnalytic);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_primitive_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::Temperature<DataVector>>;
  using dg_interior_temporary_tags =
      tmpl::list<Shift, Lapse, InvSpatialMetric,
                 domain::Tags::Coordinates<3, Frame::Inertial>>;
  using dg_gridless_tags = tmpl::list<::Tags::Time>;

  std::optional<std::string> dg_ghost(
      gsl::not_null<Scalar<DataVector>*> tilde_d,
      gsl::not_null<Scalar<DataVector>*> tilde_ye,
      gsl::not_null<Scalar<DataVector>*> tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
      gsl::not_null<Scalar<DataVector>*> tilde_phi,

      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_ye_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_tau_flux,
      gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_phi_flux,

      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          spatial_velocity_one_form,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> spatial_metric,
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> electron_fraction,
      gsl::not_null<Scalar<DataVector>*> temperature,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<Scalar<DataVector>*> lorentz_factor,
      gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
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
      const tnsr::II<DataVector, 3, Frame::Inertial>&
          interior_inv_spatial_metric,
      const tnsr::I<DataVector, 3, Frame::Inertial>& coords,

      double time) const;

  // Overload for boundary corrections that do not package the spatial metric.
  std::optional<std::string> dg_ghost(
      gsl::not_null<Scalar<DataVector>*> tilde_d,
      gsl::not_null<Scalar<DataVector>*> tilde_ye,
      gsl::not_null<Scalar<DataVector>*> tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
      gsl::not_null<Scalar<DataVector>*> tilde_phi,

      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_ye_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_tau_flux,
      gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_phi_flux,

      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          spatial_velocity_one_form,
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> electron_fraction,
      gsl::not_null<Scalar<DataVector>*> temperature,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<Scalar<DataVector>*> lorentz_factor,
      gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
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
      const tnsr::II<DataVector, 3, Frame::Inertial>&
          interior_inv_spatial_metric,
      const tnsr::I<DataVector, 3, Frame::Inertial>& coords,

      double time) const;

  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>, Shift, Lapse,
                 SpatialMetric>;
  using fd_interior_primitive_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Temperature<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>>;
  using fd_gridless_tags =
      tmpl::list<::Tags::Time, ::domain::Tags::FunctionsOfTime,
                 domain::Tags::ElementMap<3, Frame::Grid>,
                 domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                             Frame::Inertial>,
                 fd::Tags::Reconstructor>;

  void fd_ghost(
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> electron_fraction,
      gsl::not_null<Scalar<DataVector>*> temperature,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          lorentz_factor_times_spatial_velocity,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> magnetic_field,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
      gsl::not_null<std::optional<Variables<db::wrap_tags_in<
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
      double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const ElementMap<3, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
          grid_to_inertial_map,
      const fd::Reconstructor& reconstructor) const;

  double nozzle_radius() const { return nozzle_radius_; }
  NozzleOutside outside() const { return outside_; }

 private:
  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
  double nozzle_radius_{0.0};
  NozzleOutside outside_{NozzleOutside::Extrapolation};
};
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions

template <>
struct Options::create_from_yaml<
    grmhd::ValenciaDivClean::BoundaryConditions::NozzleOutside> {
  template <typename Metavariables>
  static grmhd::ValenciaDivClean::BoundaryConditions::NozzleOutside create(
      const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
grmhd::ValenciaDivClean::BoundaryConditions::NozzleOutside
Options::create_from_yaml<
    grmhd::ValenciaDivClean::BoundaryConditions::NozzleOutside>::
    create<void>(const Options::Option& options);
