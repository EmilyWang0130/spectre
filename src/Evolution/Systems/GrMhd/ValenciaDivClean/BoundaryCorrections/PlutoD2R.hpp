// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoHlld.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
/*!
 * \brief Experimental flat-SR D2-R slow-mode response flux on refined HLLD.
 *
 * The compiled kernel uses analytical slow projectors, a physically checked
 * input ray and the original-face entropy budget. It has no reference-data
 * dependency. The implementation and its scope are documented in D2R/README.md.
 *
 * Supported faces have an ideal Gamma-law EOS with 1 < Gamma <= 2 and zero
 * normal mesh speed. Other faces retain a conservative light-bound HLL flux.
 * On a curved background the shared adapter solves the face in its face frame
 * (PlutoFaceFrame.hpp), but D2-R's kernel samples only the ray x/t = 0: curved
 * points whose face moves through the normal observer's frame (normal shift
 * beta^n != 0) keep the adapter's HLL baseline. A moving-face D2-R fan has not
 * been derived, so ALE uses HLL.
 *
 * The shared PLUTO adapter rotates into the normal frame, uses a common GLM
 * normal-field state and restores the GLM normal-field/Phi fluxes. Electron
 * fraction is transported as the upwind concentration times the accepted
 * mass flux, so a constant concentration remains consistent with mass.
 * D2-R explicitly enables common Bn and GLM energy pairing, independently of
 * legacy environment switches. The seven-state entropy check does not certify
 * the complete GLM, DG/subcell or Runge-Kutta update.
 */
class PlutoD2R final : public evolution::BoundaryCorrection {
 public:
  using LargestOutgoingCharSpeed = PlutoHlld::LargestOutgoingCharSpeed;
  using LargestIngoingCharSpeed = PlutoHlld::LargestIngoingCharSpeed;
  using FastOutgoingCharSpeed = PlutoHlld::FastOutgoingCharSpeed;
  using FastIngoingCharSpeed = PlutoHlld::FastIngoingCharSpeed;
  using InterfaceUnitNormal = PlutoHlld::InterfaceUnitNormal;
  using MetricFlatness = PlutoHlld::MetricFlatness;
  using InterfaceLapse = PlutoHlld::InterfaceLapse;
  using InterfaceShiftDotNormal = PlutoHlld::InterfaceShiftDotNormal;
  using InterfaceSpatialMetric = PlutoHlld::InterfaceSpatialMetric;

  struct MagneticFieldMagnitudeForHydro {
    static constexpr Options::String help = {
        "When |TildeB| is below this value on the whole face, the outer "
        "bounds use the hydro sound speed instead of the light speed (as in "
        "PlutoHlld)."};
    using type = double;
  };
  struct LightSpeedDensityCutoff {
    static constexpr Options::String help = {
        "When the density is below this value we just use the light speed for "
        "the characteristic speeds."};
    using type = double;
  };
  using options =
      tmpl::list<MagneticFieldMagnitudeForHydro, LightSpeedDensityCutoff>;
  static constexpr Options::String help = {
      "Experimental SR D2-R: analytical slow-mode response on refined PLUTO "
      "HLLD, with physical-input and original-face entropy checks. Ideal "
      "Gamma-law EOS (1 < Gamma <= 2) and a static mesh only; on a curved "
      "background only faces with zero normal shift (elsewhere HLL); "
      "otherwise conservative HLL. Common GLM Bn and GLM energy pairing are "
      "enabled."};

  PlutoD2R() = default;
  PlutoD2R(const PlutoD2R&) = default;
  PlutoD2R& operator=(const PlutoD2R&) = default;
  PlutoD2R(PlutoD2R&&) = default;
  PlutoD2R& operator=(PlutoD2R&&) = default;
  ~PlutoD2R() override = default;

  PlutoD2R(double magnetic_field_magnitude_for_hydro,
           double light_speed_density_cutoff);

  /// \cond
  explicit PlutoD2R(CkMigrateMessage* /*unused*/);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(PlutoD2R);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  // The same face fields as PlutoHlld; eligibility also checks the full metric.
  using dg_package_field_tags = tmpl::list<
      Tags::TildeD, Tags::TildeYe, Tags::TildeTau,
      Tags::TildeS<Frame::Inertial>, Tags::TildeB<Frame::Inertial>,
      Tags::TildePhi, ::Tags::NormalDotFlux<Tags::TildeD>,
      ::Tags::NormalDotFlux<Tags::TildeYe>,
      ::Tags::NormalDotFlux<Tags::TildeTau>,
      ::Tags::NormalDotFlux<Tags::TildeS<Frame::Inertial>>,
      ::Tags::NormalDotFlux<Tags::TildeB<Frame::Inertial>>,
      ::Tags::NormalDotFlux<Tags::TildePhi>, LargestOutgoingCharSpeed,
      LargestIngoingCharSpeed, FastOutgoingCharSpeed, FastIngoingCharSpeed,
      InterfaceUnitNormal, MetricFlatness, InterfaceLapse,
      InterfaceShiftDotNormal, InterfaceSpatialMetric,
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::SpatialVelocity<DataVector, 3>,
      hydro::Tags::Pressure<DataVector>, hydro::Tags::LorentzFactor<DataVector>,
      hydro::Tags::SpecificInternalEnergy<DataVector>>;
  using dg_package_data_temporary_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>,
      gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>>;
  using dg_package_data_primitive_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Temperature<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::LorentzFactor<DataVector>>;
  using dg_package_data_volume_tags =
      tmpl::list<hydro::Tags::GrmhdEquationOfState>;
  // The signal speeds are built in dg_package_data, per side, so
  // dg_boundary_terms needs nothing from the volume (as in Hll).
  using dg_boundary_terms_volume_tags = tmpl::list<>;

  double dg_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_d,
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_ye,
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> packaged_tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> packaged_tilde_b,
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_phi,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_d,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_ye,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_tilde_b,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_phi,
      gsl::not_null<Scalar<DataVector>*> packaged_largest_outgoing_char_speed,
      gsl::not_null<Scalar<DataVector>*> packaged_largest_ingoing_char_speed,
      gsl::not_null<Scalar<DataVector>*> packaged_fast_outgoing_char_speed,
      gsl::not_null<Scalar<DataVector>*> packaged_fast_ingoing_char_speed,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          packaged_interface_unit_normal,
      gsl::not_null<Scalar<DataVector>*> packaged_metric_flatness,
      gsl::not_null<Scalar<DataVector>*> packaged_interface_lapse,
      gsl::not_null<Scalar<DataVector>*> packaged_interface_shift_dot_normal,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
          packaged_interface_spatial_metric,
      gsl::not_null<Scalar<DataVector>*> packaged_rest_mass_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_spatial_velocity,
      gsl::not_null<Scalar<DataVector>*> packaged_pressure,
      gsl::not_null<Scalar<DataVector>*> packaged_lorentz_factor,
      gsl::not_null<Scalar<DataVector>*> packaged_specific_internal_energy,

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
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state)
      const;

  void dg_boundary_terms(
      gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_d,
      gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_ye,
      gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          boundary_correction_tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          boundary_correction_tilde_b,
      gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
      const Scalar<DataVector>& tilde_d_int,
      const Scalar<DataVector>& tilde_ye_int,
      const Scalar<DataVector>& tilde_tau_int,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_int,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
      const Scalar<DataVector>& tilde_phi_int,
      const Scalar<DataVector>& normal_dot_flux_tilde_d_int,
      const Scalar<DataVector>& normal_dot_flux_tilde_ye_int,
      const Scalar<DataVector>& normal_dot_flux_tilde_tau_int,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_s_int,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_b_int,
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
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_s_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_b_ext,
      const Scalar<DataVector>& normal_dot_flux_tilde_phi_ext,
      const Scalar<DataVector>& largest_outgoing_char_speed_ext,
      const Scalar<DataVector>& largest_ingoing_char_speed_ext,
      const Scalar<DataVector>& fast_outgoing_char_speed_ext,
      const Scalar<DataVector>& fast_ingoing_char_speed_ext,
      const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_ext,
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
      dg::Formulation dg_formulation) const;

 private:
  friend bool operator==(const PlutoD2R& lhs, const PlutoD2R& rhs);

  double magnetic_field_magnitude_for_hydro_{
      std::numeric_limits<double>::signaling_NaN()};
  double light_speed_density_cutoff_{
      std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const PlutoD2R& lhs, const PlutoD2R& rhs);
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
