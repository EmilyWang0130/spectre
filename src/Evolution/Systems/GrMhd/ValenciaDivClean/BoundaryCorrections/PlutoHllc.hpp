// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>
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
/// Which of PLUTO v4.4's two relativistic-MHD HLLC solvers `PlutoHllc` calls.
enum class PlutoHllcVariant {
  /// `Src/RMHD/hllc_mb.c`: Mignone & Bodo 2006 (MNRAS 368, 1040,
  /// doi:10.1111/j.1365-2966.2006.10162.x). PLUTO's default `hllc`.
  MignoneBodo,
  /// `Src/RMHD/hllc_kb.c`, which its header attributes to Balsara & Kim 2016
  /// (JCP 312, 357, doi:10.1016/j.jcp.2016.02.001; the header misprints the
  /// pages as 1-28).
  KimBalsara
};
std::ostream& operator<<(std::ostream& os, PlutoHllcVariant variant);

/*!
 * \brief PLUTO v4.4's relativistic-MHD HLLC Riemann solvers, evaluated by
 * PLUTO's own compiled code through the same shim as `PlutoHlld`.
 *
 * HLLC restores the contact wave on top of the outer fast waves: two star
 * states separated by a contact, three waves in all. Two variants, both from
 * PLUTO's `Src/RMHD/` (see `PlutoHllcVariant`):
 *
 * - `MignoneBodo` (`hllc_mb.c`). For \f$|B^x| < 10^{-6}\f$ it uses the paper's
 *   \f$B^x \to 0\f$ limit (their Eq. 47), in which the tangential field jumps
 *   across the contact like the density does, and then agrees with PLUTO's HLLD
 *   to round-off. Otherwise (their Eq. 42) both star states carry the HLL
 *   average of the field, and where the resulting star velocity is
 *   superluminal it falls back to the HLL flux. That fallback is common: it
 *   fires on the Balsara-1 (relativistic Brio-Wu) interface, for instance,
 *   where this solver's flux is HLL's to every digit.
 * - `KimBalsara` (`hllc_kb.c`). Solves for the star states iteratively (at
 *   most 15 Newton steps), starting from the primitive state recovered from
 *   the HLL average. Its star field is always the HLL average, also at
 *   \f$B^x = 0\f$, so there it does not split the tangential field across the
 *   contact and its tangential-field flux is HLL's.
 *
 * Everything outside PLUTO's call is `PlutoHlld`'s own code
 * (`detail::pluto_dg_boundary_terms`), so the two share their restrictions
 * and their fallbacks: ideal-gas EoS only; flat space only (curved faces get
 * the HLL flux); the divergence-cleaning field and the normal magnetic field
 * travel at the light speed through HLL, as PLUTO has no GLM here; `TildeYe`
 * uses HLL with the fast bounds; and any point PLUTO returns non-finite for
 * keeps the HLL flux.
 */
class PlutoHllc final : public evolution::BoundaryCorrection {
 public:
  using LargestOutgoingCharSpeed = PlutoHlld::LargestOutgoingCharSpeed;
  using LargestIngoingCharSpeed = PlutoHlld::LargestIngoingCharSpeed;
  using FastOutgoingCharSpeed = PlutoHlld::FastOutgoingCharSpeed;
  using FastIngoingCharSpeed = PlutoHlld::FastIngoingCharSpeed;
  using InterfaceUnitNormal = PlutoHlld::InterfaceUnitNormal;
  using MetricFlatness = PlutoHlld::MetricFlatness;

  struct Variant {
    using type = PlutoHllcVariant;
    static constexpr Options::String help = {
        "Which of PLUTO's RMHD HLLC solvers to call: MignoneBodo (hllc_mb.c, "
        "Mignone & Bodo 2006, PLUTO's default) or KimBalsara (hllc_kb.c, "
        "Balsara & Kim 2016)."};
  };
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
  using options = tmpl::list<Variant, MagneticFieldMagnitudeForHydro,
                             LightSpeedDensityCutoff>;
  static constexpr Options::String help = {
      "The relativistic-MHD HLLC Riemann solver, evaluated by PLUTO v4.4's own "
      "compiled implementation (Mignone & Bodo 2006, or Balsara & Kim 2016). "
      "Ideal-gas EoS and flat space only; elsewhere it falls back to HLL."};

  PlutoHllc() = default;
  PlutoHllc(const PlutoHllc&) = default;
  PlutoHllc& operator=(const PlutoHllc&) = default;
  PlutoHllc(PlutoHllc&&) = default;
  PlutoHllc& operator=(PlutoHllc&&) = default;
  ~PlutoHllc() override = default;

  PlutoHllc(PlutoHllcVariant variant, double magnetic_field_magnitude_for_hydro,
            double light_speed_density_cutoff);

  /// \cond
  explicit PlutoHllc(CkMigrateMessage* /*unused*/);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(PlutoHllc);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  // Exactly what PlutoHlld packages: dg_package_data forwards to it.
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
      InterfaceUnitNormal, MetricFlatness,
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::SpatialVelocity<DataVector, 3>,
      hydro::Tags::Pressure<DataVector>, hydro::Tags::LorentzFactor<DataVector>,
      hydro::Tags::SpecificInternalEnergy<DataVector>>;
  using dg_package_data_temporary_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>>;
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
      const Scalar<DataVector>& rest_mass_density_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_ext,
      const Scalar<DataVector>& pressure_ext,
      const Scalar<DataVector>& lorentz_factor_ext,
      const Scalar<DataVector>& specific_internal_energy_ext,
      dg::Formulation dg_formulation) const;

 private:
  friend bool operator==(const PlutoHllc& lhs, const PlutoHllc& rhs);

  PlutoHllcVariant variant_{PlutoHllcVariant::MignoneBodo};
  double magnetic_field_magnitude_for_hydro_{
      std::numeric_limits<double>::signaling_NaN()};
  double light_speed_density_cutoff_{
      std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const PlutoHllc& lhs, const PlutoHllc& rhs);
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections

/// \cond
template <>
struct Options::create_from_yaml<
    grmhd::ValenciaDivClean::BoundaryCorrections::PlutoHllcVariant> {
  template <typename Metavariables>
  static grmhd::ValenciaDivClean::BoundaryCorrections::PlutoHllcVariant create(
      const Options::Option& options) {
    return create<void>(options);
  }
};
template <>
grmhd::ValenciaDivClean::BoundaryCorrections::PlutoHllcVariant
Options::create_from_yaml<
    grmhd::ValenciaDivClean::BoundaryCorrections::PlutoHllcVariant>::
    create<void>(const Options::Option& options);
/// \endcond
