// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/BoundaryCorrection.hpp"
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
/// \endcond

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
/*!
 * \brief GR-aware HLLC Riemann solver for pure hydrodynamics (B = 0), to
 * be built via the "two-scalar hat transform" of
 * `spectre_runs/kh_solver_comparison/HLLC_GR_ONF_PLAN.md` §4 around the SR
 * Mignone & Bodo (MB05) kernel that powers `Hllc`.
 *
 * \warning Session-1 skeleton. This class is currently a verbatim rename of
 * `Hllc` (same algorithm, same behaviour). The GR-specific transform will
 * land in subsequent sessions.
 *
 * The plan pivoted (in response to a full-plan review) from the White,
 * Stone & Gammie (WSG16) local-orthonormal-frame tetrad wrapper to a
 * two-scalar form that is algebraically equivalent for pure hydrodynamics
 * and materially cheaper to implement in SpECTRE's DG interface structure.
 * At each interface point, defining
 * \f{align*}
 *   \hat F(U) &= \frac{F(U) + \beta^n U}{\alpha}, &
 *   \nu       &= \frac{\lambda + \beta^n}{\alpha},
 * \f}
 * the hatted system satisfies \f$\hat F(\tilde E) = \tilde M\f$
 * identically, restoring MB05's SR identity \f$F(E) = m^x\f$. The existing
 * MB05 quadratic + star-state math runs on the hatted quantities and the
 * inverse transform recovers the coordinate-frame flux.
 *
 * Per face:
 * -# Average the interior/exterior 3+1 geometry
 *    (\f$\alpha, \beta^i, \gamma_{ij}, n_i\f$) inside `dg_boundary_terms`
 *    — **not** per-side tetrads. This is the single-Lorentz-frame
 *    requirement for the Riemann problem to be well-posed.
 * -# Hat-transform L and R state, flux, and wave-speed inputs.
 * -# Run the existing MB05 kernel on the hatted quantities.
 * -# Inverse-transform the numerical flux.
 * -# Region selection stays in the coordinate frame; a mesh-velocity aware
 *    \f$\beta^n_{\rm eff} = \beta^n + n \cdot v_{\rm mesh}\f$ feeds the
 *    transform.
 *
 * The design makes the scheme **exact** on curved backgrounds where the
 * current `Hllc` is only an SR-inspired approximation (via MB05's identity
 * \f$F(E) = m^x\f$, which breaks when \f$\alpha \ne 1\f$ or
 * \f$\beta^n \ne 0\f$). The flat-space reduction is bit-identical to
 * `Hllc` — see the plan doc §9 rung 1.
 *
 * The full WSG16 tetrad wrapper (plan §5) is deferred to a later session
 * as (a) a curved-background cross-check against the two-scalar form
 * (plan §9 rung 6) and (b) the natural HLLD-extension architecture — for
 * HLLD the transverse-B structure needs the full 4-velocity basis and
 * the two-scalar reduction breaks.
 *
 * \note Restricted to \f$\mathbf{B} = 0\f$. For the divergence-cleaning
 * variables \f$\tilde{B}^i\f$ and \f$\tilde\Phi\f$ the HLL flux is used,
 * matching `Hllc`. Extending to MHD requires HLLD (Mignone, Ugliano & Bodo
 * 2009) as the inner kernel.
 */
class HllcGr final : public evolution::BoundaryCorrection {
 public:
  struct LargestOutgoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct LargestIngoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  /// \f$M = n^i \tilde S_i\f$, the normal projection of the covariant
  /// densitized momentum.
  struct NormalDotTildeS : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  /// \f$F(M) = n^i n_j F^j(\tilde S_i)\f$, the normal-normal projection of
  /// the momentum flux (i.e., MB05's \f$F(m_x) = m_x v_x + p\f$ analog).
  struct NormalDotFluxNormalDotTildeS : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  /// Coordinate advection speed at the face
  /// \f$u = \alpha v^n - \beta^n\f$; MB05's \f$v^x\f$ analog.
  struct AdvectionSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  /// Pressure-flux coefficient \f$\tilde p = \alpha \sqrt{\gamma}\, p\f$;
  /// MB05's \f$p\f$ analog.
  struct PressureFluxCoefficient : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  /// The (interior) outward normal covector \f$n_i\f$ at the face, needed
  /// to reconstruct the covariant star momentum via the flux ansatz
  /// \f$F^*(\tilde S_i) = \tilde S^*_i \lambda^* + \tilde p^*\, n_i\f$.
  struct InterfaceNormalCovector : db::SimpleTag {
    using type = tnsr::i<DataVector, 3, Frame::Inertial>;
  };

  struct MagneticFieldMagnitudeForHydro {
    static constexpr Options::String help = {
        "When the magnetic field is below this value we use the hydro "
        "characteristic speeds."};
    using type = double;
  };
  struct LightSpeedDensityCutoff {
    static constexpr Options::String help = {
        "When the density is below this value we just use the light speed "
        "for the characteristic speeds."};
    using type = double;
  };
  using options =
      tmpl::list<MagneticFieldMagnitudeForHydro, LightSpeedDensityCutoff>;
  static constexpr Options::String help = {
      "GR-extension HLLC boundary correction for the GRMHD system, using the "
      "two-scalar hat transform F_hat=(F+beta^n U)/alpha, "
      "nu=(lambda+beta^n)/alpha around the SR MB05 kernel. "
      "Restricted to B = 0; magnetic sector uses HLL fluxes. "
      "Session-1 skeleton: currently identical to Hllc; the hat transform "
      "will be added in subsequent sessions."};

  HllcGr() = default;
  HllcGr(const HllcGr&) = default;
  HllcGr& operator=(const HllcGr&) = default;
  HllcGr(HllcGr&&) = default;
  HllcGr& operator=(HllcGr&&) = default;
  ~HllcGr() override = default;

  HllcGr(double magnetic_field_magnitude_for_hydro,
         double light_speed_density_cutoff);

  /// \cond
  explicit HllcGr(CkMigrateMessage* /*unused*/);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(HllcGr);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  using dg_package_field_tags =
      tmpl::list<Tags::TildeD, Tags::TildeYe, Tags::TildeTau,
                 Tags::TildeS<Frame::Inertial>, Tags::TildeB<Frame::Inertial>,
                 Tags::TildePhi, ::Tags::NormalDotFlux<Tags::TildeD>,
                 ::Tags::NormalDotFlux<Tags::TildeYe>,
                 ::Tags::NormalDotFlux<Tags::TildeTau>,
                 ::Tags::NormalDotFlux<Tags::TildeS<Frame::Inertial>>,
                 ::Tags::NormalDotFlux<Tags::TildeB<Frame::Inertial>>,
                 ::Tags::NormalDotFlux<Tags::TildePhi>,
                 LargestOutgoingCharSpeed, LargestIngoingCharSpeed,
                 NormalDotTildeS, NormalDotFluxNormalDotTildeS, AdvectionSpeed,
                 PressureFluxCoefficient, InterfaceNormalCovector>;
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
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_tilde_s,
      gsl::not_null<Scalar<DataVector>*>
          packaged_normal_dot_flux_normal_dot_tilde_s,
      gsl::not_null<Scalar<DataVector>*> packaged_advection_speed,
      gsl::not_null<Scalar<DataVector>*> packaged_pressure_flux_coefficient,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          packaged_interface_normal_covector,

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
      const Scalar<DataVector>& /*specific_internal_energy*/,
      const Scalar<DataVector>& pressure,
      const Scalar<DataVector>& /*lorentz_factor*/,

      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state)
      const;

  static void dg_boundary_terms(
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
      const Scalar<DataVector>& normal_dot_tilde_s_int,
      const Scalar<DataVector>& normal_dot_flux_normal_dot_tilde_s_int,
      const Scalar<DataVector>& advection_speed_int,
      const Scalar<DataVector>& pressure_flux_coefficient_int,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          interface_normal_covector_int,
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
      const Scalar<DataVector>& normal_dot_tilde_s_ext,
      const Scalar<DataVector>& normal_dot_flux_normal_dot_tilde_s_ext,
      const Scalar<DataVector>& advection_speed_ext,
      const Scalar<DataVector>& pressure_flux_coefficient_ext,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          interface_normal_covector_ext,
      dg::Formulation dg_formulation);

 private:
  friend bool operator==(const HllcGr& lhs, const HllcGr& rhs);

  double magnetic_field_magnitude_for_hydro_{
      std::numeric_limits<double>::signaling_NaN()};
  double light_speed_density_cutoff_{
      std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const HllcGr& lhs, const HllcGr& rhs);
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
