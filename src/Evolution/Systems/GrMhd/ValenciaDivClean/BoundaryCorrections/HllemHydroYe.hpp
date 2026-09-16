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
 * \brief HLL + middle-block anti-diffusion for the hydro+\f$Y_e\f$ subsystem.
 *
 * Baseline HLL on all six evolved variables plus anti-diffusion of the
 * four-dimensional degenerate middle eigenspace of the hydro+\f$Y_e\f$
 * characteristic decomposition (entropy + two shear modes + composition),
 * via a basis-independent complementary projector
 *
 * \f{align*}
 *   P_\text{mid} = I - P_+ - P_- ,
 * \f}
 *
 * where \f$P_\pm = R_\pm \otimes L_\pm / (L_\pm \cdot R_\pm)\f$ are the
 * projectors onto the two acoustic eigenspaces of the normal-flux Jacobian.
 * The composition-pressure coupling
 * \f$\zeta = (\partial p / \partial Y_e)_{\rho,\epsilon}\f$ enters through
 * \f$P_\pm(\zeta) \to P_\text{mid}(\zeta)\f$ automatically; no composition
 * wave is treated as a named object.
 *
 * The corrected flux is
 *
 * \f{align*}
 *   G_\text{HLLEM} = G_\text{HLL}
 *     - \frac{S_L S_R}{S_R - S_L}\,
 *       \delta_\text{mid}\,
 *       P_\text{mid}\,
 *       (U_R - U_L) ,
 * \f}
 *
 * with
 * \f$\delta_\text{mid} = 1 - \min(0,\lambda_\text{mid})/S_L
 *                          - \max(0,\lambda_\text{mid})/S_R\f$.
 *
 * ## Curved backgrounds
 *
 * The eigensystem returned by
 * `grmhd::ValenciaDivClean::characteristic_eigenvectors_hydro` is the
 * eigensystem of the **Eulerian-frame** normal-flux Jacobian
 * \f$A_\text{Eul}\f$ for a general spatial metric \f$\gamma_{ij}\f$; its
 * degenerate eigenvalue is \f$\nu_\text{mid} = v\cdot\hat n\f$. The
 * coordinate-frame (Valencia) Jacobian is
 *
 * \f{align*}
 *   A_\text{coord} = \alpha\,A_\text{Eul} - \beta^n_\text{eff}\,I ,
 * \f}
 *
 * so the left and right eigenvectors -- and therefore \f$P_\pm\f$ and
 * \f$P_\text{mid}\f$ -- are *identical* in the two frames, while the
 * eigenvalues map as \f$\lambda = \alpha\nu - \beta^n_\text{eff}\f$
 * (`HatTransform::inverse_speed`). Densitization by \f$\sqrt\gamma\f$
 * rescales every slot of the state vector by the same factor and so leaves
 * the projectors unchanged.
 *
 * The HLLEM flux above is therefore used **verbatim in curved space**,
 * evaluated with coordinate-frame \f$S_L\f$, \f$S_R\f$ and
 * \f$\lambda_\text{mid} = \alpha\,(v\cdot\hat n) -
 * \beta^n_\text{eff}\f$. This is not an approximation: writing the
 * anti-diffusion as an integral of the HLLEM fan profile,
 * \f$G(\xi) = F^\text{HLL} - \xi U^\text{HLL} - \sum_k
 * \int_{S_L}^{\xi}\Phi_k\f$, and sampling in the Eulerian frame at the
 * ray \f$\xi = w_\text{face} = \beta^n_\text{eff}/\alpha\f$ at which the
 * fixed DG interface moves, the lapse and shift cancel exactly against the
 * hatted speeds and reproduce the flat-space \f$\delta_\text{mid}\f$
 * formula in coordinate-frame speeds. Note that the *naive* reading -- run
 * the flat-space algebra on hatted speeds \f$\nu\f$ while still sampling at
 * \f$\xi = 0\f$ -- is **wrong**, because it samples the wrong ray.
 *
 * The outer bounds \f$S_{L,R}\f$ are the **per-side** fluid characteristic
 * speeds packaged in `dg_package_data`, combined as
 * \f$S_R = \max(0, \lambda_+^{L}, \lambda_+^{R})\f$ and
 * \f$S_L = \min(0, \lambda_-^{L}, \lambda_-^{R})\f$ (Recipe A, Davis 1988)
 * -- exactly as in `Hll`. HLLEM requires \f$S_{L,R}\f$ to be an *upper bound*
 * on the true wave speeds estimated from the left and right **input** states;
 * see Mattia & Mignone 2021 (arXiv:2111.09369), sec. "HLL Formulation". Only
 * the anti-diffusion term uses the averaged interface state, and only through
 * \f$R_\pm\f$, \f$L_\pm\f$ and \f$\lambda_\text{mid}\f$.
 *
 * \note An earlier revision evaluated \f$S_{L,R}\f$ at the averaged state too
 * ("Recipe B"), and argued \f$\delta_\text{mid}\in[0,1]\f$ from
 * \f$\lambda_\text{mid}\f$ and \f$S_{L,R}\f$ sharing that state. Recipe B
 * under-bounds the fan at strongly-asymmetric interfaces, violating the
 * Harten-Lax-van Leer premise, and was removed. **That proof of the
 * \f$\delta_\text{mid}\f$ range died with it**, but the property survives
 * for an independent reason: a sound-speed margin.
 * \f$\lambda_+(v) = (v_n + c_s)/(1 + v_n c_s) > v_n\f$ at every state, and
 * \f$v_n(*)\f$ lies between \f$v_n(L)\f$ and \f$v_n(R)\f$, so
 * \f$S_R > v_n(*) = \lambda_\text{mid}\f$ (and symmetrically for
 * \f$S_L\f$). Measured over 4000 random subsonic interfaces with these
 * bounds: no excursions, \f$\delta_\text{mid}\in[0.216, 1]\f$. There is
 * deliberately **no clamp** -- a clamp would mask a genuinely bad bound
 * estimate -- so the guarantee rests on that margin and would fail for any
 * bound lacking it.
 *
 * Magnetized states with
 * \f$|B| \ge \text{MagneticFieldMagnitudeForHydro}\f$ still bypass the
 * anti-diffusion entirely: the packaged sound speed is left at its zero
 * sentinel, neither the averaged interface state nor the eigensystem is
 * built, and the class reduces to plain HLL on every slot (with the
 * light-speed packaged bounds, since the hydro-speed branch is also
 * skipped).
 *
 * See `notes/projects/active/hllem_hydroye/design.md` for the full design and
 * `notes/projects/active/hllem_hydroye/phase5_gr_log.md` for the curved-space
 * derivation and its verification. (Both carry errata as of 2026-09-16: the
 * design's Recipe-B bounds and its "delta_pm == 0" premise are superseded --
 * see `notes/projects/active/hllem_hydroye/cpp_findings_2026-09-16.md`.)
 *
 * ---- HLL baseline (as inherited from `Hll`) ----
 *
 * Let \f$U\f$ be the evolved variable, \f$F^i\f$ the flux, and \f$n_i\f$ be
 * the outward directed unit normal to the interface. Denoting
 * \f$F := n_i F^i\f$, the HLL boundary correction is \cite Harten1983
 *
 * \f{align*}
 * G_\text{HLL} = \frac{\lambda_\text{max} F_\text{int} +
 * \lambda_\text{min} F_\text{ext}}{\lambda_\text{max} - \lambda_\text{min}}
 * - \frac{\lambda_\text{min}\lambda_\text{max}}{\lambda_\text{max} -
 *   \lambda_\text{min}} \left(U_\text{int} - U_\text{ext}\right)
 * \f}
 *
 * where "int" and "ext" stand for interior and exterior.
 * \f$\lambda_\text{min}\f$ and \f$\lambda_\text{max}\f$ are defined as
 *
 * \f{align*}
 * \lambda_\text{min} &=
 * \text{min}\left(\lambda^{-}_\text{int},-\lambda^{+}_\text{ext}, 0\right) \\
 * \lambda_\text{max} &=
 * \text{max}\left(\lambda^{+}_\text{int},-\lambda^{-}_\text{ext}, 0\right)
 * \f}
 *
 * where \f$\lambda^{+}\f$ (\f$\lambda^{-}\f$) is the largest characteristic
 * speed in the outgoing (ingoing) direction. Note the minus signs in front of
 * \f$\lambda^{\pm}_\text{ext}\f$, which is because an outgoing speed w.r.t. the
 * neighboring element is an ingoing speed w.r.t. the local element, and vice
 * versa. Similarly, the \f$F_{\text{ext}}\f$ term in \f$G_\text{HLL}\f$ has a
 * positive sign because the outward directed normal of the neighboring element
 * has the opposite sign, i.e. \f$n_i^{\text{ext}}=-n_i^{\text{int}}\f$.
 *
 * The characteristic/signal speeds are given in the documentation for
 * `grmhd::ValenciaDivClean::characteristic_speeds()`. Since the fluid is
 * travelling slower than the speed of light, the speeds we are interested in
 * are
 *
 * \f{align*}{
 *   \lambda^{\pm}&=\pm\alpha-\beta^i n_i,
 * \f}
 *
 * which correspond to the divergence cleaning field.
 *
 * \note
 * - In the strong form the `dg_boundary_terms` function returns
 *   \f$G - F_\text{int}\f$
 * - For either \f$\lambda_\text{min} = 0\f$ or \f$\lambda_\text{max} = 0\f$
 *   (i.e. all characteristics move in the same direction) the HLL boundary
 *   correction reduces to pure upwinding.
 * - Some references use \f$S\f$ instead of \f$\lambda\f$ for the
 *   signal/characteristic speeds
 * - It may be possible to use the slower speeds for the magnetic field and
 *   fluid part of the system in order to make the flux less dissipative for
 *   those variables.
 */
class HllemHydroYe final : public evolution::BoundaryCorrection {
 public:
  struct LargestOutgoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct LargestIngoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  /// Interface unit normal (covector), used to project the normal magnetic
  /// field for the divergence-cleaning (Phi, B_n) subsystem.
  struct InterfaceUnitNormal : db::SimpleTag {
    using type = tnsr::i<DataVector, 3, Frame::Inertial>;
  };
  /// Lapse \f$\alpha\f$ at the interface. Together with `ShiftDotNormal`
  /// and `InterfaceSpatialMetric` this is the 3+1 geometry the interface
  /// frame is built from (see `HatTransform.hpp`); it maps the Eulerian-frame
  /// characteristic speeds of `characteristic_eigenvectors_hydro` into the
  /// coordinate frame.
  struct LapseAtInterface : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  /// \f$\beta^n_\text{eff} = \beta^i n_i + n \cdot v_\text{mesh}\f$ at
  /// the interface, signed with the interior's outward normal (averaged
  /// antisymmetrically in `dg_boundary_terms`). The mesh velocity is folded
  /// in so that \f$\lambda = \alpha\nu - \beta^n_\text{eff}\f$ agrees
  /// with the packaged characteristic speeds, which already subtract
  /// \f$n\cdot v_\text{mesh}\f$.
  struct ShiftDotNormal : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  /// Spatial metric \f$\gamma_{ij}\f$ at the interface. Needed to build the
  /// hydro eigensystem (which raises and lowers indices, and constructs the
  /// two tangent one-forms) and to renormalize the averaged interface normal.
  struct InterfaceSpatialMetric : db::SimpleTag {
    using type = tnsr::ii<DataVector, 3, Frame::Inertial>;
  };

  struct MagneticFieldMagnitudeForHydro {
    static constexpr Options::String help = {
        "When the magnetic field is below this value we use the hydro "
        "characteristic speeds."};
    using type = double;
  };
  struct LightSpeedDensityCutoff {
    static constexpr Options::String help = {
        "When the density is below this value we just use the light speed for "
        "the characteristic speeds."};
    using type = double;
  };
  struct RestoreMiddleBlock {
    static constexpr Options::String help = {
        "If true, apply the middle-block anti-diffusion "
        "-(S_L S_R)/(S_R - S_L) * delta_mid * P_mid * (U_R - U_L) on top of "
        "the HLL baseline. If false, the class reduces to plain Hll -- "
        "verified bit-identical to it on both a uniform state and a Y_e "
        "contact."};
    using type = bool;
  };
  /// \deprecated RETIRED 2026-09-16 — leave at the default `True`.
  ///
  /// Setting this to `False` does not produce a physical limit: it zeroes
  /// zeta while the EoS still supplies zeta != 0, so the resulting vectors
  /// are not an eigenbasis of any flux Jacobian. The comparison it was built
  /// for (P_mid(zeta) vs P_mid(0), design.md 13.5) is not a correctness test
  /// either — both sides come from the same code and neither is ground truth.
  ///
  /// It is also a blunter operation than its name suggests. The override
  /// fires *before* `zeta_max_abs` in `characteristic_eigenvectors_hydro`
  /// (`Characteristics.cpp:1621`), so it additionally flips R3's coupling off
  /// and R4/L3/L4 to their degenerate forms. That makes it NOT equivalent to
  /// the "fake zeta" patch in `scripts/projector_diagnostic.py`, which
  /// touches only L_pm. Harmless for `P_mid` itself, since R_pm/L_pm carry no
  /// such branch.
  ///
  /// The option is kept, rather than deleted, only so that the 29 archived
  /// run yamls that set it still parse — SpECTRE hard-errors on unknown
  /// options, and editing a completed run's yaml would misrepresent what ran.
  /// It has been removed from `spectre_runs/_templates/`. Its one legitimate
  /// use is bug localisation ("does this failure depend on zeta at all?"),
  /// where nothing is claimed to be physical.
  struct UsePhysicalZeta {
    static constexpr Options::String help = {
        "RETIRED -- leave at True. False does not give a physical zeta = 0 "
        "limit (the EoS still supplies zeta != 0, so the basis is not an "
        "eigenbasis of any flux Jacobian) and it also degenerates R3/R4/L3/L4, "
        "so it is not a clean zeta switch. Kept only so archived yamls parse. "
        "Diagnostic use only. Has no effect while RestoreMiddleBlock is "
        "false."};
    using type = bool;
  };
  using options =
      tmpl::list<MagneticFieldMagnitudeForHydro, LightSpeedDensityCutoff,
                 RestoreMiddleBlock, UsePhysicalZeta>;
  static constexpr Options::String help = {
      "HLL + middle-block anti-diffusion for the hydro+Y_e subsystem. Works "
      "on curved backgrounds. Outer HLL bounds are the per-side (Davis) "
      "speeds, as in Hll; only the anti-diffusion uses the averaged interface "
      "state. With RestoreMiddleBlock=false the class reduces to plain Hll."};

  HllemHydroYe() = default;
  HllemHydroYe(const HllemHydroYe&) = default;
  HllemHydroYe& operator=(const HllemHydroYe&) = default;
  HllemHydroYe(HllemHydroYe&&) = default;
  HllemHydroYe& operator=(HllemHydroYe&&) = default;
  ~HllemHydroYe() override = default;

  HllemHydroYe(double magnetic_field_magnitude_for_hydro,
               double light_speed_density_cutoff, bool restore_middle_block,
               bool use_physical_zeta);

  /// \cond
  explicit HllemHydroYe(CkMigrateMessage* /*unused*/);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(HllemHydroYe);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  using dg_package_field_tags = tmpl::list<
      Tags::TildeD, Tags::TildeYe, Tags::TildeTau,
      Tags::TildeS<Frame::Inertial>, Tags::TildeB<Frame::Inertial>,
      Tags::TildePhi, ::Tags::NormalDotFlux<Tags::TildeD>,
      ::Tags::NormalDotFlux<Tags::TildeYe>,
      ::Tags::NormalDotFlux<Tags::TildeTau>,
      ::Tags::NormalDotFlux<Tags::TildeS<Frame::Inertial>>,
      ::Tags::NormalDotFlux<Tags::TildeB<Frame::Inertial>>,
      ::Tags::NormalDotFlux<Tags::TildePhi>, LargestOutgoingCharSpeed,
      LargestIngoingCharSpeed, InterfaceUnitNormal, LapseAtInterface,
      ShiftDotNormal, InterfaceSpatialMetric,
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::ElectronFraction<DataVector>,
      hydro::Tags::SoundSpeedSquared<DataVector>,
      hydro::Tags::Temperature<DataVector>,
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
  // The equation of state is needed in dg_boundary_terms for the middle-block
  // eigensystem only: the on-EOS (p_avg, h_avg) at the averaged (rho, eps,
  // Y_e). The outer bounds need no EOS call -- they are the per-side speeds
  // packaged in dg_package_data. So this tag is unused when
  // RestoreMiddleBlock is false.
  using dg_boundary_terms_volume_tags =
      tmpl::list<hydro::Tags::GrmhdEquationOfState>;

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
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          packaged_interface_unit_normal,
      gsl::not_null<Scalar<DataVector>*> packaged_lapse_at_interface,
      gsl::not_null<Scalar<DataVector>*> packaged_shift_dot_normal,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
          packaged_spatial_metric,
      gsl::not_null<Scalar<DataVector>*> packaged_rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> packaged_electron_fraction,
      gsl::not_null<Scalar<DataVector>*> packaged_sound_speed_squared,
      gsl::not_null<Scalar<DataVector>*> packaged_temperature,
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

  // Non-static (unlike Hll) so the body can consult restore_middle_block_
  // and use_physical_zeta_.
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
      const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_int,
      const Scalar<DataVector>& lapse_at_interface_int,
      const Scalar<DataVector>& shift_dot_normal_int,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric_int,
      const Scalar<DataVector>& rest_mass_density_int,
      const Scalar<DataVector>& electron_fraction_int,
      const Scalar<DataVector>& sound_speed_squared_int,
      const Scalar<DataVector>& temperature_int,
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
      const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_ext,
      const Scalar<DataVector>& lapse_at_interface_ext,
      const Scalar<DataVector>& shift_dot_normal_ext,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric_ext,
      const Scalar<DataVector>& rest_mass_density_ext,
      const Scalar<DataVector>& electron_fraction_ext,
      const Scalar<DataVector>& sound_speed_squared_ext,
      const Scalar<DataVector>& temperature_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_ext,
      const Scalar<DataVector>& pressure_ext,
      const Scalar<DataVector>& lorentz_factor_ext,
      const Scalar<DataVector>& specific_internal_energy_ext,
      dg::Formulation dg_formulation,
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state)
      const;

 private:
  friend bool operator==(const HllemHydroYe& lhs, const HllemHydroYe& rhs);

  double magnetic_field_magnitude_for_hydro_{
      std::numeric_limits<double>::signaling_NaN()};
  double light_speed_density_cutoff_{
      std::numeric_limits<double>::signaling_NaN()};
  bool restore_middle_block_{false};
  bool use_physical_zeta_{true};
};
bool operator!=(const HllemHydroYe& lhs, const HllemHydroYe& rhs);
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
