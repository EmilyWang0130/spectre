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
 * \brief An HLLC Riemann solver for pure hydrodynamics (B = 0).
 *
 * Restores the contact wave that HLL averages away. Follows Mignone & Bodo
 * 2005 (MNRAS 364, 126) \cite MignoneBodo2005: replaces HLL's single star
 * region with two star states \f$U^*_L\f$, \f$U^*_R\f$ separated by a contact
 * moving at speed \f$\lambda^*\f$, across which the normal velocity and total
 * pressure are continuous.
 *
 * Algorithm (per face):
 * -# Compute outer signal speeds \f$\lambda_L, \lambda_R\f$ as in HLL.
 * -# Trivial branches: if \f$\lambda_L \ge 0\f$ use \f$F_L\f$; if
 *    \f$\lambda_R \le 0\f$ use \f$F_R\f$.
 * -# Otherwise form the HLL averages of \f$E, m_x\f$ and their fluxes,
 *    then solve the MB05 quadratic (Eq. 18) for the contact speed
 *    \f{align*}
 *      F^E_{\rm hll} \, (\lambda^*)^2
 *        - (E_{\rm hll} + F^{m_x}_{\rm hll})\, \lambda^*
 *        + m^x_{\rm hll} = 0,
 *    \f}
 *    taking the minus-sign root (the only one satisfying
 *    \f$\lambda_L \le \lambda^* \le \lambda_R\f$; see MB05 Appendix A).
 * -# Star pressure from MB05 Eq. 17 (evaluated on either side; the
 *    quadratic guarantees they agree):
 *    \f{align*}
 *      A_\alpha &= \lambda_\alpha E_\alpha - m^x_\alpha,\\
 *      B_\alpha &= m^x_\alpha (\lambda_\alpha - v^x_\alpha) - p_\alpha,\\
 *      p^* &= \frac{A_\alpha \lambda^* - B_\alpha}
 *                  {1 + \lambda_\alpha \lambda^*}.
 *    \f}
 * -# Star conserved states from MB05 Eq. 16.
 * -# Star fluxes via Rankine-Hugoniot
 *    \f$F^*_\alpha = F_\alpha + \lambda_\alpha (U^*_\alpha - U_\alpha)\f$,
 *    then select the appropriate region based on the sign of \f$\lambda^*\f$.
 *
 * The general-relativistic extension identifies MB05's \f$v^x\f$ with the
 * coordinate advection speed at the face
 * \f$u = \alpha v^n - \beta^n\f$, MB05's \f$p\f$ with the pressure-flux
 * coefficient \f$\tilde{p} = \alpha \sqrt{\gamma}\, p\f$, and MB05's
 * \f$m^x, E\f$ with the normal-projected densitized momentum
 * \f$M = n^i \tilde{S}_i\f$ and total energy
 * \f$\tilde{E} = \tilde\tau + \tilde D\f$. This substitution keeps the
 * MB05 SR quadratic and Eq. 17 in their original form. The result is
 * **exact for a Minkowski background** (\f$\alpha=1, \beta^i=0,
 * \sqrt{\gamma}=1\f$) and gives a well-defined but strictly speaking
 * approximate algorithm on curved backgrounds — the SR identity
 * \f$F(E)=m^x\f$ becomes \f$F(\tilde E)=\alpha M - \beta^n \tilde E\f$
 * in GR, so the star-state pressure and contact speed do not exactly
 * reduce to \f$p_L = p_R\f$ and \f$\lambda^* = u\f$ for an identical
 * L = R state when \f$\alpha \ne 1\f$ or \f$\beta^n \ne 0\f$. Extending
 * the derivation to the full GR identity is a follow-up; the flat-
 * background reduction is what the Kelvin-Helmholtz application needs.
 *
 * \note Restricted to \f$\mathbf{B} = 0\f$. For the divergence-cleaning
 * variables \f$\tilde{B}^i\f$ and \f$\tilde\Phi\f$ the HLL flux is used,
 * which is the standard practice for hybrid HLLC-MHD schemes (Mignone,
 * Ugliano & Bodo 2009 \cite MignoneUglianoBodo2009).
 */
class Hllc final : public evolution::BoundaryCorrection {
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
      "Computes the HLLC boundary correction term for the GRMHD system. "
      "Restricted to B = 0; magnetic sector uses HLL fluxes. Follows "
      "Mignone & Bodo 2005."};

  Hllc() = default;
  Hllc(const Hllc&) = default;
  Hllc& operator=(const Hllc&) = default;
  Hllc(Hllc&&) = default;
  Hllc& operator=(Hllc&&) = default;
  ~Hllc() override = default;

  Hllc(double magnetic_field_magnitude_for_hydro,
       double light_speed_density_cutoff);

  /// \cond
  explicit Hllc(CkMigrateMessage* /*unused*/);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Hllc);  // NOLINT
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
  friend bool operator==(const Hllc& lhs, const Hllc& rhs);

  double magnetic_field_magnitude_for_hydro_{
      std::numeric_limits<double>::signaling_NaN()};
  double light_speed_density_cutoff_{
      std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const Hllc& lhs, const Hllc& rhs);
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
