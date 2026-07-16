// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Marquina.hpp"
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
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
/*!
 * \brief Marquina flux with the Complementary Projection Method (CPM)
 * replacing the degenerate-block characteristic decomposition.
 *
 * Behaves identically to `Marquina` on the two acoustic modes. For the
 * fourfold-degenerate \f$\lambda_0 = v_n\f$ eigenspace, the flux is assembled
 * via the spectral projector complement
 * \f$(\mathbf{I} - \mathbf{R}_+ \mathbf{L}_+^\top -
 * \mathbf{R}_- \mathbf{L}_-^\top)\mathbf{F}\f$ under a single-speed Marquina
 * upwind at \f$v_n\f$ (Fedkiw, Merriman & Osher 1997 §4.3, eqs 6 and 47).
 * The four non-unique degenerate eigenvectors and their \f$\zeta \to 0\f$
 * regularization are never constructed.
 *
 * Analytically identical to `Marquina` at every face point. Implementation
 * lives in `Marquina::dg_package_data_impl` and
 * `Marquina::dg_boundary_terms_impl`, gated by a `bool` first argument;
 * this class is a thin wrapper that always passes `true` and provides a
 * separate factory entry so YAML input files can select the CPM variant
 * explicitly. SpECTRE requires all createable `BoundaryCorrection`s to
 * be `final`, so `MarquinaCpm` cannot inherit from `Marquina`; instead it
 * mirrors the same public tag types and forwards to Marquina's static
 * `_impl` helpers.
 */
class MarquinaCpm final : public evolution::BoundaryCorrection {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Marquina flux with the Complementary Projection Method (CPM) "
      "degenerate-block assembly. Analytically identical to Marquina; "
      "avoids constructing the four non-unique degenerate eigenvectors."};

  MarquinaCpm() = default;
  MarquinaCpm(const MarquinaCpm&) = default;
  MarquinaCpm& operator=(const MarquinaCpm&) = default;
  MarquinaCpm(MarquinaCpm&&) = default;
  MarquinaCpm& operator=(MarquinaCpm&&) = default;
  ~MarquinaCpm() override = default;

  /// \cond
  explicit MarquinaCpm(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(MarquinaCpm);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  // Package-data / boundary-terms tags are identical to `Marquina` so
  // that the DG framework treats the two classes interchangeably. We
  // alias to `Marquina`'s public tag types instead of duplicating them.
  using dg_package_field_tags = Marquina::dg_package_field_tags;
  using dg_package_data_temporary_tags =
      Marquina::dg_package_data_temporary_tags;
  using dg_package_data_primitive_tags =
      Marquina::dg_package_data_primitive_tags;
  using dg_package_data_volume_tags = Marquina::dg_package_data_volume_tags;
  using dg_boundary_terms_volume_tags =
      Marquina::dg_boundary_terms_volume_tags;

  static double dg_package_data(
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
      gsl::not_null<tnsr::i<DataVector, 3, Frame::NoFrame>*>
          packaged_characteristic_speeds,
      gsl::not_null<tnsr::iJ<DataVector, 6, Frame::NoFrame>*>
          packaged_left_characteristic_fields,
      gsl::not_null<tnsr::ij<DataVector, 6, Frame::NoFrame>*>
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
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state);

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
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_s_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_b_ext,
      const Scalar<DataVector>& normal_dot_flux_tilde_phi_ext,
      const tnsr::i<DataVector, 3, Frame::NoFrame>& characteristic_speeds_ext,
      const tnsr::iJ<DataVector, 6, Frame::NoFrame>&
          left_characteristic_fields_ext,
      const tnsr::ij<DataVector, 6, Frame::NoFrame>&
          right_characteristic_fields_ext,
      dg::Formulation dg_formulation);
};

bool operator==(const MarquinaCpm& lhs, const MarquinaCpm& rhs);
bool operator!=(const MarquinaCpm& lhs, const MarquinaCpm& rhs);
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
