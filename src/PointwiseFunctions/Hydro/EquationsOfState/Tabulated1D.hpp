// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/control/expr_iif.hpp>
#include <boost/preprocessor/list/adt.hpp>
#include <boost/preprocessor/repetition/for.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <limits>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "IO/H5/EosTable.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace EquationsOfState {
/*!
 * \ingroup EquationsOfStateGroup
 * \brief Tabulated 1D barotropic equation of state loaded from an h5 file.
 *
 * Reads a 1D barotropic slice stored in an `h5::EosTable` subfile — with
 * `"number density"` as the single independent variable (in
 * fm\f$^{-3}\f$) and the standard CompOSE-natural datasets
 * `"pressure"` (MeV/fm\f$^{3}\f$), `"specific internal energy"`
 * (dimensionless), and `"chi slope"` (\f$d\ln p / d\ln n_b\f$,
 * dimensionless). Applies the runtime nuclear-to-geometric unit
 * conversion (same `nb_fm3_to_geom` and `press_MeV_to_geom` constants
 * used by `Tabulated3D`), computes the specific enthalpy
 * \f$h = 1 + \epsilon + p/\rho\f$ point-wise, and stores the resulting
 * five 1D arrays.
 *
 * The intended workflow is symmetric with the 3D pipeline: an offline
 * Python converter (e.g. `ConvertComposeBetaTo1D.py`) reads a CompOSE
 * `.beta` ASCII slice and emits the 1D h5 file. Runtime just consumes
 * it, so the h5 becomes a first-class, inspectable artifact rather than
 * a hidden per-startup projection.
 *
 * \note This is the Stage 2 class: the h5 read + array storage + PUP is
 * in place, but the pointwise field queries (`pressure_from_density`,
 * `chi_from_density`, `rest_mass_density_from_enthalpy`, ...) still
 * throw. Stages 3-5 fill those in with
 * `intrp::UniformMultiLinearSpanInterpolation<1, N>` on the stored
 * arrays and a `std::lower_bound` for the h-inversion.
 */
template <bool IsRelativistic>
class Tabulated1D : public EquationOfState<IsRelativistic, 1> {
 public:
  static constexpr size_t thermodynamic_dim = 1;
  static constexpr bool is_relativistic = IsRelativistic;

  struct TableFilename {
    using type = std::string;
    static constexpr Options::String help = {
        "Path to the h5 file containing the tabulated 1D EOS."};
  };

  struct TableSubFilename {
    using type = std::string;
    static constexpr Options::String help = {
        "Name of the EosTable subfile inside the h5 file (e.g. "
        "\"dd2.eos_beta\")."};
  };

  static constexpr Options::String help = {
      "A tabulated 1D barotropic equation of state, loaded from an h5 file "
      "produced by an offline converter (e.g. ConvertComposeBetaTo1D.py for "
      "cold beta-equilibrium slices from CompOSE .beta ASCII tables). "
      "Structurally analogous to Tabulated3D."};

  using options = tmpl::list<TableFilename, TableSubFilename>;

  Tabulated1D() = default;
  Tabulated1D(const Tabulated1D&) = default;
  Tabulated1D& operator=(const Tabulated1D&) = default;
  Tabulated1D(Tabulated1D&&) = default;
  Tabulated1D& operator=(Tabulated1D&&) = default;
  ~Tabulated1D() override = default;

  /// Construct by reading the named subfile from the h5 file at `filename`.
  Tabulated1D(const std::string& filename, const std::string& subfilename);

  /// Construct directly from an already-opened `h5::EosTable`.
  /// Useful for tests that build the table in-memory.
  explicit Tabulated1D(const h5::EosTable& spectre_eos);

  std::unique_ptr<EquationOfState<IsRelativistic, 1>> get_clone()
      const override;

  std::unique_ptr<EquationOfState<IsRelativistic, 3>> promote_to_3d_eos()
      const override;

  std::unique_ptr<EquationOfState<IsRelativistic, 2>> promote_to_2d_eos()
      const override;

  bool is_equal(const EquationOfState<IsRelativistic, 1>& rhs) const override;

  bool operator==(const Tabulated1D<IsRelativistic>& rhs) const;

  bool operator!=(const Tabulated1D<IsRelativistic>& rhs) const;

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(Tabulated1D, 1)

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<IsRelativistic, 1>), Tabulated1D);

  /// The lower bound of the rest mass density that is valid for this EOS
  double rest_mass_density_lower_bound() const override;

  /// The upper bound of the rest mass density that is valid for this EOS
  double rest_mass_density_upper_bound() const override;

  /// The lower bound of the specific enthalpy that is valid for this EOS
  double specific_enthalpy_lower_bound() const override;

  /// The lower bound of the specific internal energy that is valid for this EOS
  double specific_internal_energy_lower_bound() const override;

  /// The upper bound of the specific internal energy that is valid for this EOS
  double specific_internal_energy_upper_bound() const override;

  /// The vacuum baryon mass for this EoS
  double baryon_mass() const override {
    return hydro::units::geometric::default_baryon_mass;
  }

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(1)

  void initialize(const h5::EosTable& spectre_eos);

  // Uniformly log-spaced grid in log(rho_geom). Uniformity is required
  // by intrp::UniformMultiLinearSpanInterpolation at Stage 3.
  std::vector<double> log_rho_grid_;
  // Pressure in geometric units (linear).
  std::vector<double> pressure_;
  // Specific internal energy (dimensionless, linear).
  std::vector<double> specific_internal_energy_;
  // Specific enthalpy h = 1 + eps + p/rho (dimensionless, linear).
  std::vector<double> specific_enthalpy_;
  // d(ln p)/d(ln rho), dimensionless. Chi_geom = (p_geom/rho_geom) * chi_slope.
  std::vector<double> chi_slope_;
};

/// \cond
template <bool IsRelativistic>
PUP::able::PUP_ID EquationsOfState::Tabulated1D<IsRelativistic>::my_PUP_ID = 0;
/// \endcond
}  // namespace EquationsOfState
