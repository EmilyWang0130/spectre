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
#include "NumericalAlgorithms/Interpolation/MultiLinearSpanInterpolation.hpp"
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
 * (dimensionless), and `"adiabatic index"` (\f$\Gamma_{\rm eff} =
 * d\ln p / d\ln n_b\f$, dimensionless). Applies the runtime
 * nuclear-to-geometric unit conversion (same `nb_fm3_to_geom` and
 * `press_MeV_to_geom` constants used by `Tabulated3D`), computes the specific
 * enthalpy \f$h = 1 + \epsilon + p/\rho\f$ point-wise, and stores the resulting
 * arrays.
 *
 * Interpolation at query time uses
 * `intrp::UniformMultiLinearSpanInterpolation<1, NumberOfVars>` in
 * `log(\rho_{\rm geom})`, matching the convention of `Tabulated3D`.
 * Log-space storage of pressure and (shifted) specific internal energy
 * keeps the interpolation accurate across the many decades DD2 and
 * similar tables span.
 *
 * \note This is the Stage 3 class: `pressure_from_density` and
 * `specific_internal_energy_from_density` are implemented via the
 * interpolator. `chi_from_density` (Stage 4) and
 * `rest_mass_density_from_enthalpy` (Stage 5) still throw.
 */
template <bool IsRelativistic>
class Tabulated1D : public EquationOfState<IsRelativistic, 1> {
 public:
  static constexpr size_t thermodynamic_dim = 1;
  static constexpr bool is_relativistic = IsRelativistic;

  /// Index of each field inside `table_data_`
  /// (log_pressure, log(eps - energy_shift), adiabatic_index). Kept as an
  /// enum so the compile-time index passed to
  /// `interpolator_.template interpolate<...>` is self-documenting.
  enum InterpolationField : size_t {
    LogPressure = 0,
    LogShiftedEpsilon = 1,
    AdiabaticIndex = 2,
    NumberOfVars = 3
  };

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
      "produced by an offline converter (e.g. ConvertComposeBetaTo1D for "
      "cold beta-equilibrium slices from CompOSE .beta ASCII tables). "
      "Structurally analogous to Tabulated3D."};

  using options = tmpl::list<TableFilename, TableSubFilename>;

  Tabulated1D() = default;
  Tabulated1D(const Tabulated1D& rhs);
  Tabulated1D& operator=(const Tabulated1D& rhs);
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

  /// Rebuild the non-PUP'd interpolator from the stored grid + table data.
  /// Called at the end of every path that mutates `log_rho_grid_` or
  /// `table_data_`: `initialize`, PUP unpack, copy/assign.
  void initialize_interpolator();

  /// Uniformly log-spaced grid in log(rho_geom). Uniformity is required
  /// by intrp::UniformMultiLinearSpanInterpolation.
  std::vector<double> log_rho_grid_;
  /// Packed table [log(p_geom), log(eps - energy_shift), adiabatic_index]
  /// per grid point, variable index inner-most (matches SpECTRE's flat
  /// C-order convention). `adiabatic_index = d(ln p)/d(ln rho)` is
  /// dimensionless; the query multiplies by `p_geom/rho_geom` to get
  /// the dimensional `chi = dp/drho`.
  std::vector<double> table_data_;
  /// h = 1 + eps + p/rho (linear). Kept separate — Stage 5's h -> rho
  /// inversion uses std::lower_bound on this array directly.
  std::vector<double> specific_enthalpy_;
  /// Additive shift on epsilon so that (eps - energy_shift) is strictly
  /// positive and can be stored in log-space. Same convention Tabulated3D
  /// uses. Zero when eps_min >= 0.
  double energy_shift_ = 0.0;

  /// Non-PUP'd; rebuilt from log_rho_grid_ / table_data_ after any state
  /// change. Docs on the interpolator class note it is intentionally not
  /// PUPable.
  intrp::UniformMultiLinearSpanInterpolation<1, NumberOfVars> interpolator_{};
};

/// \cond
template <bool IsRelativistic>
PUP::able::PUP_ID EquationsOfState::Tabulated1D<IsRelativistic>::my_PUP_ID = 0;
/// \endcond
}  // namespace EquationsOfState
