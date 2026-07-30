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

#include "DataStructures/Tensor/TypeAliases.hpp"
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
 * \brief Tabulated 1D barotropic equation of state.
 *
 * Stores a projected 1D barotropic slice \f$p(\rho)\f$, \f$\epsilon(\rho)\f$
 * of an underlying 3D equation of state on a uniform grid in either
 * \f$\rho\f$ or \f$\log\rho\f$, and answers all 1D EOS API calls by
 * interpolation on the stored arrays. Intended for use with `TovStar` to
 * drive the TOV ODE integration from a tabulated 3D EOS (e.g. `Tabulated3D`
 * loaded from a CompOSE table) without a parametric fit.
 *
 * \note This is the Stage 1 skeleton: the class is factory-registered and
 * Charm-serializable, but all pointwise queries currently throw. Stage 2
 * adds the projection loop and array storage; Stages 3-5 add
 * interpolation, finite-difference derivatives, and the h-inversion.
 */
template <bool IsRelativistic>
class Tabulated1D : public EquationOfState<IsRelativistic, 1> {
 public:
  static constexpr size_t thermodynamic_dim = 1;
  static constexpr bool is_relativistic = IsRelativistic;

  struct NumberOfGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Number of grid points in the projected 1D table."};
    static size_t lower_bound() { return 32; }
  };

  struct LogSpacing {
    using type = bool;
    static constexpr Options::String help = {
        "If true, the rest mass density grid is log-spaced; else linear."};
  };

  static constexpr Options::String help = {
      "A tabulated 1D barotropic equation of state, obtained by projecting "
      "an underlying 3D equation of state onto a 1D curve (currently only "
      "cold beta-equilibrium is supported). Stage 1 skeleton: constructor "
      "stores grid parameters but does not yet build the projection."};

  using options = tmpl::list<NumberOfGridPoints, LogSpacing>;

  Tabulated1D() = default;
  Tabulated1D(const Tabulated1D&) = default;
  Tabulated1D& operator=(const Tabulated1D&) = default;
  Tabulated1D(Tabulated1D&&) = default;
  Tabulated1D& operator=(Tabulated1D&&) = default;
  ~Tabulated1D() override = default;

  Tabulated1D(size_t number_of_grid_points, bool log_spacing);

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
  double rest_mass_density_lower_bound() const override { return 0.0; }

  /// The upper bound of the rest mass density that is valid for this EOS
  double rest_mass_density_upper_bound() const override {
    return std::numeric_limits<double>::max();
  }

  /// The lower bound of the specific enthalpy that is valid for this EOS
  double specific_enthalpy_lower_bound() const override {
    return IsRelativistic ? 1.0 : 0.0;
  }

  /// The lower bound of the specific internal energy that is valid for this EOS
  double specific_internal_energy_lower_bound() const override { return 0.0; }

  /// The upper bound of the specific internal energy that is valid for this EOS
  double specific_internal_energy_upper_bound() const override {
    return std::numeric_limits<double>::max();
  }

  /// The vacuum baryon mass for this EoS
  double baryon_mass() const override {
    return hydro::units::geometric::default_baryon_mass;
  }

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(1)

  size_t number_of_grid_points_ = 0;
  bool log_spacing_ = true;
};

/// \cond
template <bool IsRelativistic>
PUP::able::PUP_ID EquationsOfState::Tabulated1D<IsRelativistic>::my_PUP_ID = 0;
/// \endcond
}  // namespace EquationsOfState
