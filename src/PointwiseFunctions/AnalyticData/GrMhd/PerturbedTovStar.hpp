// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpline.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::AnalyticData {

/*!
 * \brief The radial parts of a nonradial stellar-mode displacement,
 * \f$\xi_r(r)\f$ and \f$\xi_\perp(r)\f$, tabulated on the coordinate radius.
 *
 * The file format is: any number of comment lines starting with `#`, then one
 * line `l omega n`, then `n` rows `r xi_r xi_perp` with `r` strictly increasing
 * from 0. Everything is in geometric units (\f$G=c=M_\odot=1\f$). Outside the
 * tabulated range both displacements are zero.
 *
 * For \f$m=0\f$ the full displacement is (Reisenegger & Goldreich 1992,
 * eq. 40)
 * \f{equation}
 *   \xi^i = \xi_r(r)\,P_l(\mu)\,\hat r^i
 *         + \xi_\perp(r)\,\frac{dP_l}{d\theta}\,\hat\theta^i,\qquad
 *   \mu = \hat r\cdot\hat n,
 * \f}
 * with \f$\hat n\f$ the polar axis. Only \f$l=2\f$ is implemented.
 */
class ModeProfile {
 public:
  ModeProfile() = default;
  explicit ModeProfile(const std::string& filename);
  ModeProfile(size_t l, double omega, std::vector<double> radius,
              std::vector<double> xi_r, std::vector<double> xi_perp);

  size_t l() const { return l_; }
  double omega() const { return omega_; }
  double outer_radius() const { return xi_r_interpolant_.x_values().back(); }
  double xi_r(double r) const;
  double xi_perp(double r) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  friend bool operator==(const ModeProfile& lhs, const ModeProfile& rhs);
  void initialize(std::vector<double> radius, std::vector<double> xi_r,
                  std::vector<double> xi_perp);

  size_t l_{2};
  double omega_{std::numeric_limits<double>::signaling_NaN()};
  intrp::CubicSpline xi_r_interpolant_{};
  intrp::CubicSpline xi_perp_interpolant_{};
};

bool operator!=(const ModeProfile& lhs, const ModeProfile& rhs);

namespace perturbed_tov_detail {

using StarRegion = RelativisticEuler::Solutions::tov_detail::StarRegion;

template <typename DataType, StarRegion Region>
struct PerturbedTovVariables
    : RelativisticEuler::Solutions::tov_detail::TovVariables<DataType, Region> {
  static constexpr size_t Dim = 3;
  using Base =
      RelativisticEuler::Solutions::tov_detail::TovVariables<DataType, Region>;
  using Cache = typename Base::Cache;
  using Base::operator();
  using Base::coords;
  using Base::eos;
  using Base::radial_solution;
  using Base::radius;
  using Base::yeq_eos;

  const ModeProfile& mode;
  double displacement_amplitude;
  double velocity_amplitude;
  const std::array<double, 3>& polar_axis;

  PerturbedTovVariables(
      const tnsr::I<DataType, 3>& local_x, const DataType& local_radius,
      const RelativisticEuler::Solutions::TovSolution& local_radial_solution,
      const EquationsOfState::EquationOfState<true, 1>& local_eos,
      const EquationsOfState::EquationOfState<true, 3>* const local_yeq_eos,
      const ModeProfile& local_mode, const double local_displacement_amplitude,
      const double local_velocity_amplitude,
      const std::array<double, 3>& local_polar_axis)
      : Base(local_x, local_radius, local_radial_solution, local_eos,
             local_yeq_eos),
        mode(local_mode),
        displacement_amplitude(local_displacement_amplitude),
        velocity_amplitude(local_velocity_amplitude),
        polar_axis(local_polar_axis) {}

  // These three hide the base-class operators of the same signature rather
  // than override them: `TovVariablesCache::get_var` dispatches on the static
  // type of the computer it is handed, and `TovVariables` never requests these
  // tags from itself, so no virtual call is ever needed to reach them.
  void operator()(gsl::not_null<Scalar<DataType>*> electron_fraction,
                  gsl::not_null<Cache*> cache,
                  hydro::Tags::ElectronFraction<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> spatial_velocity,
                  gsl::not_null<Cache*> cache,
                  hydro::Tags::SpatialVelocity<DataType, 3> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> lorentz_factor,
                  gsl::not_null<Cache*> cache,
                  hydro::Tags::LorentzFactor<DataType> /*meta*/) const;

 private:
  /// The unscaled displacement \f$\xi^i\f$ and its radial part
  /// \f$\xi_r P_l(\mu)\f$ at every point.
  void displacement(gsl::not_null<tnsr::I<DataType, 3>*> xi,
                    gsl::not_null<Scalar<DataType>*> xi_radial) const;
};

}  // namespace perturbed_tov_detail

/*!
 * \brief A TOV star with a tabulated nonradial-mode displacement imposed on
 * its composition and/or velocity.
 *
 * The background is exactly `RelativisticEuler::Solutions::TovStar`,
 * including its optional `Yeq` companion EOS for the beta-equilibrium
 * electron fraction. On top of it, a displacement eigenfunction read from
 * `ModeProfileFile` (see `ModeProfile`) enters in two independent ways:
 *
 * - **Composition displacement** (`DisplacementAmplitude` \f$A_\xi\f$). With
 *   frozen composition the fluid element now at \f$r\f$ came from
 *   \f$r - A_\xi\,\xi_r P_l(\mu)\f$, so
 *   \f{equation}
 *     Y_e(x) = Y_{e,\beta}\big(\rho_0(r - A_\xi\,\xi_r(r) P_l(\mu))\big).
 *   \f}
 *   The background is evaluated at the displaced radius through the same
 *   enthalpy lookup the base class uses, so no derivative of \f$Y_e\f$ is ever
 *   taken and the result is exact for a purely radial Lagrangian displacement.
 *   Points whose source would lie outside the star keep the background value.
 *   Density, pressure and internal energy are *not* perturbed: this is a
 *   composition kick at rest, not a full eigenmode snapshot.
 *
 * - **Velocity kick** (`VelocityAmplitude` \f$A_v\f$):
 *   \f$v^i = A_v\,\omega\,\xi^i\f$, with the Lorentz factor recomputed from
 *   the TOV spatial metric. Density, pressure and composition stay on the
 *   background.
 *
 * A single-mode phase uses one of the two, not both: the displacement is the
 * \f$\cos\omega t\f$ quadrature of a standing mode and the velocity the
 * \f$\sin\omega t\f$ one.
 *
 * The polar axis of the \f$m=0\f$ pattern is `PolarAxis`; for a
 * `CartoonCylinder` domain, whose symmetry axis is \f$y\f$, use `[0, 1, 0]`.
 *
 * The profile's outer radius must agree with the star's `outer_radius()` in
 * the chosen `Coordinates` to within 2%; an areal-radius profile handed to an
 * isotropic-coordinate star is off by ~10% and is refused.
 *
 * \warning The eigenfunction supplied is only as consistent with this
 * background as whoever produced it made it. The pilot g-mode profiles were
 * shot from the *Newtonian* RG92 equations on a *TOV* background.
 */
class PerturbedTovStar : public virtual evolution::initial_data::InitialData,
                         public MarkAsAnalyticData,
                         private RelativisticEuler::Solutions::TovStar {
 private:
  using tov_star = RelativisticEuler::Solutions::TovStar;

 public:
  struct ModeProfileFile {
    using type = std::string;
    static constexpr Options::String help = {
        "Text file with the mode displacement: comment lines starting with "
        "'#', then 'l omega n', then n rows 'r xi_r xi_perp'. r is the radius "
        "in the coordinates selected by 'Coordinates'; all in geometric "
        "units."};
  };

  struct DisplacementAmplitude {
    using type = double;
    static constexpr Options::String help = {
        "Multiplier on the tabulated displacement in the composition "
        "perturbation Y_e(r) = Y_e,beta(rho_0(r - xi_r P_l)). 1 uses the "
        "file as written, 0 disables it."};
    static double suggested_value() { return 1.0; }
  };

  struct VelocityAmplitude {
    using type = double;
    static constexpr Options::String help = {
        "Multiplier A in the velocity kick v^i = A omega xi^i. 0 disables it. "
        "For a clean single-mode phase set either this or "
        "DisplacementAmplitude, not both."};
    static double suggested_value() { return 0.0; }
  };

  struct PolarAxis {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Direction of the mode's polar axis; cos(theta) = x.n / r. Need not "
        "be normalised. For a CartoonCylinder domain use [0, 1, 0]."};
  };

  using options =
      tmpl::push_back<tov_star::options, ModeProfileFile, DisplacementAmplitude,
                      VelocityAmplitude, PolarAxis>;

  static constexpr Options::String help = {
      "A TOV star with a tabulated nonradial-mode displacement imposed on its "
      "electron fraction (composition displacement) and/or its velocity "
      "(velocity kick)."};

  static constexpr size_t volume_dim = 3_st;

  template <typename DataType>
  using tags = typename tov_star::template tags<DataType>;

  PerturbedTovStar();
  PerturbedTovStar(const PerturbedTovStar& rhs);
  PerturbedTovStar& operator=(const PerturbedTovStar& rhs);
  PerturbedTovStar(PerturbedTovStar&& /*rhs*/);
  PerturbedTovStar& operator=(PerturbedTovStar&& /*rhs*/);
  ~PerturbedTovStar() override;

  PerturbedTovStar(
      double central_rest_mass_density,
      std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>
          equation_of_state,
      RelativisticEuler::Solutions::TovCoordinates coordinate_system,
      std::optional<std::unique_ptr<EquationsOfState::EquationOfState<true, 3>>>
          yeq_eos,
      const std::string& mode_profile_file, double displacement_amplitude,
      double velocity_amplitude, const std::array<double, 3>& polar_axis);

  /// Same as above but with the profile already in memory (for tests).
  PerturbedTovStar(
      double central_rest_mass_density,
      std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>
          equation_of_state,
      RelativisticEuler::Solutions::TovCoordinates coordinate_system,
      std::optional<std::unique_ptr<EquationsOfState::EquationOfState<true, 3>>>
          yeq_eos,
      ModeProfile mode, double displacement_amplitude,
      double velocity_amplitude, const std::array<double, 3>& polar_axis);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit PerturbedTovStar(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(PerturbedTovStar);
  /// \endcond

  using tov_star::equation_of_state;
  using tov_star::equation_of_state_type;
  using tov_star::radial_solution;
  using tov_star::yeq_eos;

  const ModeProfile& mode() const { return mode_; }
  double displacement_amplitude() const { return displacement_amplitude_; }
  double velocity_amplitude() const { return velocity_amplitude_; }
  const std::array<double, 3>& polar_axis() const { return polar_axis_; }

  /// Retrieve a collection of variables at `(x)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const {
    return variables_impl<perturbed_tov_detail::PerturbedTovVariables>(
        x, tmpl::list<Tags...>{}, mode_, displacement_amplitude_,
        velocity_amplitude_, polar_axis_);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(const PerturbedTovStar& lhs,
                         const PerturbedTovStar& rhs);
  /// Normalise the axis and check the profile against the star.
  void validate();

  ModeProfile mode_{};
  std::string mode_profile_file_{};
  double displacement_amplitude_{std::numeric_limits<double>::signaling_NaN()};
  double velocity_amplitude_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> polar_axis_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
};

bool operator!=(const PerturbedTovStar& lhs, const PerturbedTovStar& rhs);

}  // namespace grmhd::AnalyticData
