// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::AnalyticData {

/*!
 * \brief Kelvin-Helmholtz instability initial data with a smooth shear layer,
 * following the configuration of \cite Beckwith2011iy (also used by Mattia &
 * Mignone, MNRAS 510, 481 (2022), their eqs. 74-76).
 *
 * Unlike grmhd::AnalyticData::KhInstability (which uses a sharp top-hat strip
 * and a uniform density), this class reproduces the relativistic KH benchmark
 * exactly: a central strip of half-width \f$h\f$ centered at \f$y=0\f$ with a
 * smooth (tanh) shear layer of thickness \f$a\f$, and a density that is tied to
 * the velocity (so the heavy/light fluids trace the roll-up). The two shear
 * interfaces sit at \f$y=\pm h\f$.
 *
 * The horizontal velocity is
 *
 * \f{align*}
 * v_x(x, y) = \mathrm{sign}(y)\, v_\mathrm{sh}\,
 *   \tanh\!\left(\frac{y - \mathrm{sign}(y)\, h}{a}\right),
 * \f}
 *
 * the density is tied to the velocity through
 *
 * \f{align*}
 * \rho(x, y) = \frac{1}{2}(\rho_h + \rho_l)
 *   + \frac{1}{2}(\rho_h - \rho_l)\, \frac{v_x}{v_\mathrm{sh}},
 * \f}
 *
 * so that \f$\rho = \rho_h\f$ where \f$v_x = +v_\mathrm{sh}\f$ and
 * \f$\rho = \rho_l\f$ where \f$v_x = -v_\mathrm{sh}\f$, and the instability is
 * seeded by a single-mode vertical velocity localized at the two interfaces
 *
 * \f{align*}
 * v_y(x, y) = \mathrm{sign}(y)\, A_0\, v_\mathrm{sh}\, \sin(2\pi x)\,
 *   \exp\!\left[-\left(\frac{y - \mathrm{sign}(y)\, h}{\sigma}\right)^2\right].
 * \f}
 *
 * The pressure is constant, the system is an ideal fluid of given adiabatic
 * index in Minkowski spacetime, and a uniform magnetic field can be added.
 */
class BeckwithStoneKhInstability
    : public evolution::initial_data::InitialData,
      public MarkAsAnalyticData,
      public AnalyticDataBase,
      public hydro::TemperatureInitialization<BeckwithStoneKhInstability> {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;

  /// The adiabatic index of the fluid.
  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {
        "The adiabatic index of the fluid."};
  };

  /// The asymptotic shear speed \f$v_\mathrm{sh}\f$ along x.
  struct ShearVelocity {
    using type = double;
    static constexpr Options::String help = {
        "The asymptotic shear speed v_sh along x."};
  };

  /// The half-width of the central strip; shear layers sit at \f$y=\pm h\f$.
  struct StripHalfWidth {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Half-width h of the central strip; shear layers are at y = +/- h."};
  };

  /// The thickness \f$a\f$ of the tanh shear layer.
  struct TransitionThickness {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Thickness a of the tanh shear layer."};
  };

  /// The density \f$\rho_h\f$ where \f$v_x = +v_\mathrm{sh}\f$.
  struct UpperDensity {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The density where v_x = +v_sh."};
  };

  /// The density \f$\rho_l\f$ where \f$v_x = -v_\mathrm{sh}\f$.
  struct LowerDensity {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The density where v_x = -v_sh."};
  };

  /// The initial (constant) pressure of the fluid.
  struct Pressure {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The initial (constant) pressure."};
  };

  /// The amplitude \f$A_0\f$ of the perturbation.
  struct PerturbAmplitude {
    using type = double;
    static constexpr Options::String help = {
        "The amplitude A_0 of the perturbation."};
  };

  /// The Gaussian width \f$\sigma\f$ of the perturbation envelope.
  struct PerturbWidth {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The Gaussian width sigma of the perturbation envelope."};
  };

  /// The uniform magnetic field.
  struct MagneticField {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {"The uniform magnetic field."};
  };

  using options =
      tmpl::list<AdiabaticIndex, ShearVelocity, StripHalfWidth,
                 TransitionThickness, UpperDensity, LowerDensity, Pressure,
                 PerturbAmplitude, PerturbWidth, MagneticField>;

  static constexpr Options::String help = {
      "Beckwith & Stone (2011) relativistic Kelvin-Helmholtz instability with a "
      "smooth tanh shear layer and density tied to the velocity."};

  BeckwithStoneKhInstability() = default;
  BeckwithStoneKhInstability(const BeckwithStoneKhInstability& /*rhs*/) =
      default;
  BeckwithStoneKhInstability& operator=(
      const BeckwithStoneKhInstability& /*rhs*/) = default;
  BeckwithStoneKhInstability(BeckwithStoneKhInstability&& /*rhs*/) = default;
  BeckwithStoneKhInstability& operator=(BeckwithStoneKhInstability&& /*rhs*/) =
      default;
  ~BeckwithStoneKhInstability() override = default;

  BeckwithStoneKhInstability(double adiabatic_index, double shear_velocity,
                             double strip_half_width, double transition_thickness,
                             double upper_density, double lower_density,
                             double pressure, double perturbation_amplitude,
                             double perturbation_width,
                             const std::array<double, 3>& magnetic_field);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit BeckwithStoneKhInstability(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BeckwithStoneKhInstability);
  /// \endcond

  /// @{
  /// Retrieve the GRMHD variables at a given position.
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Temperature<DataType>> {
    return TemperatureInitialization::variables(
        x, tmpl::list<hydro::Tags::Temperature<DataType>>{});
  }
  /// @}

  /// Retrieve a collection of hydrodynamic variables at position x
  template <typename DataType, typename Tag1, typename Tag2, typename... Tags>
  tuples::TaggedTuple<Tag1, Tag2, Tags...> variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<Tag1, Tag2, Tags...> /*meta*/) const {
    return {tuples::get<Tag1>(variables(x, tmpl::list<Tag1>{})),
            tuples::get<Tag2>(variables(x, tmpl::list<Tag2>{})),
            tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename DataType, typename Tag,
            Requires<tmpl::list_contains_v<
                gr::analytic_solution_tags<3, DataType>, Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const {
    constexpr double dummy_time = 0.0;
    return background_spacetime_.variables(x, dummy_time, tmpl::list<Tag>{});
  }

  const EquationsOfState::IdealFluid<true>& equation_of_state() const {
    return equation_of_state_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) override;

 private:
  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  double shear_velocity_ = std::numeric_limits<double>::signaling_NaN();
  double strip_half_width_ = std::numeric_limits<double>::signaling_NaN();
  double transition_thickness_ = std::numeric_limits<double>::signaling_NaN();
  double upper_density_ = std::numeric_limits<double>::signaling_NaN();
  double lower_density_ = std::numeric_limits<double>::signaling_NaN();
  double pressure_ = std::numeric_limits<double>::signaling_NaN();
  double perturbation_amplitude_ = std::numeric_limits<double>::signaling_NaN();
  double perturbation_width_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> magnetic_field_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  EquationsOfState::IdealFluid<true> equation_of_state_{};
  gr::Solutions::Minkowski<3> background_spacetime_{};

  friend bool operator==(const BeckwithStoneKhInstability& lhs,
                         const BeckwithStoneKhInstability& rhs);

  friend bool operator!=(const BeckwithStoneKhInstability& lhs,
                         const BeckwithStoneKhInstability& rhs);
};
}  // namespace grmhd::AnalyticData
