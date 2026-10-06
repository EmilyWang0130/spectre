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
 * \brief The relativistic magnetised Kelvin-Helmholtz problem of Mignone et
 * al. 2012 (ApJS 198, 7, Sec. 6.6), PLUTO's
 * `Test_Problems/RMHD/KH`.
 *
 * A single shear layer at \f$y = 0\f$ in a fluid of uniform density
 * \f$\rho_0\f$ and pressure \f$p_0\f$:
 *
 * \f{align*}
 * v_x(x, y) &= -v_0 \tanh(y / a), \\
 * v_y(x, y) &= A \sin(2\pi x / \lambda)\, \exp(-y^2 / \beta^2), \\
 * v_z &= 0, \qquad \vec B = \text{const},
 * \f}
 *
 * with \f$a\f$ the shear thickness, \f$A\f$ the perturbation amplitude,
 * \f$\lambda\f$ its wavelength and \f$\beta\f$ the width of its Gaussian
 * envelope (note \f$\beta^2\f$, not \f$2\beta^2\f$, in the exponent, as in
 * PLUTO's `init.c`). The electron fraction is a constant.
 *
 * PLUTO's configuration: \f$\rho_0 = 1\f$, \f$v_0 = 0.25\f$, \f$a = 0.01\f$,
 * \f$A = 0.01 v_0\f$, \f$\lambda = 1\f$, \f$\beta = 0.1\f$, \f$\Gamma = 4/3\f$,
 * the pressure set by the Mach number \f$M = v_0 / c_s = 0.435711\f$ through
 * \f$p_0 = (\Gamma - 1) / \{\Gamma [(\Gamma - 1) (M / v_0)^2 - 1]\}\f$, which
 * is \f$p_0 = 20\f$; and
 * \f$\vec B = (\sqrt{2 \sigma_\mathrm{pol} p_0}, 0, \sqrt{2 \sigma_\mathrm{tor}
 * p_0})\f$ with \f$\sigma_\mathrm{pol} = 0.01\f$, \f$\sigma_\mathrm{tor} =
 * 1\f$. The domain is \f$[0, 1] \times [-1, 1]\f$, periodic in \f$x\f$ and
 * outflow in \f$y\f$, evolved to \f$t = 5\f$ (configurations 1 and 3) or 10
 * (configuration 2).
 */
class MignoneKhInstability
    : public evolution::initial_data::InitialData,
      public MarkAsAnalyticData,
      public AnalyticDataBase,
      public hydro::TemperatureInitialization<MignoneKhInstability> {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;

  /// The adiabatic index of the fluid.
  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {
        "The adiabatic index of the fluid."};
  };
  struct Density {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {"The uniform rest-mass density."};
  };
  struct Pressure {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The uniform pressure (PLUTO sets it from the Mach number; 20 for "
        "its configuration)."};
  };
  struct ElectronFraction {
    using type = double;
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
    static constexpr Options::String help = {
        "The uniform electron fraction (passive here)."};
  };
  struct ShearVelocity {
    using type = double;
    static constexpr Options::String help = {
        "The asymptotic shear speed v_0: v_x = -v_0 tanh(y / a)."};
  };
  struct ShearThickness {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The thickness a of the tanh shear layer."};
  };
  struct PerturbAmplitude {
    using type = double;
    static constexpr Options::String help = {
        "The amplitude A of the v_y perturbation (PLUTO: 0.01 v_0)."};
  };
  struct PerturbWavelength {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The wavelength of the perturbation along x: v_y ~ sin(2 pi x / "
        "wavelength)."};
  };
  struct PerturbWidth {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The width beta of the Gaussian envelope exp(-y^2 / beta^2)."};
  };
  struct MagneticField {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "The uniform magnetic field (PLUTO: (sqrt(2 sigma_pol p), 0, "
        "sqrt(2 sigma_tor p)))."};
  };

  using options =
      tmpl::list<AdiabaticIndex, Density, Pressure, ElectronFraction,
                 ShearVelocity, ShearThickness, PerturbAmplitude,
                 PerturbWavelength, PerturbWidth, MagneticField>;

  static constexpr Options::String help = {
      "The relativistic magnetised Kelvin-Helmholtz problem of Mignone et al. "
      "2012 (PLUTO's RMHD/KH): a single tanh shear layer at y = 0 in a uniform "
      "fluid, a sinusoidal v_y perturbation with a Gaussian envelope, and a "
      "uniform magnetic field."};

  MignoneKhInstability() = default;
  MignoneKhInstability(const MignoneKhInstability& /*rhs*/) =
      default;
  MignoneKhInstability& operator=(
      const MignoneKhInstability& /*rhs*/) = default;
  MignoneKhInstability(MignoneKhInstability&& /*rhs*/) = default;
  MignoneKhInstability& operator=(MignoneKhInstability&& /*rhs*/) =
      default;
  ~MignoneKhInstability() override = default;

  MignoneKhInstability(double adiabatic_index, double density, double pressure,
                       double electron_fraction, double shear_velocity,
                       double shear_thickness, double perturbation_amplitude,
                       double perturbation_wavelength, double perturbation_width,
                       const std::array<double, 3>& magnetic_field);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit MignoneKhInstability(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(MignoneKhInstability);
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
  double density_ = std::numeric_limits<double>::signaling_NaN();
  double pressure_ = std::numeric_limits<double>::signaling_NaN();
  double electron_fraction_ = std::numeric_limits<double>::signaling_NaN();
  double shear_velocity_ = std::numeric_limits<double>::signaling_NaN();
  double shear_thickness_ = std::numeric_limits<double>::signaling_NaN();
  double perturbation_amplitude_ = std::numeric_limits<double>::signaling_NaN();
  double perturbation_wavelength_ =
      std::numeric_limits<double>::signaling_NaN();
  double perturbation_width_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> magnetic_field_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  EquationsOfState::IdealFluid<true> equation_of_state_{};
  gr::Solutions::Minkowski<3> background_spacetime_{};

  friend bool operator==(const MignoneKhInstability& lhs,
                         const MignoneKhInstability& rhs);

  friend bool operator!=(const MignoneKhInstability& lhs,
                         const MignoneKhInstability& rhs);
};
}  // namespace grmhd::AnalyticData
