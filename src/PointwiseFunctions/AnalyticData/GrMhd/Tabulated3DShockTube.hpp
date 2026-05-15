// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated3d.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::AnalyticData {

/*!
 * \brief Piecewise-constant shock-tube initial data using a tabulated 3D EOS.
 *
 * The fluid variables are constant on either side of a discontinuity at
 * `DiscontinuityPosition` in the x-direction, with Minkowski spacetime
 * background. Thermodynamic quantities are computed consistently from the
 * tabulated EOS using the user-specified density, temperature, and electron
 * fraction on each side.
 */
class Tabulated3DShockTube : public evolution::initial_data::InitialData,
                             public MarkAsAnalyticData,
                             public AnalyticDataBase {
 public:
  using equation_of_state_type = EquationsOfState::Tabulated3D<true>;

  struct TableFilename {
    using type = std::string;
    static constexpr Options::String help{"File name of the EOS table"};
  };

  struct TableSubFilename {
    using type = std::string;
    static constexpr Options::String help{
        "Subfile name of the EOS table, e.g. 'dd2'."};
  };

  struct LeftRestMassDensity {
    using type = double;
    static std::string name() { return "LeftDensity"; }
    static constexpr Options::String help = {
        "Fluid rest-mass density in the left half-domain"};
    static type lower_bound() { return 0.0; }
  };

  struct RightRestMassDensity {
    using type = double;
    static std::string name() { return "RightDensity"; }
    static constexpr Options::String help = {
        "Fluid rest-mass density in the right half-domain"};
    static type lower_bound() { return 0.0; }
  };

  struct LeftTemperature {
    using type = double;
    static constexpr Options::String help = {
        "Fluid temperature in the left half-domain"};
    static type lower_bound() { return 0.0; }
  };

  struct RightTemperature {
    using type = double;
    static constexpr Options::String help = {
        "Fluid temperature in the right half-domain"};
    static type lower_bound() { return 0.0; }
  };

  struct LeftElectronFraction {
    using type = double;
    static std::string name() { return "LeftElectronFraction"; }
    static constexpr Options::String help = {
        "Fluid electron fraction in the left half-domain"};
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };

  struct RightElectronFraction {
    using type = double;
    static std::string name() { return "RightElectronFraction"; }
    static constexpr Options::String help = {
        "Fluid electron fraction in the right half-domain"};
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };

  struct LeftSpatialVelocity {
    using type = std::array<double, 3>;
    static std::string name() { return "LeftVelocity"; }
    static constexpr Options::String help = {
        "Fluid spatial velocity in the left half-domain"};
  };

  struct RightSpatialVelocity {
    using type = std::array<double, 3>;
    static std::string name() { return "RightVelocity"; }
    static constexpr Options::String help = {
        "Fluid spatial velocity in the right half-domain"};
  };

  struct LeftMagneticField {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Magnetic field in the left half-domain"};
  };

  struct RightMagneticField {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Magnetic field in the right half-domain"};
  };

  struct DiscontinuityPosition {
    using type = double;
    static constexpr Options::String help = {
        "Position of the x-direction discontinuity"};
  };

  using options =
      tmpl::list<TableFilename, TableSubFilename, LeftRestMassDensity,
                 RightRestMassDensity, LeftTemperature, RightTemperature,
                 LeftElectronFraction, RightElectronFraction,
                 LeftSpatialVelocity, RightSpatialVelocity, LeftMagneticField,
                 RightMagneticField, DiscontinuityPosition>;

  static constexpr Options::String help = {
      "Piecewise-constant shock-tube initial data using a tabulated 3D EOS."};

  Tabulated3DShockTube() = default;
  Tabulated3DShockTube(const Tabulated3DShockTube& /*rhs*/) = default;
  Tabulated3DShockTube& operator=(const Tabulated3DShockTube& /*rhs*/) =
      default;
  Tabulated3DShockTube(Tabulated3DShockTube&& /*rhs*/) = default;
  Tabulated3DShockTube& operator=(Tabulated3DShockTube&& /*rhs*/) = default;
  ~Tabulated3DShockTube() override = default;

  Tabulated3DShockTube(
      const std::string& table_filename, const std::string& table_subfilename,
      double left_rest_mass_density, double right_rest_mass_density,
      double left_temperature, double right_temperature,
      double left_electron_fraction, double right_electron_fraction,
      const std::array<double, 3>& left_spatial_velocity,
      const std::array<double, 3>& right_spatial_velocity,
      const std::array<double, 3>& left_magnetic_field,
      const std::array<double, 3>& right_magnetic_field,
      double discontinuity_position);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit Tabulated3DShockTube(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Tabulated3DShockTube);
  /// \endcond

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Temperature<DataType>>;

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

  template <typename DataType, typename Tag1, typename Tag2, typename... Tags>
  tuples::TaggedTuple<Tag1, Tag2, Tags...> variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<Tag1, Tag2, Tags...> /*meta*/) const {
    return {tuples::get<Tag1>(variables(x, tmpl::list<Tag1>{})),
            tuples::get<Tag2>(variables(x, tmpl::list<Tag2>{})),
            tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  template <typename DataType, typename Tag,
            Requires<tmpl::list_contains_v<
                gr::analytic_solution_tags<3, DataType>, Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const {
    constexpr double dummy_time = 0.0;
    return background_spacetime_.variables(x, dummy_time, tmpl::list<Tag>{});
  }

  const equation_of_state_type& equation_of_state() const {
    return equation_of_state_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  equation_of_state_type equation_of_state_{};
  gr::Solutions::Minkowski<3> background_spacetime_{};

  double left_rest_mass_density_ = std::numeric_limits<double>::signaling_NaN();
  double right_rest_mass_density_ =
      std::numeric_limits<double>::signaling_NaN();
  double left_temperature_ = std::numeric_limits<double>::signaling_NaN();
  double right_temperature_ = std::numeric_limits<double>::signaling_NaN();
  double left_electron_fraction_ = std::numeric_limits<double>::signaling_NaN();
  double right_electron_fraction_ =
      std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> left_spatial_velocity_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  std::array<double, 3> right_spatial_velocity_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  std::array<double, 3> left_magnetic_field_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  std::array<double, 3> right_magnetic_field_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  double discontinuity_position_ = std::numeric_limits<double>::signaling_NaN();

  friend bool operator==(const Tabulated3DShockTube& lhs,
                         const Tabulated3DShockTube& rhs);
  friend bool operator!=(const Tabulated3DShockTube& lhs,
                         const Tabulated3DShockTube& rhs);
};

}  // namespace grmhd::AnalyticData
