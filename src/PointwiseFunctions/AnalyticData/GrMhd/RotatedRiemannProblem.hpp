// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <string>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
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
 * \brief A relativistic MHD Riemann problem whose interface is rotated in the
 * x-y plane: the two-dimensional version of RiemannProblem.
 *
 * The interface is the line \f$s = 0\f$ with
 * \f$s = x\cos\theta + y\sin\theta\f$, where \f$\theta\f$ is `InterfaceAngle`
 * (radians); points with \f$s \le 0\f$ take the left state. The left and right
 * velocities and magnetic fields are given in the frame of the interface,
 * \f$(n, t, z)\f$ with the normal \f$n = (\cos\theta, \sin\theta, 0)\f$ and the
 * in-plane tangent \f$t = (-\sin\theta, \cos\theta, 0)\f$, i.e. the grid-frame
 * vector is \f$v = v_n n + v_t t + v_z \hat z\f$ (the same for \f$B\f$). With
 * `InterfaceAngle: 0` this is exactly RiemannProblem. The normal component of
 * the magnetic field must be the same on both sides for the field to be
 * divergence-free; the staircase that a Cartesian grid makes of the oblique
 * interface leaves a divergence at the interface cells which the divergence
 * cleaning removes. The lapse and the x-shift are as in RiemannProblem (the
 * shift is along the grid x axis, not rotated).
 *
 * This is PLUTO's `Test_Problems/RMHD/SodRot` configuration, written for the
 * HLLx campaign (2026-10-06) to test the resolution of slow shocks across an
 * oblique grid: with \f$\tan\theta = 1/2\f$ the discontinuity runs along the
 * diagonals of 2 : 1 blocks of cells.
 */
class RotatedRiemannProblem
    : public evolution::initial_data::InitialData,
      public AnalyticDataBase,
      public hydro::TemperatureInitialization<RotatedRiemannProblem>,
      public MarkAsAnalyticData {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;

  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {
        "The adiabatic index of the ideal fluid"};
    static type lower_bound() { return 1.0; }
  };
  struct LeftRestMassDensity {
    using type = double;
    static std::string name() { return "LeftDensity"; };
    static constexpr Options::String help = {
        "Fluid rest mass density on the left side (x cos(angle) + y "
        "sin(angle) <= 0)"};
    static type lower_bound() { return 0.0; }
  };
  struct RightRestMassDensity {
    using type = double;
    static std::string name() { return "RightDensity"; };
    static constexpr Options::String help = {
        "Fluid rest mass density on the right side"};
    static type lower_bound() { return 0.0; }
  };
  struct LeftPressure {
    using type = double;
    static constexpr Options::String help = {
        "Fluid pressure on the left side"};
    static type lower_bound() { return 0.0; }
  };
  struct RightPressure {
    using type = double;
    static constexpr Options::String help = {
        "Fluid pressure on the right side"};
    static type lower_bound() { return 0.0; }
  };
  struct LeftSpatialVelocity {
    using type = std::array<double, 3>;
    static std::string name() { return "LeftVelocity"; };
    static constexpr Options::String help = {
        "Fluid spatial velocity on the left side, as (normal, tangent, z) "
        "components in the frame of the interface"};
  };
  struct RightSpatialVelocity {
    using type = std::array<double, 3>;
    static std::string name() { return "RightVelocity"; };
    static constexpr Options::String help = {
        "Fluid spatial velocity on the right side, as (normal, tangent, z) "
        "components in the frame of the interface"};
  };
  struct LeftMagneticField {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Magnetic field on the left side, as (normal, tangent, z) components "
        "in the frame of the interface"};
  };
  struct RightMagneticField {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Magnetic field on the right side, as (normal, tangent, z) components "
        "in the frame of the interface; the normal component should equal the "
        "left one"};
  };
  struct Lapse {
    using type = double;
    static constexpr Options::String help = {
        "The value of the lapse. Standard is 1."};
    static type lower_bound() { return 0.0; }
  };
  struct ShiftX {
    using type = double;
    static constexpr Options::String help = {
        "The value of the x-component of the shift, beta^x. Standard is 0."};
  };
  struct InterfaceAngle {
    using type = double;
    static constexpr Options::String help = {
        "Angle (radians) between the interface normal and the x axis, in the "
        "x-y plane. 0 reproduces RiemannProblem."};
  };

  using options =
      tmpl::list<AdiabaticIndex, LeftRestMassDensity, RightRestMassDensity,
                 LeftPressure, RightPressure, LeftSpatialVelocity,
                 RightSpatialVelocity, LeftMagneticField, RightMagneticField,
                 Lapse, ShiftX, InterfaceAngle>;

  static constexpr Options::String help = {
      "Analytic initial data for a GRMHD Riemann problem whose interface is "
      "rotated by InterfaceAngle in the x-y plane. The fluid variables are set "
      "homogeneously on either side of the line x cos(angle) + y sin(angle) = "
      "0; velocities and magnetic fields are given in the interface's (normal, "
      "tangent, z) frame."};

  RotatedRiemannProblem() = default;
  RotatedRiemannProblem(const RotatedRiemannProblem& /*rhs*/) = default;
  RotatedRiemannProblem& operator=(const RotatedRiemannProblem& /*rhs*/) =
      default;
  RotatedRiemannProblem(RotatedRiemannProblem&& /*rhs*/) = default;
  RotatedRiemannProblem& operator=(RotatedRiemannProblem&& /*rhs*/) = default;
  ~RotatedRiemannProblem() override = default;

  RotatedRiemannProblem(double adiabatic_index, double left_rest_mass_density,
                        double right_rest_mass_density, double left_pressure,
                        double right_pressure,
                        const std::array<double, 3>& left_spatial_velocity,
                        const std::array<double, 3>& right_spatial_velocity,
                        const std::array<double, 3>& left_magnetic_field,
                        const std::array<double, 3>& right_magnetic_field,
                        double lapse, double shift, double interface_angle);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit RotatedRiemannProblem(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(RotatedRiemannProblem);
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
                 tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<gr::Tags::Lapse<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<gr::Tags::Shift<DataType, 3>> /*meta*/) const
      -> tuples::TaggedTuple<gr::Tags::Shift<DataType, 3>>;

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
  template <typename DataType, typename Tag>
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
  friend bool operator==(const RotatedRiemannProblem& lhs,
                         const RotatedRiemannProblem& rhs);

  friend bool operator!=(const RotatedRiemannProblem& lhs,
                         const RotatedRiemannProblem& rhs);

  /// A (normal, tangent, z) vector in the grid frame.
  std::array<double, 3> to_grid_frame(const std::array<double, 3>& v) const;

  /// The signed distance from the interface, s = x cos(angle) + y sin(angle).
  template <typename DataType>
  DataType signed_distance(const tnsr::I<DataType, 3>& x) const;

  EquationsOfState::IdealFluid<true> equation_of_state_{};
  gr::Solutions::Minkowski<3> background_spacetime_{};

  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  double left_rest_mass_density_ = std::numeric_limits<double>::signaling_NaN();
  double right_rest_mass_density_ =
      std::numeric_limits<double>::signaling_NaN();
  double left_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double right_pressure_ = std::numeric_limits<double>::signaling_NaN();
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
  double lapse_ = std::numeric_limits<double>::signaling_NaN();
  double shift_ = std::numeric_limits<double>::signaling_NaN();
  double interface_angle_ = std::numeric_limits<double>::signaling_NaN();
};

bool operator==(const RotatedRiemannProblem& lhs,
                const RotatedRiemannProblem& rhs);

bool operator!=(const RotatedRiemannProblem& lhs,
                const RotatedRiemannProblem& rhs);
}  // namespace grmhd::AnalyticData
