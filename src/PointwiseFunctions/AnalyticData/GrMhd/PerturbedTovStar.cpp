// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/PerturbedTovStar.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <pup.h>
#include <pup_stl.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace grmhd::AnalyticData {

ModeProfile::ModeProfile(const std::string& filename) {
  std::ifstream file(filename);
  if (not file.is_open()) {
    ERROR("PerturbedTovStar: cannot open the mode profile '" << filename
                                                             << "'.");
  }
  std::string line{};
  size_t num_points = 0;
  size_t line_number = 0;
  bool header_read = false;
  std::vector<double> radius{};
  std::vector<double> xi_r{};
  std::vector<double> xi_perp{};
  while (std::getline(file, line)) {
    ++line_number;
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos or line[first] == '#') {
      continue;
    }
    std::istringstream stream(line);
    if (not header_read) {
      stream >> l_ >> omega_ >> num_points;
      if (stream.fail()) {
        ERROR("PerturbedTovStar: expected 'l omega n' on line "
              << line_number << " of '" << filename << "', got '" << line
              << "'.");
      }
      header_read = true;
      radius.reserve(num_points);
      xi_r.reserve(num_points);
      xi_perp.reserve(num_points);
      continue;
    }
    double r = 0.0;
    double xr = 0.0;
    double xp = 0.0;
    stream >> r >> xr >> xp;
    if (stream.fail()) {
      ERROR("PerturbedTovStar: expected 'r xi_r xi_perp' on line "
            << line_number << " of '" << filename << "', got '" << line
            << "'.");
    }
    radius.push_back(r);
    xi_r.push_back(xr);
    xi_perp.push_back(xp);
  }
  if (not header_read) {
    ERROR("PerturbedTovStar: no data in the mode profile '" << filename
                                                            << "'.");
  }
  if (radius.size() != num_points) {
    ERROR("PerturbedTovStar: the header of '"
          << filename << "' announces " << num_points << " rows but "
          << radius.size() << " were read.");
  }
  initialize(std::move(radius), std::move(xi_r), std::move(xi_perp));
}

ModeProfile::ModeProfile(const size_t l, const double omega,
                         std::vector<double> radius, std::vector<double> xi_r,
                         std::vector<double> xi_perp)
    : l_(l), omega_(omega) {
  initialize(std::move(radius), std::move(xi_r), std::move(xi_perp));
}

void ModeProfile::initialize(std::vector<double> radius,
                             std::vector<double> xi_r,
                             std::vector<double> xi_perp) {
  if (l_ != 2) {
    ERROR("PerturbedTovStar: only l = 2 is implemented, the profile has l = "
          << l_ << ".");
  }
  if (not(omega_ > 0.0)) {
    ERROR("PerturbedTovStar: the mode frequency must be positive, got omega = "
          << omega_ << ".");
  }
  if (radius.size() < 4 or radius.size() != xi_r.size() or
      radius.size() != xi_perp.size()) {
    ERROR(
        "PerturbedTovStar: the mode profile needs at least 4 rows with "
        "matching column lengths, got "
        << radius.size() << ", " << xi_r.size() << ", " << xi_perp.size()
        << ".");
  }
  if (radius.front() != 0.0) {
    ERROR(
        "PerturbedTovStar: the mode profile must start at r = 0, it starts "
        "at r = "
        << radius.front() << ".");
  }
  for (size_t i = 1; i < radius.size(); ++i) {
    if (radius[i] <= radius[i - 1]) {
      ERROR("PerturbedTovStar: the profile radius must increase strictly; row "
            << i << " has r = " << radius[i] << " after r = " << radius[i - 1]
            << ".");
    }
  }
  xi_r_interpolant_ = intrp::CubicSpline(radius, std::move(xi_r));
  xi_perp_interpolant_ =
      intrp::CubicSpline(std::move(radius), std::move(xi_perp));
}

double ModeProfile::xi_r(const double r) const {
  if (r < 0.0 or r > outer_radius()) {
    return 0.0;
  }
  return xi_r_interpolant_(r);
}

double ModeProfile::xi_perp(const double r) const {
  if (r < 0.0 or r > outer_radius()) {
    return 0.0;
  }
  return xi_perp_interpolant_(r);
}

void ModeProfile::pup(PUP::er& p) {
  p | l_;
  p | omega_;
  p | xi_r_interpolant_;
  p | xi_perp_interpolant_;
}

bool operator==(const ModeProfile& lhs, const ModeProfile& rhs) {
  return lhs.l_ == rhs.l_ and lhs.omega_ == rhs.omega_ and
         lhs.xi_r_interpolant_.x_values() ==
             rhs.xi_r_interpolant_.x_values() and
         lhs.xi_r_interpolant_.y_values() ==
             rhs.xi_r_interpolant_.y_values() and
         lhs.xi_perp_interpolant_.y_values() ==
             rhs.xi_perp_interpolant_.y_values();
}

bool operator!=(const ModeProfile& lhs, const ModeProfile& rhs) {
  return not(lhs == rhs);
}

namespace perturbed_tov_detail {

template <typename DataType, StarRegion Region>
void PerturbedTovVariables<DataType, Region>::displacement(
    const gsl::not_null<tnsr::I<DataType, 3>*> xi,
    const gsl::not_null<Scalar<DataType>*> xi_radial) const {
  const size_t num_points = get_size(radius);
  set_number_of_grid_points(xi, num_points);
  set_number_of_grid_points(xi_radial, num_points);
  for (size_t p = 0; p < num_points; ++p) {
    const double r = get_element(radius, p);
    const double xr = mode.xi_r(r);
    const double xp = mode.xi_perp(r);
    if (r <= 0.0 or (xr == 0.0 and xp == 0.0)) {
      get_element(get(*xi_radial), p) = 0.0;
      for (size_t i = 0; i < 3; ++i) {
        get_element(xi->get(i), p) = 0.0;
      }
      continue;
    }
    std::array<double, 3> r_hat{};
    double mu = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(r_hat, i) = get_element(coords.get(i), p) / r;
      mu += gsl::at(r_hat, i) * gsl::at(polar_axis, i);
    }
    // l = 2 only:  P_2 = (3 mu^2 - 1) / 2  and, writing theta-hat as
    // (mu r-hat - n) / sin(theta),  (dP_2/dtheta) theta-hat = -3 mu (mu r-hat
    // - n): the sin(theta) cancels, so the axis is regular.
    const double p2 = 0.5 * (3.0 * mu * mu - 1.0);
    const double radial_part = xr * p2 - 3.0 * mu * mu * xp;
    get_element(get(*xi_radial), p) = xr * p2;
    for (size_t i = 0; i < 3; ++i) {
      get_element(xi->get(i), p) = radial_part * gsl::at(r_hat, i) +
                                   3.0 * mu * xp * gsl::at(polar_axis, i);
    }
  }
}

template <typename DataType, StarRegion Region>
void PerturbedTovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> electron_fraction,
    const gsl::not_null<Cache*> cache,
    hydro::Tags::ElectronFraction<DataType> /*meta*/) const {
  Base::operator()(electron_fraction, cache,
                   hydro::Tags::ElectronFraction<DataType>{});
  if constexpr (Region == StarRegion::Interior) {
    if (displacement_amplitude == 0.0 or yeq_eos == nullptr) {
      return;
    }
    tnsr::I<DataType, 3> xi{};
    Scalar<DataType> xi_radial{};
    displacement(make_not_null(&xi), make_not_null(&xi_radial));

    // With frozen composition the fluid element now at `radius` carries the
    // beta-equilibrium Y_e of its source radius `radius - xi_r P_l`. Look the
    // background up there through the same enthalpy chain the base class
    // uses. Points whose source falls outside the star keep the background
    // value: for those, look up their own (always valid) radius and discard.
    const double outer_radius = radial_solution.outer_radius();
    const size_t num_points = get_size(radius);
    DataType source_radius = radius;
    std::vector<bool> source_inside(num_points, false);
    for (size_t p = 0; p < num_points; ++p) {
      const double r_source =
          get_element(radius, p) -
          displacement_amplitude * get_element(get(xi_radial), p);
      if (r_source > 0.0 and r_source < outer_radius) {
        get_element(source_radius, p) = r_source;
        source_inside[p] = true;
      }
    }
    const Scalar<DataType> source_enthalpy{
        exp(radial_solution.log_specific_enthalpy(source_radius))};
    const Scalar<DataType> source_density =
        eos.rest_mass_density_from_enthalpy(source_enthalpy);
    const Scalar<DataType> temperature = make_with_value<Scalar<DataType>>(
        source_density, yeq_eos->temperature_lower_bound());
    const Scalar<DataType> source_electron_fraction =
        yeq_eos->equilibrium_electron_fraction_from_density_temperature(
            source_density, temperature);
    for (size_t p = 0; p < num_points; ++p) {
      if (source_inside[p]) {
        get_element(get(*electron_fraction), p) =
            get_element(get(source_electron_fraction), p);
      }
    }
  }
}

template <typename DataType, StarRegion Region>
void PerturbedTovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> spatial_velocity,
    const gsl::not_null<Cache*> cache,
    hydro::Tags::SpatialVelocity<DataType, 3> /*meta*/) const {
  Base::operator()(spatial_velocity, cache,
                   hydro::Tags::SpatialVelocity<DataType, 3>{});
  if constexpr (Region == StarRegion::Interior) {
    if (velocity_amplitude == 0.0) {
      return;
    }
    tnsr::I<DataType, 3> xi{};
    Scalar<DataType> xi_radial{};
    displacement(make_not_null(&xi), make_not_null(&xi_radial));
    for (size_t i = 0; i < 3; ++i) {
      spatial_velocity->get(i) = velocity_amplitude * mode.omega() * xi.get(i);
    }
  }
}

template <typename DataType, StarRegion Region>
void PerturbedTovVariables<DataType, Region>::operator()(
    const gsl::not_null<Scalar<DataType>*> lorentz_factor,
    const gsl::not_null<Cache*> cache,
    hydro::Tags::LorentzFactor<DataType> /*meta*/) const {
  if constexpr (Region == StarRegion::Interior) {
    if (velocity_amplitude != 0.0) {
      const auto& spatial_velocity =
          cache->get_var(*this, hydro::Tags::SpatialVelocity<DataType, 3>{});
      const auto& spatial_metric =
          cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3>{});
      get(*lorentz_factor) =
          1.0 / sqrt(1.0 - get(dot_product(spatial_velocity, spatial_velocity,
                                           spatial_metric)));
      return;
    }
  }
  Base::operator()(lorentz_factor, cache,
                   hydro::Tags::LorentzFactor<DataType>{});
}

}  // namespace perturbed_tov_detail

PerturbedTovStar::PerturbedTovStar() = default;
PerturbedTovStar::PerturbedTovStar(PerturbedTovStar&& /*rhs*/) = default;
PerturbedTovStar& PerturbedTovStar::operator=(PerturbedTovStar&& /*rhs*/) =
    default;
PerturbedTovStar::~PerturbedTovStar() = default;

PerturbedTovStar::PerturbedTovStar(
    const double central_rest_mass_density,
    std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>
        equation_of_state,
    const RelativisticEuler::Solutions::TovCoordinates coordinate_system,
    std::optional<std::unique_ptr<EquationsOfState::EquationOfState<true, 3>>>
        yeq_eos,
    const std::string& mode_profile_file, const double displacement_amplitude,
    const double velocity_amplitude, const std::array<double, 3>& polar_axis)
    : tov_star(central_rest_mass_density, std::move(equation_of_state),
               coordinate_system, std::move(yeq_eos)),
      mode_(mode_profile_file),
      mode_profile_file_(mode_profile_file),
      displacement_amplitude_(displacement_amplitude),
      velocity_amplitude_(velocity_amplitude),
      polar_axis_(polar_axis) {
  validate();
}

PerturbedTovStar::PerturbedTovStar(
    const double central_rest_mass_density,
    std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>
        equation_of_state,
    const RelativisticEuler::Solutions::TovCoordinates coordinate_system,
    std::optional<std::unique_ptr<EquationsOfState::EquationOfState<true, 3>>>
        yeq_eos,
    ModeProfile mode, const double displacement_amplitude,
    const double velocity_amplitude, const std::array<double, 3>& polar_axis)
    : tov_star(central_rest_mass_density, std::move(equation_of_state),
               coordinate_system, std::move(yeq_eos)),
      mode_(std::move(mode)),
      displacement_amplitude_(displacement_amplitude),
      velocity_amplitude_(velocity_amplitude),
      polar_axis_(polar_axis) {
  validate();
}

PerturbedTovStar::PerturbedTovStar(const PerturbedTovStar& rhs)
    : evolution::initial_data::InitialData{rhs},
      tov_star(static_cast<const tov_star&>(rhs)),
      mode_(rhs.mode_),
      mode_profile_file_(rhs.mode_profile_file_),
      displacement_amplitude_(rhs.displacement_amplitude_),
      velocity_amplitude_(rhs.velocity_amplitude_),
      polar_axis_(rhs.polar_axis_) {}

PerturbedTovStar& PerturbedTovStar::operator=(const PerturbedTovStar& rhs) {
  if (this == &rhs) {
    return *this;
  }
  static_cast<tov_star&>(*this) = static_cast<const tov_star&>(rhs);
  mode_ = rhs.mode_;
  mode_profile_file_ = rhs.mode_profile_file_;
  displacement_amplitude_ = rhs.displacement_amplitude_;
  velocity_amplitude_ = rhs.velocity_amplitude_;
  polar_axis_ = rhs.polar_axis_;
  return *this;
}

void PerturbedTovStar::validate() {
  const double norm = sqrt(square(polar_axis_[0]) + square(polar_axis_[1]) +
                           square(polar_axis_[2]));
  if (not(norm > 1.0e-12)) {
    ERROR("PerturbedTovStar: PolarAxis must be a nonzero vector.");
  }
  for (auto& component : polar_axis_) {
    component /= norm;
  }
  const double star_radius = radial_solution().outer_radius();
  const double profile_radius = mode_.outer_radius();
  if (std::abs(profile_radius / star_radius - 1.0) > 0.02) {
    ERROR("PerturbedTovStar: the mode profile ends at r = "
          << profile_radius
          << " but the star's outer radius in the selected coordinates is "
          << star_radius << " (ratio " << profile_radius / star_radius
          << "). A profile tabulated on the areal radius handed to an "
             "isotropic-coordinate star, or vice versa, fails this check; "
             "regenerate the profile in the coordinates the input file "
             "selects.");
  }
}

std::unique_ptr<evolution::initial_data::InitialData>
PerturbedTovStar::get_clone() const {
  return std::make_unique<PerturbedTovStar>(*this);
}

PerturbedTovStar::PerturbedTovStar(CkMigrateMessage* msg) : tov_star(msg) {}

void PerturbedTovStar::pup(PUP::er& p) {
  tov_star::pup(p);
  p | mode_;
  p | mode_profile_file_;
  p | displacement_amplitude_;
  p | velocity_amplitude_;
  p | polar_axis_;
}

PUP::able::PUP_ID PerturbedTovStar::my_PUP_ID = 0;

bool operator==(const PerturbedTovStar& lhs, const PerturbedTovStar& rhs) {
  return static_cast<const typename PerturbedTovStar::tov_star&>(lhs) ==
             static_cast<const typename PerturbedTovStar::tov_star&>(rhs) and
         lhs.mode_ == rhs.mode_ and
         lhs.displacement_amplitude_ == rhs.displacement_amplitude_ and
         lhs.velocity_amplitude_ == rhs.velocity_amplitude_ and
         lhs.polar_axis_ == rhs.polar_axis_;
}

bool operator!=(const PerturbedTovStar& lhs, const PerturbedTovStar& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define REGION(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                               \
  template struct perturbed_tov_detail::PerturbedTovVariables<DTYPE(data), \
                                                              REGION(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector),
                        (perturbed_tov_detail::StarRegion::Center,
                         perturbed_tov_detail::StarRegion::Interior,
                         perturbed_tov_detail::StarRegion::Exterior))

#undef INSTANTIATE
#undef DTYPE
#undef REGION
}  // namespace grmhd::AnalyticData
