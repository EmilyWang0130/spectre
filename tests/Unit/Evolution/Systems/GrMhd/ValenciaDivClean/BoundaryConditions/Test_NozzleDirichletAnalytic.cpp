// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/HydroFreeOutflow.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/NozzleDirichletAnalytic.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Reflective.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
namespace {
using GhostFluxes =
    Variables<db::wrap_tags_in<::Tags::Flux, typename System::flux_variables,
                               tmpl::size_t<3>, Frame::Inertial>>;
using Solution = Solutions::SmoothFlow;

Solution solution() {
  return Solution{{{0.23, 0.01, 0.31}}, {{0.11, 0.23, 0.32}}, 1.0, 1.4, 0.3};
}

template <typename TensorType>
void check_point_equal(const TensorType& actual, const TensorType& expected,
                       const size_t point) {
  for (size_t component = 0; component < actual.size(); ++component) {
    CHECK(actual[component][point] == expected[component][point]);
  }
}

// The interior state of one element on the FD subcell grid.
struct Interior {
  Scalar<DataVector> rest_mass_density;
  Scalar<DataVector> electron_fraction;
  Scalar<DataVector> temperature;
  Scalar<DataVector> pressure;
  Scalar<DataVector> specific_internal_energy;
  Scalar<DataVector> lorentz_factor;
  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity;
  tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field;
  tnsr::I<DataVector, 3, Frame::Inertial> shift;
  Scalar<DataVector> lapse;
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric;
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric;
};

template <typename Generator>
Interior random_interior(const gsl::not_null<Generator*> generator,
                         const size_t number_of_points) {
  std::uniform_real_distribution<> positive(0.5, 1.0);
  std::uniform_real_distribution<> signed_small(-0.3, 0.3);
  const DataVector used_for_size(number_of_points);
  Interior interior{};
  interior.rest_mass_density = make_with_random_values<Scalar<DataVector>>(
      generator, positive, used_for_size);
  interior.electron_fraction = make_with_random_values<Scalar<DataVector>>(
      generator, positive, used_for_size);
  interior.temperature = make_with_random_values<Scalar<DataVector>>(
      generator, positive, used_for_size);
  interior.pressure = make_with_random_values<Scalar<DataVector>>(
      generator, positive, used_for_size);
  interior.specific_internal_energy =
      make_with_random_values<Scalar<DataVector>>(generator, positive,
                                                  used_for_size);
  interior.spatial_velocity =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          generator, signed_small, used_for_size);
  DataVector velocity_squared(number_of_points, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    velocity_squared += square(interior.spatial_velocity.get(i));
  }
  interior.lorentz_factor =
      Scalar<DataVector>{DataVector{1.0 / sqrt(1.0 - velocity_squared)}};
  interior.magnetic_field =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          generator, signed_small, used_for_size);
  interior.shift =
      tnsr::I<DataVector, 3, Frame::Inertial>{number_of_points, 0.0};
  interior.lapse = Scalar<DataVector>{number_of_points, 1.0};
  interior.spatial_metric =
      tnsr::ii<DataVector, 3, Frame::Inertial>{number_of_points, 0.0};
  interior.inv_spatial_metric =
      tnsr::II<DataVector, 3, Frame::Inertial>{number_of_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    interior.spatial_metric.get(i, i) = 1.0;
    interior.inv_spatial_metric.get(i, i) = 1.0;
  }
  return interior;
}

struct FdGhost {
  FdGhost(const size_t number_of_points, const bool with_fluxes)
      : rest_mass_density{number_of_points},
        electron_fraction{number_of_points},
        temperature{number_of_points},
        lorentz_factor_times_spatial_velocity{number_of_points},
        magnetic_field{number_of_points},
        divergence_cleaning_field{number_of_points} {
    if (with_fluxes) {
      fluxes.emplace(number_of_points);
    }
  }
  Scalar<DataVector> rest_mass_density;
  Scalar<DataVector> electron_fraction;
  Scalar<DataVector> temperature;
  tnsr::I<DataVector, 3, Frame::Inertial> lorentz_factor_times_spatial_velocity;
  tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field;
  Scalar<DataVector> divergence_cleaning_field;
  std::optional<GhostFluxes> fluxes{};
};

void test_fd(const NozzleOutside outside, const Direction<3>& direction,
             const double nozzle_radius,
             const gsl::not_null<std::mt19937*> generator) {
  CAPTURE(outside);
  CAPTURE(direction);
  CAPTURE(nozzle_radius);
  const Mesh<3> dg_mesh{3, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const fd::MonotonisedCentralPrim reconstructor{};
  const size_t ghost_zone_size = reconstructor.ghost_zone_size();
  const size_t normal_dimension = direction.dimension();
  const Index<3>& extents = subcell_mesh.extents();
  const size_t number_of_ghost_points =
      ghost_zone_size * extents.slice_away(normal_dimension).product();
  const double time = 0.5;
  const std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  // The element is [-1, 1]^3, so the subcell centers sit at x^0 = 0, +-0.4
  // and +-0.8, and a nozzle radius of 0.5 splits every face along x^1 or x^2.
  const ElementMap<3, Frame::Grid> logical_to_grid_map(
      ElementId<3>{0},
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          domain::CoordinateMaps::Identity<3>{}));
  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});
  const auto interior =
      random_interior(generator, subcell_mesh.number_of_grid_points());

  const NozzleDirichletAnalytic nozzle{std::make_unique<Solution>(solution()),
                                       nozzle_radius, outside};
  const auto clone = serialize_and_deserialize(nozzle);
  CHECK(clone.nozzle_radius() == nozzle_radius);
  CHECK(clone.outside() == outside);

  const auto ghost_inertial_coords = (*grid_to_inertial_map)(
      logical_to_grid_map(
          evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
              subcell_mesh, ghost_zone_size, direction)),
      time, functions_of_time);

  for (const bool with_fluxes : {false, true}) {
    CAPTURE(with_fluxes);
    if (with_fluxes and outside == NozzleOutside::Extrapolation) {
      FdGhost result{number_of_ghost_points, true};
      CHECK_THROWS_WITH(
          clone.fd_ghost(
              make_not_null(&result.rest_mass_density),
              make_not_null(&result.electron_fraction),
              make_not_null(&result.temperature),
              make_not_null(&result.lorentz_factor_times_spatial_velocity),
              make_not_null(&result.magnetic_field),
              make_not_null(&result.divergence_cleaning_field),
              make_not_null(&result.fluxes), direction, subcell_mesh,
              interior.shift, interior.lapse, interior.spatial_metric,
              interior.rest_mass_density, interior.electron_fraction,
              interior.temperature, interior.pressure,
              interior.specific_internal_energy, interior.lorentz_factor,
              interior.spatial_velocity, interior.magnetic_field, time,
              functions_of_time, logical_to_grid_map, *grid_to_inertial_map,
              reconstructor),
          Catch::Matchers::ContainsSubstring(
              "does not support cell-centered ghost fluxes"));
      continue;
    }
    FdGhost result{number_of_ghost_points, with_fluxes};
    clone.fd_ghost(make_not_null(&result.rest_mass_density),
                   make_not_null(&result.electron_fraction),
                   make_not_null(&result.temperature),
                   make_not_null(&result.lorentz_factor_times_spatial_velocity),
                   make_not_null(&result.magnetic_field),
                   make_not_null(&result.divergence_cleaning_field),
                   make_not_null(&result.fluxes), direction, subcell_mesh,
                   interior.shift, interior.lapse, interior.spatial_metric,
                   interior.rest_mass_density, interior.electron_fraction,
                   interior.temperature, interior.pressure,
                   interior.specific_internal_energy, interior.lorentz_factor,
                   interior.spatial_velocity, interior.magnetic_field, time,
                   functions_of_time, logical_to_grid_map,
                   *grid_to_inertial_map, reconstructor);

    FdGhost dirichlet{number_of_ghost_points, with_fluxes};
    DirichletAnalytic{std::make_unique<Solution>(solution())}.fd_ghost(
        make_not_null(&dirichlet.rest_mass_density),
        make_not_null(&dirichlet.electron_fraction),
        make_not_null(&dirichlet.temperature),
        make_not_null(&dirichlet.lorentz_factor_times_spatial_velocity),
        make_not_null(&dirichlet.magnetic_field),
        make_not_null(&dirichlet.divergence_cleaning_field),
        make_not_null(&dirichlet.fluxes), direction, subcell_mesh, time,
        functions_of_time, logical_to_grid_map, *grid_to_inertial_map,
        reconstructor);

    FdGhost outer{number_of_ghost_points, with_fluxes};
    if (outside == NozzleOutside::Reflective or
        outside == NozzleOutside::ReflectiveBoth) {
      Reflective{outside == NozzleOutside::ReflectiveBoth}.fd_ghost(
          make_not_null(&outer.rest_mass_density),
          make_not_null(&outer.electron_fraction),
          make_not_null(&outer.temperature),
          make_not_null(&outer.lorentz_factor_times_spatial_velocity),
          make_not_null(&outer.magnetic_field),
          make_not_null(&outer.divergence_cleaning_field),
          make_not_null(&outer.fluxes), direction, subcell_mesh, interior.shift,
          interior.lapse, interior.spatial_metric, interior.rest_mass_density,
          interior.electron_fraction, interior.temperature, interior.pressure,
          interior.specific_internal_energy, interior.lorentz_factor,
          interior.spatial_velocity, interior.magnetic_field, reconstructor);
    } else {
      HydroFreeOutflow::fd_ghost(
          make_not_null(&outer.rest_mass_density),
          make_not_null(&outer.electron_fraction),
          make_not_null(&outer.temperature),
          make_not_null(&outer.lorentz_factor_times_spatial_velocity),
          make_not_null(&outer.magnetic_field),
          make_not_null(&outer.divergence_cleaning_field),
          make_not_null(&outer.fluxes), direction, subcell_mesh, interior.shift,
          interior.lapse, interior.spatial_metric, interior.rest_mass_density,
          interior.electron_fraction, interior.temperature, interior.pressure,
          interior.specific_internal_energy, interior.lorentz_factor,
          interior.spatial_velocity, interior.magnetic_field, reconstructor);
    }

    // The ghost data are laid out like the subcell grid with the normal
    // extent replaced by the ghost zone size. For Extrapolation, a ghost
    // point's normal W v^i is that of the boundary cell it sits beyond.
    Index<3> ghost_extents = extents;
    ghost_extents[normal_dimension] = ghost_zone_size;
    size_t number_inside = 0;
    for (size_t point = 0; point < number_of_ghost_points; ++point) {
      CAPTURE(point);
      const bool inside =
          std::abs(get<0>(ghost_inertial_coords)[point]) <= nozzle_radius;
      number_inside += inside ? 1 : 0;
      const FdGhost& expected = inside ? dirichlet : outer;
      check_point_equal(result.rest_mass_density, expected.rest_mass_density,
                        point);
      check_point_equal(result.electron_fraction, expected.electron_fraction,
                        point);
      check_point_equal(result.temperature, expected.temperature, point);
      check_point_equal(result.magnetic_field, expected.magnetic_field, point);
      check_point_equal(result.divergence_cleaning_field,
                        expected.divergence_cleaning_field, point);
      for (size_t i = 0; i < 3; ++i) {
        if (i == normal_dimension and not inside and
            outside == NozzleOutside::Extrapolation) {
          std::array<size_t, 3> index{};
          index[0] = point % ghost_extents[0];
          index[1] = (point / ghost_extents[0]) % ghost_extents[1];
          index[2] = point / (ghost_extents[0] * ghost_extents[1]);
          index[normal_dimension] = direction.side() == Side::Lower
                                        ? 0
                                        : extents[normal_dimension] - 1;
          const size_t volume_point =
              index[0] + extents[0] * (index[1] + extents[1] * index[2]);
          CHECK(result.lorentz_factor_times_spatial_velocity.get(i)[point] ==
                approx(get(interior.lorentz_factor)[volume_point] *
                       interior.spatial_velocity.get(i)[volume_point]));
        } else {
          CHECK(result.lorentz_factor_times_spatial_velocity.get(i)[point] ==
                expected.lorentz_factor_times_spatial_velocity.get(i)[point]);
        }
      }
      if (with_fluxes) {
        const size_t number_of_components =
            result.fluxes->number_of_independent_components;
        for (size_t component = 0; component < number_of_components;
             ++component) {
          CHECK(result.fluxes
                    ->data()[component * number_of_ghost_points + point] ==
                expected.fluxes
                    ->data()[component * number_of_ghost_points + point]);
        }
      }
    }
    if (normal_dimension == 0) {
      CHECK(number_inside == 0);
    } else if (nozzle_radius > 1.0) {
      CHECK(number_inside == number_of_ghost_points);
    } else {
      // Three of the five subcell columns along x^0 are inside 0.5.
      CHECK(number_inside == 3 * number_of_ghost_points / 5);
    }
  }
}

// The outputs of a DG ghost call.
struct DgGhost {
  explicit DgGhost(const size_t n)
      : tilde_d{n},
        tilde_ye{n},
        tilde_tau{n},
        tilde_s{n},
        tilde_b{n},
        tilde_phi{n},
        tilde_d_flux{n},
        tilde_ye_flux{n},
        tilde_tau_flux{n},
        tilde_s_flux{n},
        tilde_b_flux{n},
        tilde_phi_flux{n},
        lapse{n},
        shift{n},
        spatial_velocity_one_form{n},
        spatial_metric{n},
        rest_mass_density{n},
        electron_fraction{n},
        temperature{n},
        spatial_velocity{n},
        specific_internal_energy{n},
        pressure{n},
        lorentz_factor{n},
        inv_spatial_metric{n} {}
  Scalar<DataVector> tilde_d;
  Scalar<DataVector> tilde_ye;
  Scalar<DataVector> tilde_tau;
  tnsr::i<DataVector, 3, Frame::Inertial> tilde_s;
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_b;
  Scalar<DataVector> tilde_phi;
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_d_flux;
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_ye_flux;
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_tau_flux;
  tnsr::Ij<DataVector, 3, Frame::Inertial> tilde_s_flux;
  tnsr::IJ<DataVector, 3, Frame::Inertial> tilde_b_flux;
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_phi_flux;
  Scalar<DataVector> lapse;
  tnsr::I<DataVector, 3, Frame::Inertial> shift;
  tnsr::i<DataVector, 3, Frame::Inertial> spatial_velocity_one_form;
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric;
  Scalar<DataVector> rest_mass_density;
  Scalar<DataVector> electron_fraction;
  Scalar<DataVector> temperature;
  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity;
  Scalar<DataVector> specific_internal_energy;
  Scalar<DataVector> pressure;
  Scalar<DataVector> lorentz_factor;
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric;
};

void check_dg_point(const DgGhost& actual, const DgGhost& expected,
                    const size_t point) {
  check_point_equal(actual.tilde_d, expected.tilde_d, point);
  check_point_equal(actual.tilde_ye, expected.tilde_ye, point);
  check_point_equal(actual.tilde_tau, expected.tilde_tau, point);
  check_point_equal(actual.tilde_s, expected.tilde_s, point);
  check_point_equal(actual.tilde_b, expected.tilde_b, point);
  check_point_equal(actual.tilde_phi, expected.tilde_phi, point);
  check_point_equal(actual.tilde_d_flux, expected.tilde_d_flux, point);
  check_point_equal(actual.tilde_ye_flux, expected.tilde_ye_flux, point);
  check_point_equal(actual.tilde_tau_flux, expected.tilde_tau_flux, point);
  check_point_equal(actual.tilde_s_flux, expected.tilde_s_flux, point);
  check_point_equal(actual.tilde_b_flux, expected.tilde_b_flux, point);
  check_point_equal(actual.tilde_phi_flux, expected.tilde_phi_flux, point);
  check_point_equal(actual.lapse, expected.lapse, point);
  check_point_equal(actual.shift, expected.shift, point);
  check_point_equal(actual.spatial_velocity_one_form,
                    expected.spatial_velocity_one_form, point);
  check_point_equal(actual.spatial_metric, expected.spatial_metric, point);
  check_point_equal(actual.rest_mass_density, expected.rest_mass_density,
                    point);
  check_point_equal(actual.electron_fraction, expected.electron_fraction,
                    point);
  check_point_equal(actual.temperature, expected.temperature, point);
  check_point_equal(actual.spatial_velocity, expected.spatial_velocity, point);
  check_point_equal(actual.specific_internal_energy,
                    expected.specific_internal_energy, point);
  check_point_equal(actual.pressure, expected.pressure, point);
  check_point_equal(actual.lorentz_factor, expected.lorentz_factor, point);
  check_point_equal(actual.inv_spatial_metric, expected.inv_spatial_metric,
                    point);
}

void test_dg(const NozzleOutside outside,
             const gsl::not_null<std::mt19937*> generator) {
  CAPTURE(outside);
  // A 3 x 3 face normal to x^1 with x^0 = -1, 0, 1 and nozzle radius 0.5, so
  // the middle column is inside.
  const size_t number_of_points = 9;
  const double nozzle_radius = 0.5;
  const double time = 0.5;
  tnsr::I<DataVector, 3, Frame::Inertial> coords{number_of_points, -1.0};
  for (size_t point = 0; point < number_of_points; ++point) {
    get<0>(coords)[point] = static_cast<double>(point % 3) - 1.0;
    get<2>(coords)[point] = static_cast<double>(point / 3) - 1.0;
  }
  tnsr::i<DataVector, 3, Frame::Inertial> normal_covector{number_of_points,
                                                          0.0};
  get<1>(normal_covector) = -1.0;
  tnsr::I<DataVector, 3, Frame::Inertial> normal_vector{number_of_points, 0.0};
  get<1>(normal_vector) = -1.0;
  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>
      face_mesh_velocity{};
  const auto interior = random_interior(generator, number_of_points);

  const NozzleDirichletAnalytic nozzle{std::make_unique<Solution>(solution()),
                                       nozzle_radius, outside};
  DgGhost result{number_of_points};
  const auto error = nozzle.dg_ghost(
      make_not_null(&result.tilde_d), make_not_null(&result.tilde_ye),
      make_not_null(&result.tilde_tau), make_not_null(&result.tilde_s),
      make_not_null(&result.tilde_b), make_not_null(&result.tilde_phi),
      make_not_null(&result.tilde_d_flux), make_not_null(&result.tilde_ye_flux),
      make_not_null(&result.tilde_tau_flux),
      make_not_null(&result.tilde_s_flux), make_not_null(&result.tilde_b_flux),
      make_not_null(&result.tilde_phi_flux), make_not_null(&result.lapse),
      make_not_null(&result.shift),
      make_not_null(&result.spatial_velocity_one_form),
      make_not_null(&result.spatial_metric),
      make_not_null(&result.rest_mass_density),
      make_not_null(&result.electron_fraction),
      make_not_null(&result.temperature),
      make_not_null(&result.spatial_velocity),
      make_not_null(&result.specific_internal_energy),
      make_not_null(&result.pressure), make_not_null(&result.lorentz_factor),
      make_not_null(&result.inv_spatial_metric), face_mesh_velocity,
      normal_covector, normal_vector, interior.rest_mass_density,
      interior.electron_fraction, interior.specific_internal_energy,
      interior.spatial_velocity, interior.magnetic_field,
      interior.lorentz_factor, interior.pressure, interior.temperature,
      interior.shift, interior.lapse, interior.inv_spatial_metric, coords,
      time);
  CHECK(not error.has_value());

  DgGhost dirichlet{number_of_points};
  CHECK(not DirichletAnalytic{std::make_unique<Solution>(solution())}
                .dg_ghost(make_not_null(&dirichlet.tilde_d),
                          make_not_null(&dirichlet.tilde_ye),
                          make_not_null(&dirichlet.tilde_tau),
                          make_not_null(&dirichlet.tilde_s),
                          make_not_null(&dirichlet.tilde_b),
                          make_not_null(&dirichlet.tilde_phi),
                          make_not_null(&dirichlet.tilde_d_flux),
                          make_not_null(&dirichlet.tilde_ye_flux),
                          make_not_null(&dirichlet.tilde_tau_flux),
                          make_not_null(&dirichlet.tilde_s_flux),
                          make_not_null(&dirichlet.tilde_b_flux),
                          make_not_null(&dirichlet.tilde_phi_flux),
                          make_not_null(&dirichlet.lapse),
                          make_not_null(&dirichlet.shift),
                          make_not_null(&dirichlet.spatial_velocity_one_form),
                          make_not_null(&dirichlet.spatial_metric),
                          make_not_null(&dirichlet.rest_mass_density),
                          make_not_null(&dirichlet.electron_fraction),
                          make_not_null(&dirichlet.temperature),
                          make_not_null(&dirichlet.spatial_velocity),
                          make_not_null(&dirichlet.specific_internal_energy),
                          make_not_null(&dirichlet.pressure),
                          make_not_null(&dirichlet.lorentz_factor),
                          make_not_null(&dirichlet.inv_spatial_metric),
                          face_mesh_velocity, normal_covector, normal_vector,
                          coords, time)
                .has_value());

  DgGhost outer{number_of_points};
  const auto call_outer = [&](const auto& condition) {
    return condition.dg_ghost(
        make_not_null(&outer.tilde_d), make_not_null(&outer.tilde_ye),
        make_not_null(&outer.tilde_tau), make_not_null(&outer.tilde_s),
        make_not_null(&outer.tilde_b), make_not_null(&outer.tilde_phi),
        make_not_null(&outer.tilde_d_flux), make_not_null(&outer.tilde_ye_flux),
        make_not_null(&outer.tilde_tau_flux),
        make_not_null(&outer.tilde_s_flux), make_not_null(&outer.tilde_b_flux),
        make_not_null(&outer.tilde_phi_flux), make_not_null(&outer.lapse),
        make_not_null(&outer.shift),
        make_not_null(&outer.spatial_velocity_one_form),
        make_not_null(&outer.spatial_metric),
        make_not_null(&outer.rest_mass_density),
        make_not_null(&outer.electron_fraction),
        make_not_null(&outer.temperature),
        make_not_null(&outer.spatial_velocity),
        make_not_null(&outer.specific_internal_energy),
        make_not_null(&outer.pressure), make_not_null(&outer.lorentz_factor),
        make_not_null(&outer.inv_spatial_metric), face_mesh_velocity,
        normal_covector, normal_vector, interior.rest_mass_density,
        interior.electron_fraction, interior.specific_internal_energy,
        interior.spatial_velocity, interior.magnetic_field,
        interior.lorentz_factor, interior.pressure, interior.temperature,
        interior.shift, interior.lapse, interior.inv_spatial_metric);
  };
  if (outside == NozzleOutside::Reflective or
      outside == NozzleOutside::ReflectiveBoth) {
    CHECK(not call_outer(Reflective{outside == NozzleOutside::ReflectiveBoth})
                  .has_value());
  } else {
    CHECK(not call_outer(HydroFreeOutflow{}).has_value());
  }

  for (size_t point = 0; point < number_of_points; ++point) {
    CAPTURE(point);
    const bool inside = std::abs(get<0>(coords)[point]) <= nozzle_radius;
    CHECK(inside == (point % 3 == 1));
    check_dg_point(result, inside ? dirichlet : outer, point);
  }
}

void test_options() {
  CHECK(TestHelpers::test_creation<NozzleOutside>("Extrapolation") ==
        NozzleOutside::Extrapolation);
  CHECK(TestHelpers::test_creation<NozzleOutside>("FreeOutflow") ==
        NozzleOutside::FreeOutflow);
  CHECK(TestHelpers::test_creation<NozzleOutside>("Reflective") ==
        NozzleOutside::Reflective);
  CHECK(TestHelpers::test_creation<NozzleOutside>("ReflectiveBoth") ==
        NozzleOutside::ReflectiveBoth);
  CHECK_THROWS_WITH(TestHelpers::test_creation<NozzleOutside>("Wall"),
                    Catch::Matchers::ContainsSubstring(
                        "Failed to convert \"Wall\" to NozzleOutside"));
  CHECK(get_output(NozzleOutside::Extrapolation) == "Extrapolation");
  CHECK(get_output(NozzleOutside::ReflectiveBoth) == "ReflectiveBoth");
}

SPECTRE_TEST_CASE("Unit.GrMhd.BoundaryConditions.NozzleDirichletAnalytic",
                  "[Unit][GrMhd]") {
  register_classes_with_charm(tmpl::list<Solution>{});
  MAKE_GENERATOR(generator);
  test_options();
  for (const auto outside :
       {NozzleOutside::Extrapolation, NozzleOutside::FreeOutflow,
        NozzleOutside::Reflective, NozzleOutside::ReflectiveBoth}) {
    for (const auto& direction :
         {Direction<3>::lower_eta(), Direction<3>::upper_eta(),
          Direction<3>::lower_zeta(), Direction<3>::lower_xi()}) {
      test_fd(outside, direction, 0.5, make_not_null(&generator));
    }
    test_fd(outside, Direction<3>::lower_eta(), 2.0, make_not_null(&generator));
    test_dg(outside, make_not_null(&generator));
  }
}
}  // namespace
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
