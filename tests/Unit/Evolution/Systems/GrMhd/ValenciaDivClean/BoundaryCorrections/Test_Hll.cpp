// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace {
namespace helpers = TestHelpers::evolution::dg;

struct ConvertPolytropic {
  using unpacked_container = bool;
  using packed_container = EquationsOfState::EquationOfState<true, 3>;
  using packed_type = bool;

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    return true;
  }

  [[noreturn]] static inline void pack(
      const gsl::not_null<packed_container*> /*packed*/,
      const unpacked_container& /*unpacked*/,
      const size_t /*grid_point_index*/) {
    ERROR("Should not be converting an EOS from an unpacked to a packed type");
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

struct PackagedSpeeds {
  Scalar<DataVector> largest_out{};
  Scalar<DataVector> largest_in{};
  Scalar<DataVector> fast_out{};
  Scalar<DataVector> fast_in{};
  Scalar<DataVector> flatness{};
};

// Package the speeds of a flat-space state with zero flux data, the given
// velocity, field and unit normal, and the polytrope's temperature of zero.
PackagedSpeeds package_speeds(
    const Scalar<DataVector>& rest_mass_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const EquationsOfState::EquationOfState<true, 3>& eos) {
  const size_t num_points = get(rest_mass_density).size();
  const Scalar<DataVector> electron_fraction{DataVector{num_points, 0.1}};
  const Scalar<DataVector> temperature{DataVector{num_points, 0.0}};
  const Scalar<DataVector> specific_internal_energy =
      eos.specific_internal_energy_from_density_and_temperature(
          rest_mass_density, temperature, electron_fraction);
  const Scalar<DataVector> pressure = eos.pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy, electron_fraction);
  tnsr::i<DataVector, 3, Frame::Inertial> spatial_velocity_one_form{num_points,
                                                                    0.0};
  DataVector v_squared{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    spatial_velocity_one_form.get(i) = spatial_velocity.get(i);
    v_squared += spatial_velocity.get(i) * spatial_velocity.get(i);
  }
  const Scalar<DataVector> lorentz_factor{1.0 / sqrt(1.0 - v_squared)};
  const Scalar<DataVector> lapse{DataVector{num_points, 1.0}};
  const tnsr::I<DataVector, 3, Frame::Inertial> shift{num_points, 0.0};
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = 1.0;
  }
  const tnsr::I<DataVector, 3, Frame::Inertial> normal_vector{num_points, 0.0};

  const Scalar<DataVector> zero_scalar{DataVector{num_points, 0.0}};
  const tnsr::i<DataVector, 3, Frame::Inertial> zero_lower{num_points, 0.0};
  const tnsr::I<DataVector, 3, Frame::Inertial> zero_upper{num_points, 0.0};
  const tnsr::Ij<DataVector, 3, Frame::Inertial> zero_mixed{num_points, 0.0};
  const tnsr::IJ<DataVector, 3, Frame::Inertial> zero_upper2{num_points, 0.0};

  Scalar<DataVector> p_tilde_d{num_points, 0.0};
  Scalar<DataVector> p_tilde_ye{num_points, 0.0};
  Scalar<DataVector> p_tilde_tau{num_points, 0.0};
  Scalar<DataVector> p_tilde_phi{num_points, 0.0};
  Scalar<DataVector> p_nf_d{num_points, 0.0};
  Scalar<DataVector> p_nf_ye{num_points, 0.0};
  Scalar<DataVector> p_nf_tau{num_points, 0.0};
  Scalar<DataVector> p_nf_phi{num_points, 0.0};
  tnsr::i<DataVector, 3, Frame::Inertial> p_tilde_s{num_points, 0.0};
  tnsr::i<DataVector, 3, Frame::Inertial> p_nf_s{num_points, 0.0};
  tnsr::i<DataVector, 3, Frame::Inertial> p_normal{num_points, 0.0};
  tnsr::I<DataVector, 3, Frame::Inertial> p_tilde_b{num_points, 0.0};
  tnsr::I<DataVector, 3, Frame::Inertial> p_nf_b{num_points, 0.0};
  PackagedSpeeds speeds{};
  for (auto* scalar : {&speeds.largest_out, &speeds.largest_in,
                       &speeds.fast_out, &speeds.fast_in, &speeds.flatness}) {
    *scalar = Scalar<DataVector>{num_points, 0.0};
  }

  grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8}
      .dg_package_data(
          make_not_null(&p_tilde_d), make_not_null(&p_tilde_ye),
          make_not_null(&p_tilde_tau), make_not_null(&p_tilde_s),
          make_not_null(&p_tilde_b), make_not_null(&p_tilde_phi),
          make_not_null(&p_nf_d), make_not_null(&p_nf_ye),
          make_not_null(&p_nf_tau), make_not_null(&p_nf_s),
          make_not_null(&p_nf_b), make_not_null(&p_nf_phi),
          make_not_null(&speeds.largest_out), make_not_null(&speeds.largest_in),
          make_not_null(&speeds.fast_out), make_not_null(&speeds.fast_in),
          make_not_null(&p_normal), make_not_null(&speeds.flatness),

          zero_scalar, zero_scalar, zero_scalar, zero_lower, magnetic_field,
          zero_scalar,

          zero_upper, zero_upper, zero_upper, zero_mixed, zero_upper2,
          zero_upper,

          lapse, shift, spatial_velocity_one_form, spatial_metric,

          rest_mass_density, electron_fraction, temperature, spatial_velocity,
          specific_internal_energy, pressure, lorentz_factor,

          normal_covector, normal_vector, std::nullopt, std::nullopt, eos);
  return speeds;
}

// Without a magnetic field the packaged fast speeds must equal the largest
// speeds, which the hydro branch computes.
void test_fast_speeds_reduce_to_hydro() {
  INFO("HLL: with B = 0 the packaged fast speeds are the hydro speeds");
  const size_t num_points = 4;
  const auto eos =
      EquationsOfState::PolytropicFluid<true>{100.0, 2.0}.promote_to_3d_eos();
  const Scalar<DataVector> rest_mass_density{DataVector{1.0, 0.5, 2.0, 0.8}};
  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{num_points, 0.0};
  get<0>(spatial_velocity) = DataVector{0.10, -0.25, 0.05, 0.30};
  get<1>(spatial_velocity) = DataVector{-0.20, 0.15, 0.00, 0.10};
  get<2>(spatial_velocity) = DataVector{0.05, 0.05, -0.10, 0.00};
  tnsr::i<DataVector, 3, Frame::Inertial> normal_covector{num_points, 0.0};
  get<0>(normal_covector) = DataVector{num_points, 0.6};
  get<1>(normal_covector) = DataVector{num_points, 0.8};
  const tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{num_points, 0.0};

  const PackagedSpeeds speeds =
      package_speeds(rest_mass_density, spatial_velocity, magnetic_field,
                     normal_covector, *eos);
  const Approx speed_approx = Approx::custom().epsilon(1.0e-13).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(get(speeds.fast_out), get(speeds.largest_out),
                               speed_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get(speeds.fast_in), get(speeds.largest_in),
                               speed_approx);
  CHECK(max(abs(get(speeds.fast_out))) < 1.0);
  CHECK(max(abs(get(speeds.fast_in))) < 1.0);
  CHECK(max(get(speeds.flatness)) == approx(0.0));
}

// The fast speeds against closed forms of the dispersion relation for the
// estimate a^2 = c_s^2 + v_A^2 (1 - c_s^2), v_A^2 = b^2 / (b^2 + rho h): with
// v = 0 they are +-a; with v along the normal the relativistic sum
// (v_n +- a) / (1 +- v_n a); with v across the normal
// +-a sqrt((1 - v^2) / (1 - v^2 a^2)).
void test_fast_speeds_closed_forms() {
  INFO("HLL: fast speeds against the closed forms");
  const size_t num_points = 3;
  const auto eos =
      EquationsOfState::PolytropicFluid<true>{100.0, 2.0}.promote_to_3d_eos();
  const Scalar<DataVector> rest_mass_density{DataVector{1.0, 0.5, 2.0}};
  tnsr::i<DataVector, 3, Frame::Inertial> normal_covector{num_points, 0.0};
  get<0>(normal_covector) = DataVector{num_points, 0.6};
  get<1>(normal_covector) = DataVector{num_points, 0.8};
  // v = 0, then v_n = 0.4 along the normal, then |v| = 0.5 across it
  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{num_points, 0.0};
  get<0>(spatial_velocity) = DataVector{0.0, 0.4 * 0.6, -0.5 * 0.8};
  get<1>(spatial_velocity) = DataVector{0.0, 0.4 * 0.8, 0.5 * 0.6};
  tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{num_points, 0.0};
  get<0>(magnetic_field) = DataVector{num_points, 0.3};
  get<1>(magnetic_field) = DataVector{num_points, -0.2};
  get<2>(magnetic_field) = DataVector{num_points, 0.5};

  const PackagedSpeeds speeds =
      package_speeds(rest_mass_density, spatial_velocity, magnetic_field,
                     normal_covector, *eos);

  const Scalar<DataVector> electron_fraction{DataVector{num_points, 0.1}};
  const Scalar<DataVector> temperature{DataVector{num_points, 0.0}};
  const DataVector specific_internal_energy =
      get(eos->specific_internal_energy_from_density_and_temperature(
          rest_mass_density, temperature, electron_fraction));
  const DataVector pressure = get(eos->pressure_from_density_and_energy(
      rest_mass_density, Scalar<DataVector>{specific_internal_energy},
      electron_fraction));
  const DataVector sound_speed_squared =
      get(eos->sound_speed_squared_from_density_and_temperature(
          rest_mass_density, temperature, electron_fraction));
  const DataVector rho_h =
      get(rest_mass_density) * (1.0 + specific_internal_energy) + pressure;
  DataVector v_squared{num_points, 0.0};
  DataVector b_dot_v{num_points, 0.0};
  DataVector b_squared{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    v_squared += square(spatial_velocity.get(i));
    b_dot_v += magnetic_field.get(i) * spatial_velocity.get(i);
    b_squared += square(magnetic_field.get(i));
  }
  const DataVector comoving_b_squared =
      b_squared * (1.0 - v_squared) + square(b_dot_v);
  const DataVector a_squared =
      sound_speed_squared + comoving_b_squared / (comoving_b_squared + rho_h) *
                                (1.0 - sound_speed_squared);
  const DataVector a = sqrt(a_squared);
  const double v_n = 0.4;
  const double v_t_squared = 0.25;
  const DataVector expected_out{
      a[0], (v_n + a[1]) / (1.0 + v_n * a[1]),
      a[2] * sqrt((1.0 - v_t_squared) / (1.0 - v_t_squared * a_squared[2]))};
  const DataVector expected_in{
      -a[0], (v_n - a[1]) / (1.0 - v_n * a[1]),
      -a[2] * sqrt((1.0 - v_t_squared) / (1.0 - v_t_squared * a_squared[2]))};
  const Approx speed_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(get(speeds.fast_out), expected_out,
                               speed_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get(speeds.fast_in), expected_in, speed_approx);
  CHECK(max(get(speeds.flatness)) == approx(0.0));
}

// The flux must change sign under the int<->ext swap when the fast speeds are
// inside the light speeds, also when the two sides' normals are not exactly
// opposite. One-sided fast bounds, or a split along one side's normal, fail
// this.
void test_int_ext_swap_symmetry() {
  INFO("HLL: two-sided fast bounds, antisymmetric under the int<->ext swap");
  MAKE_GENERATOR(gen);
  const DataVector used_for_size{5};
  std::uniform_real_distribution<> dist(0.5, 1.5);
  std::uniform_real_distribution<> speed_dist(0.2, 0.9);

  const auto scalar = [&gen, &dist, &used_for_size]() {
    return make_with_random_values<Scalar<DataVector>>(
        make_not_null(&gen), make_not_null(&dist), used_for_size);
  };
  const auto covector = [&gen, &dist, &used_for_size]() {
    return make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
        make_not_null(&gen), make_not_null(&dist), used_for_size);
  };
  const auto vector = [&gen, &dist, &used_for_size]() {
    return make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
        make_not_null(&gen), make_not_null(&dist), used_for_size);
  };
  const auto speed = [&gen, &speed_dist, &used_for_size](const double sign) {
    auto result = make_with_random_values<Scalar<DataVector>>(
        make_not_null(&gen), make_not_null(&speed_dist), used_for_size);
    get(result) *= sign;
    return result;
  };

  struct Packet {
    Scalar<DataVector> tilde_d{};
    Scalar<DataVector> tilde_ye{};
    Scalar<DataVector> tilde_tau{};
    tnsr::i<DataVector, 3, Frame::Inertial> tilde_s{};
    tnsr::I<DataVector, 3, Frame::Inertial> tilde_b{};
    Scalar<DataVector> tilde_phi{};
    Scalar<DataVector> nf_d{};
    Scalar<DataVector> nf_ye{};
    Scalar<DataVector> nf_tau{};
    tnsr::i<DataVector, 3, Frame::Inertial> nf_s{};
    tnsr::I<DataVector, 3, Frame::Inertial> nf_b{};
    Scalar<DataVector> nf_phi{};
    Scalar<DataVector> largest_out{};
    Scalar<DataVector> largest_in{};
    Scalar<DataVector> fast_out{};
    Scalar<DataVector> fast_in{};
    tnsr::i<DataVector, 3, Frame::Inertial> normal{};
    Scalar<DataVector> flatness{};
  };

  const auto make_packet = [&scalar, &covector, &vector, &speed,
                            &used_for_size](const double normal_sign) {
    Packet p{};
    p.tilde_d = scalar();
    p.tilde_ye = scalar();
    p.tilde_tau = scalar();
    p.tilde_s = covector();
    p.tilde_b = vector();
    p.tilde_phi = scalar();
    p.nf_d = scalar();
    p.nf_ye = scalar();
    p.nf_tau = scalar();
    p.nf_s = covector();
    p.nf_b = vector();
    p.nf_phi = scalar();
    p.largest_out = Scalar<DataVector>{DataVector{used_for_size.size(), 1.0}};
    p.largest_in = Scalar<DataVector>{DataVector{used_for_size.size(), -1.0}};
    p.fast_out = speed(1.0);
    p.fast_in = speed(-1.0);
    p.normal =
        tnsr::i<DataVector, 3, Frame::Inertial>{used_for_size.size(), 0.0};
    get<0>(p.normal) = normal_sign * 0.6;
    get<1>(p.normal) = normal_sign * 0.8;
    p.flatness = Scalar<DataVector>{DataVector{used_for_size.size(), 0.0}};
    return p;
  };

  const Packet interior = make_packet(1.0);
  const Packet exterior = make_packet(-1.0);

  const auto boundary_terms = [](const Packet& in, const Packet& ex) {
    std::array<Scalar<DataVector>, 4> scalars{};
    tnsr::i<DataVector, 3, Frame::Inertial> s{get(in.tilde_d).size(), 0.0};
    tnsr::I<DataVector, 3, Frame::Inertial> b{get(in.tilde_d).size(), 0.0};
    for (auto& entry : scalars) {
      entry = Scalar<DataVector>{DataVector{get(in.tilde_d).size(), 0.0}};
    }
    grmhd::ValenciaDivClean::BoundaryCorrections::Hll::dg_boundary_terms(
        make_not_null(scalars.data()), make_not_null(&scalars[1]),
        make_not_null(&scalars[2]), make_not_null(&s), make_not_null(&b),
        make_not_null(&scalars[3]),

        in.tilde_d, in.tilde_ye, in.tilde_tau, in.tilde_s, in.tilde_b,
        in.tilde_phi, in.nf_d, in.nf_ye, in.nf_tau, in.nf_s, in.nf_b, in.nf_phi,
        in.largest_out, in.largest_in, in.fast_out, in.fast_in, in.normal,
        in.flatness,

        ex.tilde_d, ex.tilde_ye, ex.tilde_tau, ex.tilde_s, ex.tilde_b,
        ex.tilde_phi, ex.nf_d, ex.nf_ye, ex.nf_tau, ex.nf_s, ex.nf_b, ex.nf_phi,
        ex.largest_out, ex.largest_in, ex.fast_out, ex.fast_in, ex.normal,
        ex.flatness,

        dg::Formulation::WeakInertial);
    return std::make_tuple(scalars, s, b);
  };

  const auto check_antisymmetric = [&boundary_terms](const Packet& in,
                                                     const Packet& ex) {
    const auto forward = boundary_terms(in, ex);
    const auto swapped = boundary_terms(ex, in);
    const Approx swap_approx = Approx::custom().epsilon(1.0e-14).scale(1.0);
    for (size_t i = 0; i < 4; ++i) {
      CAPTURE(i);
      const DataVector negated = -get(gsl::at(std::get<0>(swapped), i));
      CHECK_ITERABLE_CUSTOM_APPROX(get(gsl::at(std::get<0>(forward), i)),
                                   negated, swap_approx);
    }
    for (size_t i = 0; i < 3; ++i) {
      CAPTURE(i);
      const DataVector negated_s = -std::get<1>(swapped).get(i);
      CHECK_ITERABLE_CUSTOM_APPROX(std::get<1>(forward).get(i), negated_s,
                                   swap_approx);
      const DataVector negated_b = -std::get<2>(swapped).get(i);
      CHECK_ITERABLE_CUSTOM_APPROX(std::get<2>(forward).get(i), negated_b,
                                   swap_approx);
    }
  };
  check_antisymmetric(interior, exterior);

  // On a non-conforming face the two sides' normals are interpolated
  // separately and are not exactly opposite; the split along their average
  // keeps the antisymmetry.
  Packet exterior_tilted = exterior;
  get<0>(exterior_tilted.normal) = -(0.6 + 1.0e-3);
  get<1>(exterior_tilted.normal) = -(0.8 - 2.0e-3);
  get<2>(exterior_tilted.normal) = -1.5e-3;
  const DataVector tilted_magnitude =
      sqrt(square(get<0>(exterior_tilted.normal)) +
           square(get<1>(exterior_tilted.normal)) +
           square(get<2>(exterior_tilted.normal)));
  for (size_t i = 0; i < 3; ++i) {
    exterior_tilted.normal.get(i) /= tilted_magnitude;
  }
  check_antisymmetric(interior, exterior_tilted);
}

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.Hll",
                  "[Unit][GrMhd]") {
  PUPable_reg(grmhd::ValenciaDivClean::BoundaryCorrections::Hll);
  test_fast_speeds_reduce_to_hydro();
  test_fast_speeds_closed_forms();
  test_int_ext_swap_symmetry();
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  using system = grmhd::ValenciaDivClean::System;

  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>>
      ranges{std::array{0.3, 1.0}, std::array{0.01, 0.02}};
  const tuples::TaggedTuple<hydro::Tags::GrmhdEquationOfState> volume_data{
      EquationsOfState::PolytropicFluid<true>{100.0, 2.0}.promote_to_3d_eos()};

  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen),
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Hll", "dg_package_data", "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  // Test the hydro speeds: the magnetic field is below
  // MagneticFieldMagnitudeForHydro, the density above the light speed cutoff
  // and the velocities are subluminal
  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>,
      helpers::Tags::Range<grmhd::ValenciaDivClean::Tags::TildeB<>>,
      helpers::Tags::Range<hydro::Tags::RestMassDensity<DataVector>>,
      helpers::Tags::Range<hydro::Tags::SpatialVelocity<DataVector, 3>>,
      helpers::Tags::Range<
          hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>>>
      ranges_hydro{std::array{0.3, 1.0},         std::array{0.01, 0.02},
                   std::array{1.0e-35, 1.0e-32}, std::array{0.5, 1.5},
                   std::array{-0.5, 0.5},        std::array{-0.5, 0.5}};
  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Hll", "dg_package_data", "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges_hydro);

  // Test light speed density cutoff
  const tuples::TaggedTuple<
      helpers::Tags::Range<hydro::Tags::RestMassDensity<DataVector>>,
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>>
      ranges_atmo{std::array{1.0e-10, 1.0e-9}, std::array{0.3, 1.0},
                  std::array{0.01, 0.02}};
  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Hll", "dg_package_data", "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges_atmo);

  // Test the fast speeds and the split of the magnetic field on a flat
  // background. The helper adds one to the random spatial metric, scales it
  // by 0.005 and adds the identity, so the range -1 gives the flat metric.
  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>,
      helpers::Tags::Range<gr::Tags::SpatialMetric<DataVector, 3>>,
      helpers::Tags::Range<hydro::Tags::RestMassDensity<DataVector>>,
      helpers::Tags::Range<hydro::Tags::SpatialVelocity<DataVector, 3>>,
      helpers::Tags::Range<hydro::Tags::LorentzFactor<DataVector>>,
      helpers::Tags::Range<hydro::Tags::SpecificInternalEnergy<DataVector>>,
      helpers::Tags::Range<hydro::Tags::Pressure<DataVector>>>
      ranges_flat{std::array{1.0, 1.0},   std::array{0.0, 0.0},
                  std::array{-1.0, -1.0}, std::array{0.5, 1.5},
                  std::array{-0.5, 0.5},  std::array{1.0, 2.0},
                  std::array{0.0, 1.0},   std::array{0.0, 1.0}};
  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen),
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges_flat);
  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Hll", "dg_package_data", "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges_flat);

  // Test the light speed density cutoff on a flat background, with points on
  // both sides of it
  auto ranges_flat_atmo = ranges_flat;
  tuples::get<helpers::Tags::Range<hydro::Tags::RestMassDensity<DataVector>>>(
      ranges_flat_atmo) = std::array{1.0e-9, 1.0e-7};
  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Hll", "dg_package_data", "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges_flat_atmo);

  const auto hll = TestHelpers::test_factory_creation<
      evolution::BoundaryCorrection,
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll>(
      "Hll:\n"
      "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
      "  LightSpeedDensityCutoff: 1.0e-8\n");

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Hll", "dg_package_data", "dg_boundary_terms",
      dynamic_cast<const grmhd::ValenciaDivClean::BoundaryCorrections::Hll&>(
          *hll),
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  CHECK_FALSE(
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8} !=
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8});
  CHECK(grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8} !=
        grmhd::ValenciaDivClean::BoundaryCorrections::Hll{2.0e-30, 1.0e-8});
  CHECK(grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8} !=
        grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 2.0e-8});
}
}  // namespace
