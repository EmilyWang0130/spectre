// Distributed under the MIT License.
// See LICENSE.txt for details.

// Microbenchmark of the per-face flux evaluation for
// `grmhd::ValenciaDivClean::BoundaryCorrections::Marquina` (full 6-mode
// characteristic decomposition) against
// `grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCpm` (Complementary
// Projection Method). Google Benchmark harness — see the sibling
// `Benchmark.cpp` for the canonical SpECTRE pattern.
//
// Setup: one random physically valid face state (`Mesh<2>{5}` = 25 face
// points, IdealFluid EoS with Gamma=1.5, flat spacetime perturbation,
// zero magnetic field). Each timing iteration performs two
// `dg_package_data` calls (interior + exterior) plus one
// `dg_boundary_terms` call.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <benchmark/benchmark.h>
#pragma GCC diagnostic pop
#include <charm++.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Marquina.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/MarquinaCpm.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
using grmhd::ValenciaDivClean::BoundaryCorrections::Marquina;
using grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCpm;
namespace grmhd_tags = grmhd::ValenciaDivClean::Tags;

using PackageTags = Marquina::dg_package_field_tags;

/// A single random face state (interior or exterior side). Owns all input
/// primitives, conservatives, fluxes, geometry, and the packaged-data
/// output buffer used by the flux scheme.
struct SideState {
  // Face-scoped scalar buffers.
  Scalar<DataVector> rest_mass_density{};
  Scalar<DataVector> electron_fraction{};
  Scalar<DataVector> temperature{};
  Scalar<DataVector> specific_internal_energy{};
  Scalar<DataVector> pressure{};
  Scalar<DataVector> lorentz_factor{};
  Scalar<DataVector> divergence_cleaning_field{};
  Scalar<DataVector> sqrt_det_spatial_metric{};
  Scalar<DataVector> lapse{};

  // 3-vectors.
  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{};
  tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{};
  tnsr::i<DataVector, 3, Frame::Inertial> spatial_velocity_one_form{};
  tnsr::I<DataVector, 3, Frame::Inertial> shift{};

  // Geometry.
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{};
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric{};

  // Normal on this side (unit under the spatial metric).
  tnsr::i<DataVector, 3, Frame::Inertial> unit_normal_covector{};
  tnsr::I<DataVector, 3, Frame::Inertial> unit_normal_vector{};

  // Conservatives.
  Scalar<DataVector> tilde_d{};
  Scalar<DataVector> tilde_ye{};
  Scalar<DataVector> tilde_tau{};
  Scalar<DataVector> tilde_phi{};
  tnsr::i<DataVector, 3, Frame::Inertial> tilde_s{};
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_b{};

  // Volume fluxes.
  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_d{};
  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_ye{};
  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_tau{};
  tnsr::Ij<DataVector, 3, Frame::Inertial> flux_tilde_s{};
  tnsr::IJ<DataVector, 3, Frame::Inertial> flux_tilde_b{};
  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_phi{};

  // Output of dg_package_data.
  Variables<PackageTags> package_data{};
};

// Fill `s` with a physically valid random face state at `n_points` grid
// points, using `gen` for randomness. `sign` flips the normal (use +1 for
// interior, -1 for exterior — flat-metric convention).
void initialize_side(SideState* s, size_t n_points, std::mt19937* gen,
                     const EquationsOfState::EquationOfState<true, 3>& eos,
                     double sign) {
  std::uniform_real_distribution<double> dist_rho(0.1, 1.0);
  std::uniform_real_distribution<double> dist_eps(0.1, 1.0);
  std::uniform_real_distribution<double> dist_ye(0.05, 0.5);
  std::uniform_real_distribution<double> dist_v(0.0, 0.4);
  std::uniform_real_distribution<double> dist_lapse(0.5, 1.0);
  std::uniform_real_distribution<double> dist_shift(-0.1, 0.1);
  std::uniform_real_distribution<double> dist_normal(-1.0, 1.0);
  std::uniform_real_distribution<double> dist_metric(-0.01, 0.01);

  DataVector used_for_size{n_points};
  const auto init_scalar = [&used_for_size](Scalar<DataVector>* out,
                                            double val) {
    get(*out) = DataVector(used_for_size.size(), val);
  };
  const auto fill_scalar = [gen](Scalar<DataVector>* out, auto& d) {
    for (size_t i = 0; i < get(*out).size(); ++i) {
      get(*out)[i] = d(*gen);
    }
  };

  // Primitives.
  get(s->rest_mass_density) = DataVector(n_points);
  fill_scalar(&s->rest_mass_density, dist_rho);
  get(s->electron_fraction) = DataVector(n_points);
  fill_scalar(&s->electron_fraction, dist_ye);
  get(s->specific_internal_energy) = DataVector(n_points);
  fill_scalar(&s->specific_internal_energy, dist_eps);
  init_scalar(&s->divergence_cleaning_field, 0.0);
  init_scalar(&s->lapse, 0.0);
  fill_scalar(&s->lapse, dist_lapse);

  // Random spatial velocity components with magnitude bounded so
  // 1 - v^2 stays comfortably positive.
  for (size_t i = 0; i < 3; ++i) {
    s->spatial_velocity.get(i) = DataVector(n_points);
    for (size_t point = 0; point < n_points; ++point) {
      s->spatial_velocity.get(i)[point] = dist_v(*gen) * (i == 0 ? 1.0 : 0.3);
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    s->shift.get(i) = DataVector(n_points);
    for (size_t point = 0; point < n_points; ++point) {
      s->shift.get(i)[point] = dist_shift(*gen);
    }
  }

  // Zero magnetic field (Marquina's B-block is a shared central-flux
  // fallback in both schemes, so it's not part of the CPM vs full
  // comparison).
  for (size_t i = 0; i < 3; ++i) {
    s->magnetic_field.get(i) = DataVector(n_points, 0.0);
  }

  // Spatial metric = identity + small symmetric perturbation.
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      s->spatial_metric.get(i, j) = DataVector(n_points);
      for (size_t point = 0; point < n_points; ++point) {
        s->spatial_metric.get(i, j)[point] =
            (i == j ? 1.0 : 0.0) + dist_metric(*gen);
      }
    }
  }

  // Determinant + inverse metric + sqrt(det g).
  const auto det_and_inv = determinant_and_inverse(s->spatial_metric);
  s->inv_spatial_metric = det_and_inv.second;
  get(s->sqrt_det_spatial_metric) = sqrt(get(det_and_inv.first));

  // v_i = gamma_ij v^j
  for (size_t i = 0; i < 3; ++i) {
    s->spatial_velocity_one_form.get(i) = DataVector(n_points, 0.0);
    for (size_t j = 0; j < 3; ++j) {
      s->spatial_velocity_one_form.get(i) +=
          s->spatial_metric.get(i, j) * s->spatial_velocity.get(j);
    }
  }

  // Lorentz factor W = 1 / sqrt(1 - v_i v^i).
  DataVector v_squared(n_points, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    v_squared +=
        s->spatial_velocity.get(i) * s->spatial_velocity_one_form.get(i);
  }
  get(s->lorentz_factor) = 1.0 / sqrt(1.0 - v_squared);

  // Pressure and temperature from the IdealFluid EoS (3D interface takes
  // rho, epsilon, and Y_e; the ideal fluid ignores Y_e).
  get(s->pressure) = get(eos.pressure_from_density_and_energy(
      s->rest_mass_density, s->specific_internal_energy, s->electron_fraction));
  get(s->temperature) = get(eos.temperature_from_density_and_energy(
      s->rest_mass_density, s->specific_internal_energy, s->electron_fraction));

  // Random normal covector, then normalize under spatial metric.
  for (size_t i = 0; i < 3; ++i) {
    s->unit_normal_covector.get(i) = DataVector(n_points);
    for (size_t point = 0; point < n_points; ++point) {
      s->unit_normal_covector.get(i)[point] = dist_normal(*gen);
    }
  }
  s->unit_normal_vector = tnsr::I<DataVector, 3, Frame::Inertial>{n_points};
  for (size_t i = 0; i < 3; ++i) {
    s->unit_normal_vector.get(i) = 0.0;
    for (size_t j = 0; j < 3; ++j) {
      s->unit_normal_vector.get(i) +=
          s->inv_spatial_metric.get(i, j) * s->unit_normal_covector.get(j);
    }
  }
  DataVector normal_mag_squared(n_points, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    normal_mag_squared +=
        s->unit_normal_covector.get(i) * s->unit_normal_vector.get(i);
  }
  const DataVector normal_mag = sqrt(normal_mag_squared);
  for (size_t i = 0; i < 3; ++i) {
    s->unit_normal_covector.get(i) /= normal_mag;
    s->unit_normal_vector.get(i) /= normal_mag;
    // Flip sign for exterior side.
    s->unit_normal_covector.get(i) *= sign;
    s->unit_normal_vector.get(i) *= sign;
  }

  // Conservatives from primitives.
  get(s->tilde_d) = DataVector(n_points);
  get(s->tilde_ye) = DataVector(n_points);
  get(s->tilde_tau) = DataVector(n_points);
  get(s->tilde_phi) = DataVector(n_points);
  for (size_t i = 0; i < 3; ++i) {
    s->tilde_s.get(i) = DataVector(n_points);
    s->tilde_b.get(i) = DataVector(n_points);
  }
  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&s->tilde_d), make_not_null(&s->tilde_ye),
      make_not_null(&s->tilde_tau), make_not_null(&s->tilde_s),
      make_not_null(&s->tilde_b), make_not_null(&s->tilde_phi),
      s->rest_mass_density, s->electron_fraction, s->specific_internal_energy,
      s->pressure, s->spatial_velocity, s->lorentz_factor, s->magnetic_field,
      s->sqrt_det_spatial_metric, s->spatial_metric,
      s->divergence_cleaning_field);

  // Fluxes from primitives + conservatives.
  for (size_t i = 0; i < 3; ++i) {
    s->flux_tilde_d.get(i) = DataVector(n_points);
    s->flux_tilde_ye.get(i) = DataVector(n_points);
    s->flux_tilde_tau.get(i) = DataVector(n_points);
    s->flux_tilde_phi.get(i) = DataVector(n_points);
    for (size_t j = 0; j < 3; ++j) {
      s->flux_tilde_s.get(i, j) = DataVector(n_points);
      s->flux_tilde_b.get(i, j) = DataVector(n_points);
    }
  }
  grmhd::ValenciaDivClean::ComputeFluxes::apply(
      make_not_null(&s->flux_tilde_d), make_not_null(&s->flux_tilde_ye),
      make_not_null(&s->flux_tilde_tau), make_not_null(&s->flux_tilde_s),
      make_not_null(&s->flux_tilde_b), make_not_null(&s->flux_tilde_phi),
      s->tilde_d, s->tilde_ye, s->tilde_tau, s->tilde_s, s->tilde_b,
      s->tilde_phi, s->lapse, s->shift, s->sqrt_det_spatial_metric,
      s->spatial_metric, s->inv_spatial_metric, s->pressure,
      s->spatial_velocity, s->lorentz_factor, s->magnetic_field);

  s->package_data.initialize(n_points);
}

// Package the given side's data using `Scheme::dg_package_data`.
template <typename Scheme>
void run_package_data(SideState* s,
                      const EquationsOfState::EquationOfState<true, 3>& eos) {
  Scheme::dg_package_data(
      make_not_null(&get<grmhd_tags::TildeD>(s->package_data)),
      make_not_null(&get<grmhd_tags::TildeYe>(s->package_data)),
      make_not_null(&get<grmhd_tags::TildeTau>(s->package_data)),
      make_not_null(&get<grmhd_tags::TildeS<Frame::Inertial>>(s->package_data)),
      make_not_null(&get<grmhd_tags::TildeB<Frame::Inertial>>(s->package_data)),
      make_not_null(&get<grmhd_tags::TildePhi>(s->package_data)),
      make_not_null(
          &get<::Tags::NormalDotFlux<grmhd_tags::TildeD>>(s->package_data)),
      make_not_null(
          &get<::Tags::NormalDotFlux<grmhd_tags::TildeYe>>(s->package_data)),
      make_not_null(
          &get<::Tags::NormalDotFlux<grmhd_tags::TildeTau>>(s->package_data)),
      make_not_null(
          &get<::Tags::NormalDotFlux<grmhd_tags::TildeS<Frame::Inertial>>>(
              s->package_data)),
      make_not_null(
          &get<::Tags::NormalDotFlux<grmhd_tags::TildeB<Frame::Inertial>>>(
              s->package_data)),
      make_not_null(
          &get<::Tags::NormalDotFlux<grmhd_tags::TildePhi>>(s->package_data)),
      make_not_null(
          &get<typename Marquina::CharacteristicSpeeds>(s->package_data)),
      make_not_null(
          &get<typename Marquina::LeftCharacteristicFields>(s->package_data)),
      make_not_null(
          &get<typename Marquina::RightCharacteristicFields>(s->package_data)),
      s->tilde_d, s->tilde_ye, s->tilde_tau, s->tilde_s, s->tilde_b,
      s->tilde_phi, s->flux_tilde_d, s->flux_tilde_ye, s->flux_tilde_tau,
      s->flux_tilde_s, s->flux_tilde_b, s->flux_tilde_phi, s->lapse, s->shift,
      s->spatial_velocity_one_form, s->spatial_metric, s->rest_mass_density,
      s->electron_fraction, s->temperature, s->spatial_velocity,
      s->specific_internal_energy, s->pressure, s->lorentz_factor,
      s->unit_normal_covector, s->unit_normal_vector, std::nullopt,
      std::nullopt, eos);
}

// One iteration of the flux evaluation: package interior + exterior + run
// boundary terms. Bench state variables are captured by reference.
template <typename Scheme>
void one_iteration(SideState* interior, SideState* exterior,
                   Scalar<DataVector>* bc_tilde_d,
                   Scalar<DataVector>* bc_tilde_ye,
                   Scalar<DataVector>* bc_tilde_tau,
                   tnsr::i<DataVector, 3, Frame::Inertial>* bc_tilde_s,
                   tnsr::I<DataVector, 3, Frame::Inertial>* bc_tilde_b,
                   Scalar<DataVector>* bc_tilde_phi,
                   const EquationsOfState::EquationOfState<true, 3>& eos) {
  run_package_data<Scheme>(interior, eos);
  run_package_data<Scheme>(exterior, eos);
  Scheme::dg_boundary_terms(
      make_not_null(bc_tilde_d), make_not_null(bc_tilde_ye),
      make_not_null(bc_tilde_tau), make_not_null(bc_tilde_s),
      make_not_null(bc_tilde_b), make_not_null(bc_tilde_phi),
      get<grmhd_tags::TildeD>(interior->package_data),
      get<grmhd_tags::TildeYe>(interior->package_data),
      get<grmhd_tags::TildeTau>(interior->package_data),
      get<grmhd_tags::TildeS<Frame::Inertial>>(interior->package_data),
      get<grmhd_tags::TildeB<Frame::Inertial>>(interior->package_data),
      get<grmhd_tags::TildePhi>(interior->package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeD>>(interior->package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeYe>>(interior->package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeTau>>(interior->package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeS<Frame::Inertial>>>(
          interior->package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeB<Frame::Inertial>>>(
          interior->package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildePhi>>(interior->package_data),
      get<typename Marquina::CharacteristicSpeeds>(interior->package_data),
      get<typename Marquina::LeftCharacteristicFields>(interior->package_data),
      get<typename Marquina::RightCharacteristicFields>(interior->package_data),
      get<grmhd_tags::TildeD>(exterior->package_data),
      get<grmhd_tags::TildeYe>(exterior->package_data),
      get<grmhd_tags::TildeTau>(exterior->package_data),
      get<grmhd_tags::TildeS<Frame::Inertial>>(exterior->package_data),
      get<grmhd_tags::TildeB<Frame::Inertial>>(exterior->package_data),
      get<grmhd_tags::TildePhi>(exterior->package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeD>>(exterior->package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeYe>>(exterior->package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeTau>>(exterior->package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeS<Frame::Inertial>>>(
          exterior->package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeB<Frame::Inertial>>>(
          exterior->package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildePhi>>(exterior->package_data),
      get<typename Marquina::CharacteristicSpeeds>(exterior->package_data),
      get<typename Marquina::LeftCharacteristicFields>(exterior->package_data),
      get<typename Marquina::RightCharacteristicFields>(exterior->package_data),
      ::dg::Formulation::WeakInertial);
}

template <typename Scheme>
void bench_scheme(benchmark::State& state) {
  const size_t face_extent = 5;
  const Mesh<2> face_mesh{face_extent, Spectral::Basis::Legendre,
                          Spectral::Quadrature::Gauss};
  const size_t n_points = face_mesh.number_of_grid_points();  // 25

  // Fixed seed → same random face state across all benchmark runs, so
  // Marquina and MarquinaCpm are timed on identical inputs.
  std::mt19937 gen(12345);

  const EquationsOfState::IdealFluid<true> base_eos{1.5, 0.0};
  const std::unique_ptr<EquationsOfState::EquationOfState<true, 3>> eos_3d =
      base_eos.promote_to_3d_eos();

  SideState interior;
  SideState exterior;
  initialize_side(&interior, n_points, &gen, *eos_3d, +1.0);
  initialize_side(&exterior, n_points, &gen, *eos_3d, -1.0);

  Scalar<DataVector> bc_tilde_d{DataVector(n_points, 0.0)};
  Scalar<DataVector> bc_tilde_ye{DataVector(n_points, 0.0)};
  Scalar<DataVector> bc_tilde_tau{DataVector(n_points, 0.0)};
  Scalar<DataVector> bc_tilde_phi{DataVector(n_points, 0.0)};
  tnsr::i<DataVector, 3, Frame::Inertial> bc_tilde_s{};
  tnsr::I<DataVector, 3, Frame::Inertial> bc_tilde_b{};
  for (size_t i = 0; i < 3; ++i) {
    bc_tilde_s.get(i) = DataVector(n_points, 0.0);
    bc_tilde_b.get(i) = DataVector(n_points, 0.0);
  }

  for (auto _ : state) {
    one_iteration<Scheme>(&interior, &exterior, &bc_tilde_d, &bc_tilde_ye,
                          &bc_tilde_tau, &bc_tilde_s, &bc_tilde_b,
                          &bc_tilde_phi, *eos_3d);
    benchmark::DoNotOptimize(bc_tilde_d);
    benchmark::DoNotOptimize(bc_tilde_ye);
    benchmark::DoNotOptimize(bc_tilde_tau);
    benchmark::DoNotOptimize(bc_tilde_s);
    benchmark::DoNotOptimize(bc_tilde_b);
    benchmark::DoNotOptimize(bc_tilde_phi);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(n_points));
}

BENCHMARK(bench_scheme<Marquina>)->Name("Marquina");
BENCHMARK(bench_scheme<MarquinaCpm>)->Name("MarquinaCpm");
}  // namespace

BENCHMARK_MAIN();
