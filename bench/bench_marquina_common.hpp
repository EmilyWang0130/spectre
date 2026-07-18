// Distributed under the MIT License.
// See LICENSE.txt for details.
//
// Shared harness for the SpECTRE Marquina flux-kernel throughput benchmarks
// (bench_marquina.cpp times the full-decomposition `Marquina`;
// bench_marquina_cpm.cpp times the Complementary Projection Method
// `MarquinaCpm`). Both drive the real kernel entry points
// (`Scheme::dg_package_data` ×2 + `Scheme::dg_boundary_terms`) over N random,
// physically valid hydro-only interface state pairs.
//
// Configuration is the hydro-only limit for an apples-to-apples comparison
// against WHISKY-Marquina (see bench/bench_whisky.F90): B = 0, Y_e = const,
// ideal-gas EOS with Gamma = 2, flat spatial metric, lapse = 1, shift = 0.
// Interfaces are processed in fixed-size chunks so the DataVector (SoA) layout
// vectorizes across grid points while peak memory stays bounded and buffers
// are reused chunk-to-chunk. Only the kernel calls are timed
// (std::chrono::steady_clock); state generation and conservative/flux setup
// are excluded.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Marquina.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace bench_marquina {
namespace grmhd_tags = grmhd::ValenciaDivClean::Tags;
// Both schemes share Marquina's package-data tag types.
using Marquina = grmhd::ValenciaDivClean::BoundaryCorrections::Marquina;
using PackageTags = Marquina::dg_package_field_tags;

constexpr size_t kNumInterfaces = 1000000;
constexpr size_t kChunkSize = 10000;
constexpr double kGamma = 2.0;
constexpr double kYe = 0.1;

// Rough per-interface floating-point-operation estimate. Used only for the
// reported GFlop/s figure, which is an order-of-magnitude sanity check.
constexpr double kApproxFlopsPerInterface = 1600.0;

// All inputs and outputs for one side of a chunk of interfaces.
struct SideState {
  Scalar<DataVector> rest_mass_density{};
  Scalar<DataVector> electron_fraction{};
  Scalar<DataVector> temperature{};
  Scalar<DataVector> specific_internal_energy{};
  Scalar<DataVector> pressure{};
  Scalar<DataVector> lorentz_factor{};
  Scalar<DataVector> divergence_cleaning_field{};
  Scalar<DataVector> sqrt_det_spatial_metric{};
  Scalar<DataVector> lapse{};

  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{};
  tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{};
  tnsr::i<DataVector, 3, Frame::Inertial> spatial_velocity_one_form{};
  tnsr::I<DataVector, 3, Frame::Inertial> shift{};

  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{};
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric{};

  tnsr::i<DataVector, 3, Frame::Inertial> unit_normal_covector{};
  tnsr::I<DataVector, 3, Frame::Inertial> unit_normal_vector{};

  Scalar<DataVector> tilde_d{};
  Scalar<DataVector> tilde_ye{};
  Scalar<DataVector> tilde_tau{};
  Scalar<DataVector> tilde_phi{};
  tnsr::i<DataVector, 3, Frame::Inertial> tilde_s{};
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_b{};

  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_d{};
  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_ye{};
  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_tau{};
  tnsr::Ij<DataVector, 3, Frame::Inertial> flux_tilde_s{};
  tnsr::IJ<DataVector, 3, Frame::Inertial> flux_tilde_b{};
  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_phi{};

  Variables<PackageTags> package_data{};

  void allocate(size_t n) {
    const DataVector zero(n, 0.0);
    get(rest_mass_density) = DataVector(n);
    get(electron_fraction) = DataVector(n, kYe);
    get(temperature) = DataVector(n);
    get(specific_internal_energy) = DataVector(n);
    get(pressure) = DataVector(n);
    get(lorentz_factor) = DataVector(n);
    get(divergence_cleaning_field) = zero;
    get(sqrt_det_spatial_metric) = DataVector(n, 1.0);
    get(lapse) = DataVector(n, 1.0);
    for (size_t i = 0; i < 3; ++i) {
      spatial_velocity.get(i) = DataVector(n);
      magnetic_field.get(i) = zero;
      spatial_velocity_one_form.get(i) = DataVector(n);
      shift.get(i) = zero;
      unit_normal_covector.get(i) = zero;
      unit_normal_vector.get(i) = zero;
      tilde_s.get(i) = DataVector(n);
      tilde_b.get(i) = zero;
      flux_tilde_d.get(i) = DataVector(n);
      flux_tilde_ye.get(i) = DataVector(n);
      flux_tilde_tau.get(i) = DataVector(n);
      flux_tilde_phi.get(i) = DataVector(n);
      for (size_t j = 0; j < 3; ++j) {
        spatial_metric.get(i, j) = DataVector(n, i == j ? 1.0 : 0.0);
        inv_spatial_metric.get(i, j) = DataVector(n, i == j ? 1.0 : 0.0);
        flux_tilde_s.get(i, j) = DataVector(n);
        flux_tilde_b.get(i, j) = zero;
      }
    }
    get(tilde_d) = DataVector(n);
    get(tilde_ye) = DataVector(n);
    get(tilde_tau) = DataVector(n);
    get(tilde_phi) = zero;
    package_data.initialize(n);
  }
};

// Flat-metric raw primitive arrays for the whole run.
struct PrimitiveStore {
  std::vector<double> rho;
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;
  std::vector<double> eps;
};

inline void generate(PrimitiveStore* s, std::mt19937* gen, size_t n) {
  std::uniform_real_distribution<double> dist_rho(0.1, 10.0);
  std::uniform_real_distribution<double> dist_v(-0.3, 0.3);
  std::uniform_real_distribution<double> dist_eps(0.01, 1.0);
  s->rho.resize(n);
  s->vx.resize(n);
  s->vy.resize(n);
  s->vz.resize(n);
  s->eps.resize(n);
  for (size_t i = 0; i < n; ++i) {
    s->rho[i] = dist_rho(*gen);
    s->vx[i] = dist_v(*gen);
    s->vy[i] = dist_v(*gen);
    s->vz[i] = dist_v(*gen);
    s->eps[i] = dist_eps(*gen);
  }
}

// Fill one side's DataVectors for the chunk [offset, offset+n) from the raw
// primitives, then derive Lorentz factor, EOS pressure/temperature,
// conservatives and fluxes. Flat metric, so v_i = v^i, W = 1/sqrt(1 - v^2),
// unit normal = x-hat (matches WHISKY's normal-along-x convention).
inline void fill_side(SideState* st, const PrimitiveStore& prim, size_t offset,
                      size_t n, double normal_sign,
                      const EquationsOfState::EquationOfState<true, 3>& eos) {
  for (size_t i = 0; i < n; ++i) {
    const size_t k = offset + i;
    get(st->rest_mass_density)[i] = prim.rho[k];
    get(st->specific_internal_energy)[i] = prim.eps[k];
    st->spatial_velocity.get(0)[i] = prim.vx[k];
    st->spatial_velocity.get(1)[i] = prim.vy[k];
    st->spatial_velocity.get(2)[i] = prim.vz[k];
  }
  for (size_t i = 0; i < 3; ++i) {
    st->spatial_velocity_one_form.get(i) = st->spatial_velocity.get(i);
  }
  DataVector v_squared(n, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    v_squared += st->spatial_velocity.get(i) * st->spatial_velocity.get(i);
  }
  get(st->lorentz_factor) = 1.0 / sqrt(1.0 - v_squared);

  get(st->pressure) = get(eos.pressure_from_density_and_energy(
      st->rest_mass_density, st->specific_internal_energy,
      st->electron_fraction));
  get(st->temperature) = get(eos.temperature_from_density_and_energy(
      st->rest_mass_density, st->specific_internal_energy,
      st->electron_fraction));

  st->unit_normal_covector.get(0) = DataVector(n, normal_sign);
  st->unit_normal_covector.get(1) = DataVector(n, 0.0);
  st->unit_normal_covector.get(2) = DataVector(n, 0.0);
  st->unit_normal_vector.get(0) = DataVector(n, normal_sign);
  st->unit_normal_vector.get(1) = DataVector(n, 0.0);
  st->unit_normal_vector.get(2) = DataVector(n, 0.0);

  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&st->tilde_d), make_not_null(&st->tilde_ye),
      make_not_null(&st->tilde_tau), make_not_null(&st->tilde_s),
      make_not_null(&st->tilde_b), make_not_null(&st->tilde_phi),
      st->rest_mass_density, st->electron_fraction,
      st->specific_internal_energy, st->pressure, st->spatial_velocity,
      st->lorentz_factor, st->magnetic_field, st->sqrt_det_spatial_metric,
      st->spatial_metric, st->divergence_cleaning_field);

  grmhd::ValenciaDivClean::ComputeFluxes::apply(
      make_not_null(&st->flux_tilde_d), make_not_null(&st->flux_tilde_ye),
      make_not_null(&st->flux_tilde_tau), make_not_null(&st->flux_tilde_s),
      make_not_null(&st->flux_tilde_b), make_not_null(&st->flux_tilde_phi),
      st->tilde_d, st->tilde_ye, st->tilde_tau, st->tilde_s, st->tilde_b,
      st->tilde_phi, st->lapse, st->shift, st->sqrt_det_spatial_metric,
      st->spatial_metric, st->inv_spatial_metric, st->pressure,
      st->spatial_velocity, st->lorentz_factor, st->magnetic_field);
}

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

struct BoundaryCorrection {
  Scalar<DataVector> tilde_d{};
  Scalar<DataVector> tilde_ye{};
  Scalar<DataVector> tilde_tau{};
  Scalar<DataVector> tilde_phi{};
  tnsr::i<DataVector, 3, Frame::Inertial> tilde_s{};
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_b{};

  void allocate(size_t n) {
    get(tilde_d) = DataVector(n, 0.0);
    get(tilde_ye) = DataVector(n, 0.0);
    get(tilde_tau) = DataVector(n, 0.0);
    get(tilde_phi) = DataVector(n, 0.0);
    for (size_t i = 0; i < 3; ++i) {
      tilde_s.get(i) = DataVector(n, 0.0);
      tilde_b.get(i) = DataVector(n, 0.0);
    }
  }
};

template <typename Scheme>
void run_boundary_terms(BoundaryCorrection* bc, const SideState& interior,
                        const SideState& exterior) {
  Scheme::dg_boundary_terms(
      make_not_null(&bc->tilde_d), make_not_null(&bc->tilde_ye),
      make_not_null(&bc->tilde_tau), make_not_null(&bc->tilde_s),
      make_not_null(&bc->tilde_b), make_not_null(&bc->tilde_phi),
      get<grmhd_tags::TildeD>(interior.package_data),
      get<grmhd_tags::TildeYe>(interior.package_data),
      get<grmhd_tags::TildeTau>(interior.package_data),
      get<grmhd_tags::TildeS<Frame::Inertial>>(interior.package_data),
      get<grmhd_tags::TildeB<Frame::Inertial>>(interior.package_data),
      get<grmhd_tags::TildePhi>(interior.package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeD>>(interior.package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeYe>>(interior.package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeTau>>(interior.package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeS<Frame::Inertial>>>(
          interior.package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeB<Frame::Inertial>>>(
          interior.package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildePhi>>(interior.package_data),
      get<typename Marquina::CharacteristicSpeeds>(interior.package_data),
      get<typename Marquina::LeftCharacteristicFields>(interior.package_data),
      get<typename Marquina::RightCharacteristicFields>(interior.package_data),
      get<grmhd_tags::TildeD>(exterior.package_data),
      get<grmhd_tags::TildeYe>(exterior.package_data),
      get<grmhd_tags::TildeTau>(exterior.package_data),
      get<grmhd_tags::TildeS<Frame::Inertial>>(exterior.package_data),
      get<grmhd_tags::TildeB<Frame::Inertial>>(exterior.package_data),
      get<grmhd_tags::TildePhi>(exterior.package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeD>>(exterior.package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeYe>>(exterior.package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeTau>>(exterior.package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeS<Frame::Inertial>>>(
          exterior.package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildeB<Frame::Inertial>>>(
          exterior.package_data),
      get<::Tags::NormalDotFlux<grmhd_tags::TildePhi>>(exterior.package_data),
      get<typename Marquina::CharacteristicSpeeds>(exterior.package_data),
      get<typename Marquina::LeftCharacteristicFields>(exterior.package_data),
      get<typename Marquina::RightCharacteristicFields>(exterior.package_data),
      ::dg::Formulation::WeakInertial);
}

// Run the chunked, timed benchmark for `Scheme` over the given state pairs.
// Fills `results` (6 boundary-correction components per interface, order
// tilde_d, s_x, s_y, s_z, tilde_tau, tilde_ye) and returns ns/interface.
template <typename Scheme>
double run_benchmark(const PrimitiveStore& interior_prim,
                     const PrimitiveStore& exterior_prim,
                     const EquationsOfState::EquationOfState<true, 3>& eos,
                     std::vector<double>* results) {
  SideState interior;
  SideState exterior;
  interior.allocate(kChunkSize);
  exterior.allocate(kChunkSize);
  BoundaryCorrection bc;
  bc.allocate(kChunkSize);
  results->resize(kNumInterfaces * 6);

  std::chrono::steady_clock::duration kernel_time{0};
  for (size_t offset = 0; offset < kNumInterfaces; offset += kChunkSize) {
    const size_t n = std::min(kChunkSize, kNumInterfaces - offset);
    fill_side(&interior, interior_prim, offset, n, +1.0, eos);
    fill_side(&exterior, exterior_prim, offset, n, -1.0, eos);

    const auto t0 = std::chrono::steady_clock::now();
    run_package_data<Scheme>(&interior, eos);
    run_package_data<Scheme>(&exterior, eos);
    run_boundary_terms<Scheme>(&bc, interior, exterior);
    const auto t1 = std::chrono::steady_clock::now();
    kernel_time += t1 - t0;

    for (size_t i = 0; i < n; ++i) {
      double* r = &(*results)[(offset + i) * 6];
      r[0] = get(bc.tilde_d)[i];
      r[1] = bc.tilde_s.get(0)[i];
      r[2] = bc.tilde_s.get(1)[i];
      r[3] = bc.tilde_s.get(2)[i];
      r[4] = get(bc.tilde_tau)[i];
      r[5] = get(bc.tilde_ye)[i];
    }
  }
  const double total_ns = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_time)
          .count());
  return total_ns / static_cast<double>(kNumInterfaces);
}

inline void write_binary(const std::vector<double>& data,
                         const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    std::fprintf(stderr, "ERROR: cannot open %s for writing.\n", path.c_str());
    std::exit(1);
  }
  std::fwrite(data.data(), sizeof(double), data.size(), f);
  std::fclose(f);
}

// Compare `out` against a saved binary reference (rel `rel_tol` / abs
// `abs_tol`). Returns true if they match or the reference is absent; on
// mismatch prints the worst offender.
inline bool compare_with_reference(const std::vector<double>& out,
                                   const std::string& ref_path, double rel_tol,
                                   double abs_tol, const char* label) {
  std::FILE* f = std::fopen(ref_path.c_str(), "rb");
  if (f == nullptr) {
    return true;
  }
  std::vector<double> ref(out.size());
  const size_t read = std::fread(ref.data(), sizeof(double), ref.size(), f);
  std::fclose(f);
  if (read != ref.size()) {
    std::fprintf(stderr, "ERROR: reference has %zu doubles, expected %zu.\n",
                 read, ref.size());
    return false;
  }
  double worst_rel = 0.0;
  size_t worst_idx = 0;
  bool ok = true;
  for (size_t i = 0; i < out.size(); ++i) {
    const double abs_diff = std::abs(out[i] - ref[i]);
    const double rel_diff = abs_diff / (std::abs(ref[i]) + 1.0e-300);
    if (abs_diff > abs_tol and rel_diff > rel_tol) {
      ok = false;
      if (rel_diff > worst_rel) {
        worst_rel = rel_diff;
        worst_idx = i;
      }
    }
  }
  if (not ok) {
    std::fprintf(stderr,
                 "ERROR: %s output diverged from reference. Worst at index "
                 "%zu: got %.17g, expected %.17g (rel %.3e).\n",
                 label, worst_idx, out[worst_idx], ref[worst_idx], worst_rel);
  }
  return ok;
}

// Print the value at percentile `pct` (0..100) of an already-collected diff
// sample, using nth_element on a scratch copy.
inline double percentile(std::vector<double>* sample, double pct) {
  if (sample->empty()) {
    return 0.0;
  }
  const size_t k = static_cast<size_t>((pct / 100.0) *
                                       static_cast<double>(sample->size() - 1));
  std::nth_element(sample->begin(), sample->begin() + k, sample->end());
  return (*sample)[k];
}

// Check two output vectors agree within (rel `rel_tol`, abs `abs_tol`), the
// documented CPM-vs-full-Marquina equivalence envelope, and dump the abs and
// rel diff *distributions* (p50/p90/p99/p99.9/max) — the standalone equivalent
// of the MarquinaCpm.DumpEquivalence gate. Returns true if every component is
// inside the envelope.
inline bool check_envelope(const std::vector<double>& a,
                           const std::vector<double>& b, double rel_tol,
                           double abs_tol, const char* label) {
  double worst_abs = 0.0;
  double worst_rel = 0.0;
  size_t worst_idx = 0;
  bool ok = true;
  std::vector<double> abs_diffs(a.size());
  std::vector<double> rel_diffs(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    const double abs_diff = std::abs(a[i] - b[i]);
    const double rel_diff = abs_diff / (std::abs(b[i]) + 1.0e-300);
    abs_diffs[i] = abs_diff;
    rel_diffs[i] = rel_diff;
    worst_abs = std::max(worst_abs, abs_diff);
    worst_rel = std::max(worst_rel, rel_diff);
    if (abs_diff > abs_tol and rel_diff > rel_tol) {
      ok = false;
      worst_idx = i;
    }
  }
  std::printf(
      "%s equivalence dump over %zu samples (envelope rel %.0e / abs "
      "%.0e):\n",
      label, a.size(), rel_tol, abs_tol);
  std::printf("  abs diff: p50=%.2e p90=%.2e p99=%.2e p99.9=%.2e max=%.2e\n",
              percentile(&abs_diffs, 50.0), percentile(&abs_diffs, 90.0),
              percentile(&abs_diffs, 99.0), percentile(&abs_diffs, 99.9),
              worst_abs);
  std::printf("  rel diff: p50=%.2e p90=%.2e p99=%.2e p99.9=%.2e max=%.2e\n",
              percentile(&rel_diffs, 50.0), percentile(&rel_diffs, 90.0),
              percentile(&rel_diffs, 99.0), percentile(&rel_diffs, 99.9),
              worst_rel);
  if (not ok) {
    std::fprintf(stderr,
                 "ERROR: %s exceeded equivalence envelope (rel %.1e / abs "
                 "%.1e) at index %zu.\n",
                 label, rel_tol, abs_tol, worst_idx);
  }
  return ok;
}
}  // namespace bench_marquina
