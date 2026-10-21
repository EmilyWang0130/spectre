// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoFaceFrame.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <optional>

#include "Parallel/Printf/Printf.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections::detail {
namespace {
// (i, j) -> position in the (xx, xy, xz, yy, yz, zz) storage.
constexpr std::array<std::array<size_t, 3>, 3> symmetric_index{
    {{{0, 1, 2}}, {{1, 3, 4}}, {{2, 4, 5}}}};

bool all_finite(const std::array<double, 3>& v) {
  return std::isfinite(v[0]) and std::isfinite(v[1]) and std::isfinite(v[2]);
}

thread_local FaceFrameCounts counts{};
const size_t report_every = [] {
  const char* e = std::getenv("SPECTRE_GR_TETRAD_STATS");
  const long n = e == nullptr ? 0L : std::atol(e);
  return n > 0 ? static_cast<size_t>(n) : size_t{0};
}();
}  // namespace

std::array<double, 3> FaceFrame::components(
    const std::array<double, 3>& vector) const {
  std::array<double, 3> result{};
  for (size_t a = 0; a < 3; ++a) {
    result[a] = duals[a][0] * vector[0] + duals[a][1] * vector[1] +
                duals[a][2] * vector[2];
  }
  return result;
}

std::array<double, 3> FaceFrame::vector(
    const std::array<double, 3>& components) const {
  std::array<double, 3> result{};
  for (size_t i = 0; i < 3; ++i) {
    result[i] = components[0] * legs[0][i] + components[1] * legs[1][i] +
                components[2] * legs[2][i];
  }
  return result;
}

std::array<double, 3> FaceFrame::covector(
    const std::array<double, 3>& components) const {
  std::array<double, 3> result{};
  for (size_t i = 0; i < 3; ++i) {
    result[i] = components[0] * duals[0][i] + components[1] * duals[1][i] +
                components[2] * duals[2][i];
  }
  return result;
}

std::optional<FaceFrame> make_face_frame(
    const double lapse_int, const double lapse_ext,
    const double shift_dot_normal_int, const double shift_dot_normal_ext,
    const std::array<double, 6>& spatial_metric_int,
    const std::array<double, 6>& spatial_metric_ext,
    const std::array<double, 3>& normal_int,
    const std::array<double, 3>& normal_ext) {
  const double lapse = 0.5 * (lapse_int + lapse_ext);
  if (not(std::isfinite(lapse) and lapse > 0.0)) {
    return std::nullopt;
  }
  std::array<std::array<double, 3>, 3> g{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      const size_t k = symmetric_index[i][j];
      g[i][j] = 0.5 * (spatial_metric_int[k] + spatial_metric_ext[k]);
    }
  }
  // Cofactors of the symmetric 3x3 metric.
  const double c00 = g[1][1] * g[2][2] - g[1][2] * g[1][2];
  const double c01 = g[0][2] * g[1][2] - g[0][1] * g[2][2];
  const double c02 = g[0][1] * g[1][2] - g[0][2] * g[1][1];
  const double c11 = g[0][0] * g[2][2] - g[0][2] * g[0][2];
  const double c12 = g[0][1] * g[0][2] - g[0][0] * g[1][2];
  const double c22 = g[0][0] * g[1][1] - g[0][1] * g[0][1];
  const double det = g[0][0] * c00 + g[0][1] * c01 + g[0][2] * c02;
  if (not(std::isfinite(det) and det > 0.0)) {
    return std::nullopt;
  }
  const double inv_det = 1.0 / det;
  const std::array<std::array<double, 3>, 3> g_inv{
      {{{c00 * inv_det, c01 * inv_det, c02 * inv_det}},
       {{c01 * inv_det, c11 * inv_det, c12 * inv_det}},
       {{c02 * inv_det, c12 * inv_det, c22 * inv_det}}}};

  std::array<double, 3> normal{};
  for (size_t i = 0; i < 3; ++i) {
    normal[i] = 0.5 * (normal_int[i] - normal_ext[i]);
  }
  double norm_squared = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      norm_squared += g_inv[i][j] * normal[i] * normal[j];
    }
  }
  if (not(std::isfinite(norm_squared) and norm_squared > 0.0)) {
    return std::nullopt;
  }
  const double inv_norm = 1.0 / std::sqrt(norm_squared);
  for (size_t i = 0; i < 3; ++i) {
    normal[i] *= inv_norm;
  }

  FaceFrame frame{};
  frame.lapse = lapse;
  // Each side's normal shift is linear in its own normal, so it is averaged
  // antisymmetrically like the normal and rescaled by the same factor.
  frame.shift_dot_normal =
      0.5 * (shift_dot_normal_int - shift_dot_normal_ext) * inv_norm;
  frame.face_speed = frame.shift_dot_normal / lapse;
  frame.sqrt_det_spatial_metric = std::sqrt(det);

  // e_(0) = n^i, unit in gamma; e^(0) = n_i.
  frame.duals[0] = normal;
  for (size_t i = 0; i < 3; ++i) {
    frame.legs[0][i] = g_inv[i][0] * normal[0] + g_inv[i][1] * normal[1] +
                       g_inv[i][2] * normal[2];
  }
  // e_(1): the gamma-orthogonal part of a coordinate axis, x unless the
  // normal is within a gamma-cosine of 0.9 of it. For the axis k,
  // gamma(axis, e_(0)) = n_k and |axis - n_k e_(0)|^2 = gamma_kk - n_k^2.
  const size_t axis =
      std::abs(normal[0]) < 0.9 * std::sqrt(g[0][0]) ? size_t{0} : size_t{1};
  const double axis_norm_squared = g[axis][axis] - normal[axis] * normal[axis];
  if (not(std::isfinite(axis_norm_squared) and axis_norm_squared > 0.0)) {
    return std::nullopt;
  }
  const double inv_axis_norm = 1.0 / std::sqrt(axis_norm_squared);
  for (size_t i = 0; i < 3; ++i) {
    frame.legs[1][i] =
        ((i == axis ? 1.0 : 0.0) - normal[axis] * frame.legs[0][i]) *
        inv_axis_norm;
  }
  for (size_t i = 0; i < 3; ++i) {
    frame.duals[1][i] = g[i][0] * frame.legs[1][0] +
                        g[i][1] * frame.legs[1][1] + g[i][2] * frame.legs[1][2];
  }
  // e^(2)_l = sqrt(gamma) eps_{ljk} e_(0)^j e_(1)^k (the gamma cross product,
  // already a covector and of unit norm), e_(2)^i = gamma^{il} e^(2)_l.
  const auto& e0 = frame.legs[0];
  const auto& e1 = frame.legs[1];
  const double sqrt_det = frame.sqrt_det_spatial_metric;
  frame.duals[2] = {{sqrt_det * (e0[1] * e1[2] - e0[2] * e1[1]),
                     sqrt_det * (e0[2] * e1[0] - e0[0] * e1[2]),
                     sqrt_det * (e0[0] * e1[1] - e0[1] * e1[0])}};
  for (size_t i = 0; i < 3; ++i) {
    frame.legs[2][i] = g_inv[i][0] * frame.duals[2][0] +
                       g_inv[i][1] * frame.duals[2][1] +
                       g_inv[i][2] * frame.duals[2][2];
  }
  for (size_t a = 0; a < 3; ++a) {
    if (not(all_finite(frame.legs[a]) and all_finite(frame.duals[a]))) {
      return std::nullopt;
    }
  }
  if (not std::isfinite(frame.face_speed)) {
    return std::nullopt;
  }
  return frame;
}

FaceFrameCounts face_frame_counts() { return counts; }

void reset_face_frame_counts() { counts = FaceFrameCounts{}; }

void add_face_frame_counts(const FaceFrameCounts& increment) {
  counts.calls += increment.calls;
  counts.flat_points += increment.flat_points;
  counts.curved_points += increment.curved_points;
  counts.curved_solved += increment.curved_solved;
  counts.outside_light_cone += increment.outside_light_cone;
  counts.fallback_frame += increment.fallback_frame;
  counts.fallback_state += increment.fallback_state;
  counts.fallback_ray += increment.fallback_ray;
  counts.fallback_solver += increment.fallback_solver;
  counts.unsupported_points += increment.unsupported_points;
  counts.density_cutoff += increment.density_cutoff;
  counts.forced_hll += increment.forced_hll;
  // The forced-HLL switch announces itself once per process, whatever the
  // stats cadence, so a run's log shows that its HLLE arm was HLLE.
  static std::atomic<bool> forced_hll_announced{false};
  if (increment.forced_hll > 0 and not forced_hll_announced.exchange(true)) {
    Parallel::printf(
        "PLUTO-backed corrections: SPECTRE_PLUTO_FORCE_HLL is set; every face "
        "keeps the HLL flux.\n");
  }
  if (report_every > 0 and increment.calls > 0 and
      counts.calls % report_every == 0) {
    Parallel::printf(
        "GR tetrad stats after %zu calls (this thread): flat %zu, curved %zu "
        "(solved %zu, |w|>=1 %zu; HLL kept: frame %zu, state %zu, ray %zu, "
        "solver %zu), unsupported %zu, density cutoff %zu, forced HLL %zu\n",
        counts.calls, counts.flat_points, counts.curved_points,
        counts.curved_solved, counts.outside_light_cone, counts.fallback_frame,
        counts.fallback_state, counts.fallback_ray, counts.fallback_solver,
        counts.unsupported_points, counts.density_cutoff, counts.forced_hll);
  }
}
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::detail
