// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

namespace grmhd::ValenciaDivClean::BoundaryCorrections::detail {
/*!
 * \brief The local frame in which the PLUTO-backed boundary corrections solve
 * the face problem on a curved background.
 *
 * Every ValenciaDivClean flux has the form
 * \f$F^i(\tilde U) = \sqrt{\gamma}\,[\alpha \hat F^i(U) - \beta^i U]\f$, with
 * \f$\hat F^i\f$ the flux measured by the normal (Eulerian) observer. In the
 * normal observer's frame, with the spatial vectors written in a
 * \f$\gamma\f$-orthonormal triad whose first leg is the unit normal, the face
 * problem is the special-relativistic one, and the coordinate face moves
 * through that frame with speed
 * \f$w = (\beta^i n_i + n_i v_g^i)/\alpha\f$ (\f$v_g\f$ the mesh velocity). The
 * numerical flux is then
 * \f$G = \alpha\sqrt{\gamma}\,[\hat F_\mathrm{sel} - w U_\mathrm{sel}]\f$, the
 * fan sampled on the ray \f$x/t = w\f$. This is a Galilean translation in the
 * local frame, not a boost, and it needs no special case where
 * \f$|w| \geq 1\f$ (a timelike face normal, inside a horizon): the ray then
 * lies outside the fan and the upwind flux is selected.
 *
 * The frame is built from both sides' packaged geometry, never from one side
 * alone: the lapse and the spatial metric are averaged, the normal shift and
 * the normal are averaged antisymmetrically (each side packages them against
 * its own outward normal), and the normal is renormalized in the averaged
 * metric. Swapping interior and exterior turns the triad
 * \f$(e_{(0)}, e_{(1)}, e_{(2)})\f$ into \f$(-e_{(0)}, e_{(1)}, -e_{(2)})\f$,
 * the mirror image of the one-dimensional problem, as in flat space.
 *
 * The contract, its limiting cases and the comparison with `HllcGr`'s hat
 * transform are written out in the HLLx campaign's `gr_tetrad_adapter.md`.
 */
struct FaceFrame {
  /// The lapse \f$\alpha\f$ at the face.
  double lapse;
  /// \f$\beta^i n_i + n_i v_g^i\f$, signed with the interior's outward normal.
  double shift_dot_normal;
  /// The speed of the face in the normal observer's frame,
  /// \f$w = \beta^n_\mathrm{eff}/\alpha\f$.
  double face_speed;
  /// \f$\sqrt{\gamma}\f$ of the averaged spatial metric.
  double sqrt_det_spatial_metric;
  /// \f$e_{(a)}^i\f$: `legs[a][i]`; `legs[0]` is the unit normal vector
  /// \f$n^i\f$.
  std::array<std::array<double, 3>, 3> legs;
  /// \f$e^{(a)}_i = \gamma_{ij} e_{(a)}^j\f$: `duals[a][i]`; `duals[0]` is the
  /// unit normal covector \f$n_i\f$.
  std::array<std::array<double, 3>, 3> duals;

  /// Triad components \f$X^{(a)} = e^{(a)}_i X^i\f$ of a vector.
  std::array<double, 3> components(const std::array<double, 3>& vector) const;
  /// \f$X^i = \sum_a c_a e_{(a)}^i\f$: a vector from triad components.
  std::array<double, 3> vector(const std::array<double, 3>& components) const;
  /// \f$S_i = \sum_a c_a e^{(a)}_i\f$: a covector from triad components.
  std::array<double, 3> covector(const std::array<double, 3>& components) const;
};

/*!
 * \brief Build the face frame of one face point from the two sides' packaged
 * geometry, or `std::nullopt` where it cannot be built.
 *
 * Symmetric spatial metrics are passed as their six independent components in
 * the order xx, xy, xz, yy, yz, zz. `shift_dot_normal_*` is each side's
 * \f$\beta^i n_i + n_i v_g^i\f$ against its own outward normal, and
 * `normal_*` its own outward unit normal covector. The frame is not built
 * (and the caller keeps its HLL flux) when the lapse or
 * \f$\det\gamma\f$ is not positive, the averaged normal has no positive norm,
 * or anything is not finite.
 */
std::optional<FaceFrame> make_face_frame(
    double lapse_int, double lapse_ext, double shift_dot_normal_int,
    double shift_dot_normal_ext,
    const std::array<double, 6>& spatial_metric_int,
    const std::array<double, 6>& spatial_metric_ext,
    const std::array<double, 3>& normal_int,
    const std::array<double, 3>& normal_ext);

/// The process-wide-per-thread counts of the curved-face path, for the
/// opt-in report and for unit tests.
struct FaceFrameCounts {
  /// Calls of `pluto_dg_boundary_terms`.
  size_t calls = 0;
  /// Face points solved by the flat (special-relativistic) path.
  size_t flat_points = 0;
  /// Face points on a curved background or a moving mesh.
  size_t curved_points = 0;
  /// Curved points whose fan was solved in the face frame.
  size_t curved_solved = 0;
  /// Curved points solved with the face outside the light cone, |w| >= 1.
  size_t outside_light_cone = 0;
  /// Curved points kept on the HLL flux because the frame could not be built.
  size_t fallback_frame = 0;
  /// Curved points kept on the HLL flux because a state is not admissible
  /// for the ideal-gas kernel (rho, eps or p not positive).
  size_t fallback_state = 0;
  /// Curved points kept on the HLL flux because the kernel samples only the
  /// ray x/t = 0 (D2-R) and the face moves, w != 0.
  size_t fallback_ray = 0;
  /// Curved points kept on the HLL flux because the kernel's result was not
  /// finite.
  size_t fallback_solver = 0;
  /// Points the packaging marked as unsupported (infinite flatness): HLL.
  size_t unsupported_points = 0;
  /// Curved points kept on the HLL flux because a side's density is at or
  /// below `LightSpeedDensityCutoff` (the light-speed bounds there, as `Hll`
  /// and `HllcGr` use in an atmosphere).
  size_t density_cutoff = 0;
  /// Points (flat or curved) kept on the HLL flux because
  /// `SPECTRE_PLUTO_FORCE_HLL=1` is set: the HLLE arm of a solver comparison.
  size_t forced_hll = 0;
};

/// This thread's counts since the last reset.
FaceFrameCounts face_frame_counts();
/// Reset this thread's counts (unit tests).
void reset_face_frame_counts();
/// Add to this thread's counts; with `SPECTRE_GR_TETRAD_STATS=N` (N > 0) in the
/// environment, every N-th call also prints the thread's totals, which is how
/// a run shows that its faces took the curved path.
void add_face_frame_counts(const FaceFrameCounts& increment);
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::detail
