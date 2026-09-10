// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
/// @{
/*!
 * \brief Combine the interior and exterior packaged 3+1 scalars into the
 * single interface frame the Riemann algebra runs in.
 *
 * `dg_package_data` runs independently on each side of a face, so the two
 * sides must be combined before any Riemann algebra runs -- otherwise the left
 * and right states are handed to the solver in different frames and the
 * Riemann problem is not well posed.
 *
 * The lapse is a scalar and is averaged symmetrically. \f$\beta^n\f$ is
 * linear in the normal, so it is averaged *antisymmetrically*, matching the
 * \f$(n_i^\text{int} - n_i^\text{ext})/2\f$ convention used for the
 * interface normal itself. On a perfectly matched face
 * (\f$n^\text{ext} = -n^\text{int}\f$) this equals either side's value; on
 * a slightly mismatched face it enforces \f$G(\text{int},\text{ext}) =
 * -G(\text{ext},\text{int})\f$, which conservation requires.
 *
 * Both are templated on the data type so that solvers working point by point
 * (`HllcGr`) and solvers working on whole `DataVector`s (`HllemHydroYe`) share
 * one definition of the convention.
 */
template <typename DataType>
DataType interface_lapse(const DataType& lapse_int, const DataType& lapse_ext) {
  return 0.5 * (lapse_int + lapse_ext);
}

template <typename DataType>
DataType interface_shift_dot_normal(const DataType& shift_dot_normal_int,
                                    const DataType& shift_dot_normal_ext) {
  return 0.5 * (shift_dot_normal_int - shift_dot_normal_ext);
}
/// @}

/*!
 * \brief Map an Eulerian-frame characteristic speed to the coordinate frame,
 * \f$\lambda = \alpha\nu - \beta^n_\text{eff}\f$.
 *
 * This is the *only* part of the hat transform an HLLEM-type solver needs:
 * the coordinate-frame normal-flux Jacobian is
 * \f$A_\text{coord} = \alpha A_\text{Eulerian} - \beta^n_\text{eff} I\f$,
 * so the eigenvectors are shared between the frames and only the eigenvalues
 * move.
 */
template <typename DataType>
DataType coordinate_frame_speed(const DataType& lapse,
                                const DataType& shift_dot_normal,
                                const DataType& hat_speed) {
  return lapse * hat_speed - shift_dot_normal;
}

/*!
 * \brief Two-scalar ("hat") transform between the coordinate-frame Valencia
 * variables and the local Eulerian frame at a DG interface point.
 *
 * The Valencia normal flux of a conserved variable \f$U\f$ in the coordinate
 * frame is related to the flux measured by the local Eulerian observer by
 *
 * \f{align*}
 *   F^n_\text{coord}(U) = \alpha\,\hat F(U) - \beta^n_\text{eff}\,U ,
 * \f}
 *
 * so that
 *
 * \f{align*}
 *   \hat F(U) &= \frac{F^n_\text{coord}(U) + \beta^n_\text{eff}\,U}{\alpha} ,
 *   & \nu &= \frac{\lambda + \beta^n_\text{eff}}{\alpha} ,
 * \f}
 *
 * where \f$\lambda\f$ is a coordinate-frame characteristic speed and
 * \f$\nu\f$ the corresponding Eulerian-frame speed. States themselves do not
 * transform. Because the coordinate-frame normal-flux Jacobian is
 * \f$A_\text{coord} = \alpha A_\text{Eulerian} - \beta^n_\text{eff} I\f$, the
 * left and right eigenvectors are *identical* in the two frames; only the
 * eigenvalues are shifted and rescaled by `speed()` / `inverse_speed()`.
 *
 * The Eulerian-frame system satisfies the special-relativistic identities
 * (e.g. MB05's \f$\hat F(\tilde E) = n^i \tilde S_i\f$) that the flat-space
 * Riemann-solver algebra is built on, which is why `HllcGr` runs its MB05
 * kernel on hatted quantities. `HllemHydroYe` uses the same geometry to map
 * its Eulerian-frame characteristic speeds and eigenvectors into the
 * coordinate frame.
 *
 * `w_face()` is the coordinate speed of the (fixed) DG interface as seen in
 * the Eulerian frame; solvers that sample a Riemann fan must sample at that
 * ray rather than at zero.
 *
 * \note `shift_dot_normal` is \f$\beta^n_\text{eff} = \beta^i n_i + n\cdot
 * v_\text{mesh}\f$ if the caller folds the mesh velocity in, and the raw
 * \f$\beta^i n_i\f$ otherwise. `HllcGr` currently packages the raw value
 * (see `HLLC_GR_ONF_PLAN.md` section 6); `HllemHydroYe` folds in the mesh
 * velocity so that its speeds match the packaged characteristic speeds.
 */
struct HatTransform {
  /// Lapse \f$\alpha\f$ at the interface.
  double lapse;
  /// \f$\beta^n_\text{eff}\f$ at the interface, signed with the interior's
  /// outward normal.
  double shift_dot_normal;
  /// \f$1/\alpha\f$, cached because every forward transform needs it.
  double inverse_lapse;

  /// Eulerian-frame speed of the interface, \f$\beta^n_\text{eff}/\alpha\f$.
  double w_face() const { return shift_dot_normal * inverse_lapse; }

  /// Coordinate-frame speed \f$\lambda\f$ to Eulerian-frame speed
  /// \f$\nu = (\lambda + \beta^n_\text{eff})/\alpha\f$.
  double speed(const double coordinate_speed) const {
    return (coordinate_speed + shift_dot_normal) * inverse_lapse;
  }

  /// Coordinate-frame normal flux to Eulerian-frame normal flux,
  /// \f$\hat F = (F + \beta^n_\text{eff} U)/\alpha\f$.
  double flux(const double coordinate_flux, const double state) const {
    return (coordinate_flux + shift_dot_normal * state) * inverse_lapse;
  }

  /// Divide a coordinate-frame scalar density by the lapse. Used for
  /// \f$\tilde P = \alpha\sqrt\gamma p \to \sqrt\gamma p\f$.
  double undensitize_lapse(const double coordinate_scalar) const {
    return coordinate_scalar * inverse_lapse;
  }

  /// Eulerian-frame speed \f$\nu\f$ back to the coordinate frame,
  /// \f$\lambda = \alpha\nu - \beta^n_\text{eff}\f$.
  double inverse_speed(const double hat_speed) const {
    return coordinate_frame_speed(lapse, shift_dot_normal, hat_speed);
  }

  /// Eulerian-frame normal flux back to the coordinate frame,
  /// \f$F = \alpha \hat F - \beta^n_\text{eff} U\f$. Needs **both** the
  /// selected hatted flux and the selected state: dropping the state term
  /// is silently correct only for \f$\beta^n_\text{eff} = 0\f$.
  double inverse_flux(const double hat_flux, const double state) const {
    return lapse * hat_flux - shift_dot_normal * state;
  }
};

/*!
 * \brief Build the `HatTransform` for one interface point from the interior
 * and exterior packaged lapse and \f$\beta^n_\text{eff}\f$.
 *
 * Uses the `interface_lapse` / `interface_shift_dot_normal` averaging
 * convention documented above, and caches \f$1/\alpha\f$ for the forward
 * transforms.
 */
inline HatTransform interface_hat_transform(const double lapse_int,
                                            const double lapse_ext,
                                            const double shift_dot_normal_int,
                                            const double shift_dot_normal_ext) {
  const double lapse = interface_lapse(lapse_int, lapse_ext);
  return HatTransform{
      lapse,
      interface_shift_dot_normal(shift_dot_normal_int, shift_dot_normal_ext),
      1.0 / lapse};
}
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
