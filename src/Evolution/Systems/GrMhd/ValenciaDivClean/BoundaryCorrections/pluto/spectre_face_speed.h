/* SPECTRE-MOD (ours, not PLUTO's): helpers for sampling a Riemann fan on a moving interface.
 *
 * The SpECTRE GR adapter (PlutoFaceFrame.cpp) hands the solvers states in the normal observer's
 * orthonormal frame, through which the coordinate face moves with speed w = (beta^n + n.v_mesh)/alpha
 * (Sweep::spectre_face_speed). The fan itself does not depend on w; only the ray it is sampled on
 * does, and the flux through the moving face is F_sel - w U_sel, U_sel the conserved state of the
 * region that contains the ray (energy in the reduced form E - D when RMHD_REDUCED_ENERGY, matching
 * the ENG flux). The normal total pressure stays in Sweep::press: it does not multiply a state.
 *
 * Every helper is static inline and stateless, so the vendored code stays thread-safe without a
 * lock. With w == 0 the solvers compute exactly the values they did before these edits (every
 * subtraction is guarded by w != 0.0, and the region tests compare against the same constants). */
#ifndef SPECTRE_FACE_SPEED_H
#define SPECTRE_FACE_SPEED_H

/* The face speed of interface i (0 when the caller passed none). */
static inline double SpectreFaceSpeed (const Sweep *sweep, int i)
{
  return sweep->spectre_face_speed == NULL ? 0.0 : sweep->spectre_face_speed[i];
}

/* flux -= w u over the NFLX conserved components. */
static inline void SpectreMovingFace (double *flux, double w, const double *u)
{
  int nv;
  for (nv = 0; nv < NFLX; nv++) flux[nv] -= w*u[nv];
}

/* flux -= w U_hll, the HLL state rebuilt from its definition (not from a stored Uhll, which a
   ConsToPrim call on it may have modified): U_hll = (SR uR - SL uL + fL - fR)/(SR - SL) with the
   normal total pressure, carried outside PLUTO's fluxes, restored in the normal momentum. */
static inline void SpectreMovingFaceHll (double *flux, double w,
                                         const double *uL, const double *uR,
                                         const double *fL, const double *fR,
                                         double pL, double pR, double SL, double SR)
{
  int nv;
  double scrh = 1.0/(SR - SL), u;
  for (nv = 0; nv < NFLX; nv++){
    u = (SR*uR[nv] - SL*uL[nv] + fL[nv] - fR[nv])*scrh;
    if (nv == MXn) u += (pL - pR)*scrh;
    flux[nv] -= w*u;
  }
}

#endif
