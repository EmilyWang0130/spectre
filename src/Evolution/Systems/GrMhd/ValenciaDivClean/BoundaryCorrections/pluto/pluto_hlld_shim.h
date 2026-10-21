/* C interface to PLUTO's RMHD Riemann solvers (HLLD and the two HLLCs), for
 * use from SpECTRE.
 *
 * Everything PLUTO-side is reached through this one entry point; the C++ side
 * never sees a PLUTO type. See README.md for the list of local modifications
 * made to the vendored PLUTO sources. */
#ifndef PLUTO_HLLD_SHIM_H
#define PLUTO_HLLD_SHIM_H
#ifdef __cplusplus
extern "C" {
#endif

/* Solve `npts` Riemann problems with PLUTO's HLLD_Solver.
 *
 *  vL, vR     [in]  npts*8, PLUTO RMHD PRIMITIVE order and units:
 *                   {RHO, VX1, VX2, VX3, BX1, BX2, BX3, PRS}
 *                   ALREADY ROTATED so component 1 (VX1) and 4 (BX1) are the
 *                   interface-normal velocity and magnetic field.
 *  gamma      [in]  ideal-gas adiabatic index (PLUTO's EOS is IDEAL only).
 *  flux_out   [out] npts*8, PLUTO CONSERVED order
 *                   {RHO, MX1, MX2, MX3, BX1, BX2, BX3, ENG}, where ENG is the
 *                   REDUCED energy E - D (RMHD_REDUCED_ENERGY == YES), which is
 *                   what SpECTRE's TildeTau uses.
 *  press_out  [out] npts, the normal total-pressure term. PLUTO carries this
 *                   OUTSIDE the momentum flux, so the caller must add it to the
 *                   normal momentum component itself.
 *
 * Returns 0 on success, non-zero if the solve could not be set up.
 * Thread-safe: all PLUTO state it touches is _Thread_local, and each calling
 * thread lazily builds its own Sweep. */
int pluto_hlld_flux(int npts, const double* vL, const double* vR, double gamma,
                    double* flux_out, double* press_out);

/* The same contract, through PLUTO's two RMHD HLLC solvers instead:
 *   pluto_hllc_mb_flux -- HLLC_MB_Solver (hllc_mb.c), Mignone & Bodo 2006,
 *                         PLUTO's default "hllc";
 *   pluto_hllc_kb_flux -- HLLC_KB_Solver (hllc_kb.c), which its header
 *                         attributes to Balsara & Kim 2016.
 * Both fall back to the HLL flux internally where their star state is
 * unphysical. */
int pluto_hllc_mb_flux(int npts, const double* vL, const double* vR,
                       double gamma, double* flux_out, double* press_out);
int pluto_hllc_kb_flux(int npts, const double* vL, const double* vR,
                       double gamma, double* flux_out, double* press_out);

/* The same contract, through the seven-wave solver HLLX_Solver (hllx.c, our
 * addition to PLUTO: PLUTO's HLLD stage, then slow fronts closed by a Newton
 * solve; falls back to HLLD's flux where that solve fails, and to HLL where
 * HLLD's own root does). */
int pluto_hllx_flux(int npts, const double* vL, const double* vR, double gamma,
                    double* flux_out, double* press_out);

/* The same four solvers with the fan sampled on a moving interface (the SpECTRE
 * GR adapter, PlutoFaceFrame.hpp):
 *  face_speed [in]  npts, the speed w of each interface in the frame the states
 *                   are given in (the normal observer's orthonormal frame:
 *                   w = (beta^n + n.v_mesh)/alpha).
 *  flux_out   [out] the flux through the moving interface, F_sel - w U_sel, in
 *                   the same layout as above (ENG reduced, U_sel's energy too);
 *  press_out  [out] the normal total pressure, as above (unchanged by w).
 * With every face_speed 0 these reproduce the static entry points bit for bit
 * (Test_PlutoFaceFrame.cpp). Thread-safe on the same terms: the face speeds
 * ride in the calling thread's Sweep. */
int pluto_hlld_flux_moving(int npts, const double* vL, const double* vR,
                           double gamma, const double* face_speed,
                           double* flux_out, double* press_out);
int pluto_hllc_mb_flux_moving(int npts, const double* vL, const double* vR,
                              double gamma, const double* face_speed,
                              double* flux_out, double* press_out);
int pluto_hllc_kb_flux_moving(int npts, const double* vL, const double* vR,
                              double gamma, const double* face_speed,
                              double* flux_out, double* press_out);
int pluto_hllx_flux_moving(int npts, const double* vL, const double* vR,
                           double gamma, const double* face_speed,
                           double* flux_out, double* press_out);

#ifdef __cplusplus
}
#endif
#endif
