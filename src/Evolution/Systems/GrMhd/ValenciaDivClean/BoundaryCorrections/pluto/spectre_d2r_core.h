#ifndef SPECTRE_D2R_SR_CORE_H
#define SPECTRE_D2R_SR_CORE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Seven-state laboratory frame, common fixed normal B, IDEAL EOS:
 * primitive = [rho,vn,vt,vb,p,Bt,Bb]
 * conserved = [D,mn,mt,mb,tau,Bt,Bb], tau=E-D.
 * All returned momentum fluxes INCLUDE normal pressure.
 * Grid/context, solver scratch, maximum-Mach diagnostic and allocation
 * accounting are thread-local. Independent workers may call this bridge
 * concurrently, including their first scratch allocation.
 */
/* ABI 4: the diagnostics output grew from 8 to 16 doubles in design D8.
 * Reject old ABI-3 clients before they allocate an undersized output buffer. */
#define D2R_CORE_ABI 4
#define D2R_CORE_NI 14
#define D2R_CORE_ND 16
int d2r_sr_core_abi(void);
int d2r_sr_fixed_precision(void);
/* Mode 0 native (recomputes bounds), 1 fixed supplied bounds, 2 physical maps,
 * 3 mode 0's bounds and with them, when they straddle the face (SL < 0 < SR),
 * mode 1's solve, otherwise mode 0's.
 * faces[n][16]=[gamma,Bn,wl[7],wr[7]], fixed[n][2].
 * numerical[n][7], states[n][2][7], physical[n][2][7], used_bounds[n][2].
 * integers[n][14]: status,input_valid,bounds_valid,finite_flux,
 * native_iterations,outcome,root_failure,root_iterations,root_attempted,
 * hll_recovery_called,hll_recovery_status,unresolved_velocity,
 * flags_before,flags_after.
 * doubles[n][16]: SL,SR,pressure_guess,pressure_returned,last_root_residual,
 * alfven_left,alfven_right,contact, then the left state's magnetosonic speeds
 * fast-,slow-,slow+,fast+ and the right state's (PLUTO's Magnetosonic, as the
 * pair's checks compute them; NaN where they did not).
 * Native-only root quantities are unavailable (-1 / NaN); no shadow solve.
 * status: 1 usable; 0 invalid primitive; -1 nonfinite map; -2 speed failure;
 * -3 invalid bounds; -4 nonfinite numerical flux; -5 velocity rounding conflict.
 * No input floors or flux repair are added by this API. Native HLLD may take
 * its original HLL recovery/fallback path; fixed records expose that path.
 * Return 0 only on API/allocation setup failure; per-face rejection is status.
 */
int d2r_sr_core_batch(size_t n, int mode, const double *faces,
 const double *fixed, double *numerical, double *states, double *physical,
 double *used_bounds, int *integers, double *doubles);
#ifdef __cplusplus
}
#endif
#endif
