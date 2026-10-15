#ifndef D2R_SR_FACE_H
#define D2R_SR_FACE_H

#include "pluto.h"

/* Private fixed-Bn gamma-law SR HLLD helper types. The complete external
 * seven-state bridge is declared separately in spectre_d2r_core.h.
 * Sweep/solver scratch and these contexts are per-thread in the embedding. */
enum D2R_SR_Outcome {
  D2R_SR_INVALID = 0, D2R_SR_UPWIND_LEFT = 1,
  D2R_SR_UPWIND_RIGHT = 2, D2R_SR_HLL_FLATTEN = 3,
  D2R_SR_HLL_ROOT = 4, D2R_SR_HLLD = 5
};
enum D2R_SR_RootFailure {
  D2R_SR_BAD_GUESS = 1, D2R_SR_NONFINITE_ROOT = 2,
  D2R_SR_NATIVE_STATE_FAIL = 4, D2R_SR_ITERATION_LIMIT = 8,
  D2R_SR_GROWING_RESIDUAL = 16
};

typedef struct {
  int outcome, root_failure, root_iterations, root_attempted;
  int hll_recovery_called, hll_recovery_status;
  int finite_flux, input_valid, bounds_valid;
  int unresolved_velocity;
  unsigned int flags_before, flags_after;
  double sl, sr, pressure_guess, pressure_returned, last_root_residual;
  double alfven_left, alfven_right, contact;
} D2R_SR_Record;

typedef struct { Sweep sweep; double cmax[2]; int initialized; } D2R_SR_Context;

int D2R_SR_InitContext(D2R_SR_Context *context);

void D2R_HLLD_Fixed_Solver(const Sweep *, int, int, double *, Grid *,
                          const double *, const double *, D2R_SR_Record *);
#endif
