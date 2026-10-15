/* Private persistent context allocation for the SpECTRE D2-R scalar bridge.
 * The caller sets NMAX_POINT >=1026 before this first allocation, matching
 * the legacy 1024-interface shim and its required two-point headroom.
 * No vendor global definitions, floors, time or step counters are written. */
#include "d2r_sr_face.h"

int D2R_SR_InitContext(D2R_SR_Context *context) {
  if (!context || NMAX_POINT < 1026) return 0;
  if (!context->initialized) {
    MakeState(&context->sweep);
    context->initialized = 1;
  }
  return 1;
}
