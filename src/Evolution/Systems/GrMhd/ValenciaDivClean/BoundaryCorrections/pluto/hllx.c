/* SPECTRE-VENDORED copy of Src/RMHD/hllx.c from the local PLUTO 4.4-patch4 tree
   (~/refs/pluto/pluto-4.4-patch4/PLUTO, commit dca07c7, 2026-10-06). The only
   changes are the _Thread_local qualifiers marked SPECTRE-MOD (SpECTRE runs
   Charm++ SMP; see README.md). Keep it identical to the PLUTO-tree copy
   otherwise: that copy is the one validated against the numpy reference. */
/* ///////////////////////////////////////////////////////////////////// */
/*!
  \file
  \brief Seven-wave HLLx Riemann solver for the relativistic MHD equations.

  HLLD (Mignone, Ugliano & Bodo 2009; this directory's hlld.c) extended by the
  two slow magnetosonic fronts, after Zhang, Poedts & Lani 2026 (HLLx,
  Newtonian) but with a relativistic closure of its own (X. Wang, task 2 of
  2026-10-01; numpy reference notes/projects/active/hllx_rmhd/scripts/rmhd_py/,
  which this file mirrors step by step).

  Fan:  L | fL | aL | sL | sR | aR | fR | R  with speeds
        S_fL < S_aL < S_sL < S_m < S_sR < S_aR < S_fR.
  Fast fronts: MUB09's jump map at trial total pressures p_fL, p_fR. Rotational
  fronts: MUB09's K-vector family with trial transverse fields B_t,aL, B_t,aR
  (p_T and w_T carried across, D from the mass bracket). Slow fronts: the same
  jump map at the shared central pressure p_m. The seven unknowns
  z = (p_fL, p_fR, p_m, By_aL, Bz_aL, By_aR, Bz_aR) solve the seven residuals
  [v] = 0 and [B_t] = 0 across the contact, and C_EOS = 0 in the two central
  states (the relaxed states behind the fast and rotational fronts stay off the
  EOS, as in HLLD). The slow speeds are the mean of the two input states' slow
  eigenvalues, kept at least HLLX_THETA of their (S_a, S_c) interval away from
  both ends.

  Per interface: PLUTO's HLLD exactly as in hlld.c (same secant, same checks),
  retried once with a probe of 1 + 1e-4 instead of 1.025 where the first probe
  itself failed, and accepted only if the states behind the fast and rotational
  fronts are subluminal with positive nominal gas pressure (hx_hlld_root; the
  result is the fallback and the seed; HLLDR_Solver, solver name "hlldr", stops
  here); then, if HLLD found an admissible root with
  B_x^2 >= 0.01 max(p_L, p_R), a damped Newton from z0 = (p, p, p, B_c, B_c)
  on residuals scaled per face (hx_scales: unit-invariant, tolerance HLLX_TOL
  relative to those scales) with a forward-difference Jacobian and
  Levenberg-Marquardt retries, then, if that fails, continuation in the slow
  speeds from the middle of their intervals. The flux of a converged root is assembled by Rankine-Hugoniot
  recursion from the nearer side, so the fan is integral-consistent in every
  component. Anything else returns HLLD's flux.

  Everything inside the seven-wave part uses the "task layout": conserved
  U = (D, m_x, m_y, m_z, E, B_y, B_z) with the TOTAL energy and the total
  pressure inside the normal momentum flux; the auxiliary tuple
  X = (D, v_x, v_y, v_z, B_y, B_z, w_T, p_T). PLUTO's reduced energy and
  separate pressure are converted at the entry and exit.

  Statistics (interfaces per outcome) live in static counters; HLLX_Stats()
  returns them and, with HLLX_STATS_EVERY > 0, prints them every that many
  calls of the solver.
*/
/* ///////////////////////////////////////////////////////////////////// */
#include "pluto.h"
#include "spectre_face_speed.h"   /* SPECTRE-MOD */
#include <string.h>

#ifndef HLLD_MAX_ITER
  #define HLLD_MAX_ITER  20
#endif
#ifndef HLLX_THETA
  #define HLLX_THETA 0.05
#endif
#ifndef HLLX_TOL
  #define HLLX_TOL 1.e-10      /* on the scaled residual norm; the scales are relative */
#endif
#ifndef HLLX_EOS_FLOOR
  #define HLLX_EOS_FLOOR 1.e-5 /* EOS rows scaled by max(p*, HLLX_EOS_FLOOR w_T) */
#endif
#ifndef HLLX_MAX_NEWTON
  #define HLLX_MAX_NEWTON 60
#endif
#ifndef HLLX_CONT_STEPS
  #define HLLX_CONT_STEPS 4
#endif
#ifndef HLLX_STATS_EVERY
  #define HLLX_STATS_EVERY 0
#endif

enum {HX_SUPERSONIC = 0, HX_HLL, HX_BX0, HX_HLLX, HX_HLLX_CONT, HX_HLLD, HX_BT0, HX_BT0F, HX_NSTAT};
static _Thread_local long hx_stat[HX_NSTAT];                 /* SPECTRE-MOD */
static _Thread_local long hx_newton_iters, hx_calls;         /* SPECTRE-MOD */
static _Thread_local int  hx_seven_waves = 1;   /* SPECTRE-MOD */    /* 0: HLLDR_Solver, HLLD with the converged root only */

/* ===================================================================
   1. The HLLD machinery, from hlld.c (renamed; hx_Sc and hx_Bx play the
      roles of hlld.c's file statics Sc and Bx).
   =================================================================== */

typedef struct HX_RIEMANN_STATE{
  int    fail;
  double vx, vy, vz;
  double Bx, By, Bz;
  double Kx, Ky, Kz, K2;
  double w, sw, p, rho;
  double u[NFLX], R[NVAR];
  double S;
  double Sa;
} HX_Riemann_State;

static _Thread_local double hx_Sc, hx_Bx;   /* SPECTRE-MOD */

static int HX_GetRiemannState (HX_Riemann_State *Pv, double p, int side)
{
  double A, C, G, X, s;
  double vx, vy, vz, scrh, S, *R;

  S = Pv->S;
  R = Pv->R;

  A = R[MXn] + p*(1.0 - S*S) - S*R[ENG];
  G = R[BXt]*R[BXt] + R[BXb]*R[BXb];
  C = R[BXt]*R[MXt] + R[BXb]*R[MXb];
  X = hx_Bx*(A*S*hx_Bx + C) - (A + G)*(S*p + R[ENG]);

  vx = ( hx_Bx*(A*hx_Bx + C*S) - (R[MXn] + p)*(G + A) );
  vy = ( - (A + G - hx_Bx*hx_Bx*(1.0 - S*S))*R[MXt]
         + R[BXt]*(C + hx_Bx*(S*R[MXn] - R[ENG])) );
  vz = ( - (A + G - hx_Bx*hx_Bx*(1.0 - S*S))*R[MXb]
         + R[BXb]*(C + hx_Bx*(S*R[MXn] - R[ENG])) );

  scrh = vx*R[MXn] + vy*R[MXt] + vz*R[MXb];
  scrh = X*R[ENG] - scrh;
  Pv->w = p + scrh/(X*S - vx);

  if (Pv->w < 0.0) return 0;

  Pv->vx = vx/X;
  Pv->vy = vy/X;
  Pv->vz = vz/X;

  Pv->Bx = hx_Bx;
  Pv->By = -(R[BXt]*(S*p + R[ENG]) - hx_Bx*R[MXt])/A;
  Pv->Bz = -(R[BXb]*(S*p + R[ENG]) - hx_Bx*R[MXb])/A;

  s  = hx_Bx > 0.0 ? 1.0:-1.0;
  if (side < 0) s *= -1.0;

  Pv->sw = s*sqrt(Pv->w);

  scrh = 1.0/(S*p +  R[ENG] + hx_Bx*Pv->sw);
  Pv->Kx = scrh*(R[MXn] + p + R[BXn]*Pv->sw);
  Pv->Ky = scrh*(R[MXt]     + R[BXt]*Pv->sw);
  Pv->Kz = scrh*(R[MXb]     + R[BXb]*Pv->sw);

  Pv->K2 = Pv->Kx*Pv->Kx + Pv->Ky*Pv->Ky + Pv->Kz*Pv->Kz;
  return 1;
}

static double HX_Fstar (HX_Riemann_State *PaL, HX_Riemann_State *PaR, double p)
{
  int    success = 1;
  double dK, Bxc, Byc, Bzc;
  double fun;
  double vxcL, KLBc;
  double vxcR, KRBc;

  success *= HX_GetRiemannState (PaL, p, -1);
  success *= HX_GetRiemannState (PaR, p,  1);

  dK  = PaR->Kx - PaL->Kx + 1.e-12;

  Bxc = hx_Bx*dK;

  Byc =   PaR->By*(PaR->Kx - PaR->vx)
        - PaL->By*(PaL->Kx - PaL->vx)
        + hx_Bx*(PaR->vy - PaL->vy);

  Bzc =   PaR->Bz*(PaR->Kx - PaR->vx)
        - PaL->Bz*(PaL->Kx - PaL->vx)
        + hx_Bx*(PaR->vz - PaL->vz);

  KLBc = PaL->Kx*Bxc + PaL->Ky*Byc + PaL->Kz*Bzc;
  KRBc = PaR->Kx*Bxc + PaR->Ky*Byc + PaR->Kz*Bzc;

  vxcL = PaL->Kx - dK*hx_Bx*(1.0 - PaL->K2)/(PaL->sw*dK - KLBc);
  vxcR = PaR->Kx - dK*hx_Bx*(1.0 - PaR->K2)/(PaR->sw*dK - KRBc);

  PaL->Sa = PaL->Kx;
  PaR->Sa = PaR->Kx;
  hx_Sc   = 0.5*(vxcL + vxcR);
  fun     = vxcL - vxcR;

  success  = (vxcL - PaL->Kx)  > -1.e-6;
  success *= (PaR->Kx  - vxcR) > -1.e-6;

  success *= (PaL->S - PaL->vx) < 0.0;
  success *= (PaR->S - PaR->vx) > 0.0;

  success *= (PaR->w - p) > 0.0;
  success *= (PaL->w - p) > 0.0;
  success *= (PaL->Sa - PaL->S)  > -1.e-6;
  success *= (PaR->S  - PaR->Sa) > -1.e-6;

  PaL->fail = !success;
  return (fun);
}

static void HX_GetAState (HX_Riemann_State *Pa,  double p)
{
  double vB, *ua, *R, S;
  double scrh;

  ua = Pa->u;
  S  = Pa->S;
  R  = Pa->R;

  scrh = 1.0/(S - Pa->vx);

  ua[RHO] =  R[RHO]*scrh;
  ua[BXn] =  hx_Bx;
  ua[BXt] = (R[BXt] - hx_Bx*Pa->vy)*scrh;
  ua[BXb] = (R[BXb] - hx_Bx*Pa->vz)*scrh;

  vB     = Pa->vx*ua[BXn] + Pa->vy*ua[BXt] + Pa->vz*ua[BXb];
  ua[ENG] = (R[ENG] + p*Pa->vx - vB*hx_Bx)*scrh;

  ua[MXn] = (ua[ENG] + p)*Pa->vx - vB*ua[BXn];
  ua[MXt] = (ua[ENG] + p)*Pa->vy - vB*ua[BXt];
  ua[MXb] = (ua[ENG] + p)*Pa->vz - vB*ua[BXb];
}

static void HX_GetCState (HX_Riemann_State *PaL, HX_Riemann_State *PaR, double p,
                          double *Uc, double wf)   /* SPECTRE-MOD: + wf, the face speed */
{
  double dK, *ua;
  double vxcL, vycL, vzcL, KLBc;
  double vxcR, vycR, vzcR, KRBc;
  double vxc, vyc, vzc, vBc;
  double Bxc, Byc, Bzc, Sa, vxa;
  double scrhL, scrhR;

  HX_GetAState (PaL, p);
  HX_GetAState (PaR, p);
  dK = (PaR->Kx - PaL->Kx) + 1.e-12;

  Bxc = hx_Bx*dK;

  Byc =   PaR->By*(PaR->Kx - PaR->vx)
        - PaL->By*(PaL->Kx - PaL->vx)
        + hx_Bx*(PaR->vy - PaL->vy);

  Bzc =   PaR->Bz*(PaR->Kx - PaR->vx)
        - PaL->Bz*(PaL->Kx - PaL->vx)
        + hx_Bx*(PaR->vz - PaL->vz);

  Bxc  = hx_Bx;
  Byc /= dK;
  Bzc /= dK;

  Uc[BXn] = Bxc;
  Uc[BXt] = Byc;
  Uc[BXb] = Bzc;

  KLBc = PaL->Kx*Bxc + PaL->Ky*Byc + PaL->Kz*Bzc;
  KRBc = PaR->Kx*Bxc + PaR->Ky*Byc + PaR->Kz*Bzc;

  scrhL = (1.0 - PaL->K2)/(PaL->sw - KLBc);
  scrhR = (1.0 - PaR->K2)/(PaR->sw - KRBc);

  vxcL = PaL->Kx - Uc[BXn]*scrhL;
  vxcR = PaR->Kx - Uc[BXn]*scrhR;

  vycL = PaL->Ky - Uc[BXt]*scrhL;
  vycR = PaR->Ky - Uc[BXt]*scrhR;

  vzcL = PaL->Kz - Uc[BXb]*scrhL;
  vzcR = PaR->Kz - Uc[BXb]*scrhR;

  vxc = 0.5*(vxcL + vxcR);
  vyc = 0.5*(vycL + vycR);
  vzc = 0.5*(vzcL + vzcR);

  if (vxc > wf) {   /* SPECTRE-MOD: 0 -> wf (the face is left of the contact) */
    HX_GetAState (PaL, p);
    ua  = PaL->u;
    Sa  = PaL->Sa;
    vxa = PaL->vx;
  } else {
    HX_GetAState (PaR, p);
    ua  = PaR->u;
    Sa  = PaR->Sa;
    vxa = PaR->vx;
  }

  vBc = vxc*Uc[BXn] + vyc*Uc[BXt] + vzc*Uc[BXb];

  Uc[RHO] = ua[RHO]*(Sa - vxa)/(Sa - vxc);
  Uc[ENG] = (Sa*ua[ENG] - ua[MXn] + p*vxc - vBc*hx_Bx)/(Sa - vxc);

  Uc[MXn] = (Uc[ENG] + p)*vxc - vBc*hx_Bx;
  Uc[MXt] = (Uc[ENG] + p)*vyc - vBc*Uc[BXt];
  Uc[MXb] = (Uc[ENG] + p)*vzc - vBc*Uc[BXb];
}

/* hlld.c's secant iteration, verbatim, with the first probe p = probe*p0 as a parameter
   (hlld.c uses 1.025). Returns 1 on success with *p_out the final pressure; the Riemann
   states are then re-evaluated at that pressure (hlld.c leaves them one iterate behind). */
static int hx_hlld_secant (HX_Riemann_State *PaL, HX_Riemann_State *PaR, double p0,
                           double probe, double *p_out, int *kout)
{
  int k = 0, switch_to_hll = 0;
  double f0, f, p, dp;

  f0 = HX_Fstar (PaL, PaR, p0);
  if (f0 != f0 || PaL->fail) switch_to_hll = 1;
  p = p0;
  if (fabs(f0) > 1.e-12 && !switch_to_hll){
    p  = probe*p0; f  = f0;
    for (k = 1; k < HLLD_MAX_ITER; k++){
      f  = HX_Fstar (PaL, PaR, p);
      if (f != f  || PaL->fail || (k > 7) || (fabs(f) > fabs(f0) && k > 4)) {
        switch_to_hll = 1;
        break;
      }
      dp = (p - p0)/(f - f0)*f;
      p0 = p; f0 = f;
      p -= dp;
      if (p < 0.0) p = 1.e-6;
      if (fabs(dp) < 1.e-5*p || fabs(f) < 1.e-6) break;
    }
  }
  if (PaL->fail) switch_to_hll = 1;
  *kout = k;
  if (switch_to_hll) return 0;
  f = HX_Fstar (PaL, PaR, p);            /* states consistent with the final p */
  if (f != f || PaL->fail) return 0;
  *p_out = p;
  return 1;
}

/* Physical checks hlld.c does not make on an accepted root: the states behind the fast
   and rotational fronts must be subluminal with a positive nominal gas pressure
   p_T - b^2/2 and positive density (HLLD's fans fail these on most strongly magnetised
   faces, see the task notes). */
static int hx_hlld_physical (HX_Riemann_State *PaL, HX_Riemann_State *PaR, double p)
{
  int side;
  for (side = 0; side < 2; side++){
    HX_Riemann_State *Pa = side ? PaR : PaL;
    double v2 = Pa->vx*Pa->vx + Pa->vy*Pa->vy + Pa->vz*Pa->vz;
    double vb = Pa->vx*hx_Bx + Pa->vy*Pa->By + Pa->vz*Pa->Bz;
    double bs = (hx_Bx*hx_Bx + Pa->By*Pa->By + Pa->Bz*Pa->Bz)*(1.0 - v2) + vb*vb;
    if (!(v2 < 1.0)) return 0;
    if (!(p - 0.5*bs > 0.0)) return 0;
    if (!(Pa->R[RHO]/(Pa->S - Pa->vx) > 0.0)) return 0;
  }
  return 1;
}

/* The HLLD stage of HLLX: hlld.c's secant with its own probe; if that fails at the probe
   itself (k <= 1), once more with a probe of 1 + 1e-4 (at low beta a 2.5 % change of the
   total pressure is a large change of the state and hlld.c gives up on nearly uniform
   faces); then the physical checks above. */
static int hx_hlld_root (HX_Riemann_State *PaL, HX_Riemann_State *PaR, double p0, double *p_out)
{
  int k;
  HX_Riemann_State L0 = *PaL, R0 = *PaR;
  if (hx_hlld_secant (PaL, PaR, p0, 1.025, p_out, &k))
    return hx_hlld_physical (PaL, PaR, *p_out);
  if (k > 1) return 0;
  *PaL = L0; *PaR = R0;
  if (hx_hlld_secant (PaL, PaR, p0, 1.0 + 1.e-4, p_out, &k))
    return hx_hlld_physical (PaL, PaR, *p_out);
  return 0;
}

static double HX_TotalPressure (double *v)
{
  double vel2, Bmag2, vB;

  Bmag2 = v[BX1]*v[BX1] + v[BX2]*v[BX2] + v[BX3]*v[BX3];
  vel2  = v[VX1]*v[VX1] + v[VX2]*v[VX2] + v[VX3]*v[VX3];
  vB    = v[VX1]*v[BX1] + v[VX2]*v[BX2] + v[VX3]*v[BX3];
  return v[PRS] + 0.5*(Bmag2*(1.0 - vel2) + vB*vB);
}

/* ===================================================================
   2. The seven-wave fan in the task layout.
   =================================================================== */

typedef struct {
  double UL[7], UR[7], FL[7], FR[7], RL[7], RR[7];
  double SfL, SfR, SsL, SsR, bx;
  double w;   /* SPECTRE-MOD: the face speed the fan is sampled at (0 for PLUTO's own calls) */
} HX_Ctx;

typedef struct {
  double XfL[8], XaL[8], XsL[8], XsR[8], XaR[8], XfR[8];
  double SaL, SaR, Sm;
  double res[7];
} HX_Fan;

static void hx_auxU (const double *X, double bx, double *U)
{
  double v0 = X[1], v1 = X[2], v2 = X[3], By = X[4], Bz = X[5], wT = X[6], pT = X[7];
  double w2 = 1.0/(1.0 - (v0*v0 + v1*v1 + v2*v2));
  double vb = v0*bx + v1*By + v2*Bz;
  U[0] = X[0];
  U[1] = wT*w2*v0 - (vb*bx + w2*vb*vb*v0);
  U[2] = wT*w2*v1 - (vb*By + w2*vb*vb*v1);
  U[3] = wT*w2*v2 - (vb*Bz + w2*vb*vb*v2);
  U[4] = wT*w2 - pT - w2*vb*vb;
  U[5] = By;
  U[6] = Bz;
}

static void hx_auxF (const double *X, double bx, double *F)
{
  double v0 = X[1], v1 = X[2], v2 = X[3], By = X[4], Bz = X[5], wT = X[6], pT = X[7];
  double w2 = 1.0/(1.0 - (v0*v0 + v1*v1 + v2*v2));
  double vb = v0*bx + v1*By + v2*Bz;
  double bk0 = bx*bx/w2 + vb*(bx*v0 + bx*v0) + w2*vb*vb*v0*v0;
  double bk1 = By*bx/w2 + vb*(By*v0 + bx*v1) + w2*vb*vb*v1*v0;
  double bk2 = Bz*bx/w2 + vb*(Bz*v0 + bx*v2) + w2*vb*vb*v2*v0;
  F[0] = X[0]*v0;
  F[1] = wT*w2*v0*v0 - bk0 + pT;
  F[2] = wT*w2*v1*v0 - bk1;
  F[3] = wT*w2*v2*v0 - bk2;
  F[4] = wT*w2*v0 - (vb*bx + w2*vb*vb*v0);
  F[5] = By*v0 - bx*v1;
  F[6] = Bz*v0 - bx*v2;
}

/* MUB09's jump map: downstream auxiliary tuple across a front of speed s from
   R = s U_up - F_up (task layout) and the trial downstream total pressure.  */
static void hx_jump (const double *R, double s, double bx, double pT, double *X)
{
  double RD = R[0], Rmx = R[1], Rmy = R[2], Rmz = R[3], RE = R[4], RBy = R[5], RBz = R[6];
  double aa = Rmx - s*RE + pT*(1.0 - s*s);
  double gg = RBy*RBy + RBz*RBz;
  double cc = Rmy*RBy + Rmz*RBz;
  double qq = -aa - gg + bx*bx*(1.0 - s*s);
  double xx = bx*(aa*s*bx + cc) - (aa + gg)*(s*pT + RE);
  double vx = (bx*(aa*bx + s*cc) - (aa + gg)*(pT + Rmx))/xx;
  double vy = (qq*Rmy + RBy*(cc + bx*(s*Rmx - RE)))/xx;
  double vz = (qq*Rmz + RBz*(cc + bx*(s*Rmx - RE)))/xx;
  X[0] = RD/(s - vx);
  X[1] = vx;  X[2] = vy;  X[3] = vz;
  X[4] = (RBy - bx*vy)/(s - vx);
  X[5] = (RBz - bx*vz)/(s - vx);
  X[6] = pT + (RE - (vx*Rmx + vy*Rmy + vz*Rmz))/(s - vx);
  X[7] = pT;
}

static void hx_kvec (const double *X, double bx, double zeta, double *K)
{
  double v0 = X[1], v1 = X[2], v2 = X[3];
  double w2 = 1.0/(1.0 - (v0*v0 + v1*v1 + v2*v2));
  double vb = v0*bx + v1*X[4] + v2*X[5];
  double den = w2*(zeta + vb);
  K[0] = v0 + bx/den;
  K[1] = v1 + X[4]/den;
  K[2] = v2 + X[5]/den;
}

static void hx_rotvel (const double *K, double zeta, const double *Bt, double bx, double *v)
{
  double K2 = K[0]*K[0] + K[1]*K[1] + K[2]*K[2];
  double KB = K[0]*bx + K[1]*Bt[0] + K[2]*Bt[1];
  double f = (1.0 - K2)/(zeta - KB);
  v[0] = K[0] - bx*f;
  v[1] = K[1] - Bt[0]*f;
  v[2] = K[2] - Bt[1]*f;
}

static double hx_ceos (const double *X, double bx)
{
  double v0 = X[1], v1 = X[2], v2 = X[3], By = X[4], Bz = X[5];
  double v2s = v0*v0 + v1*v1 + v2*v2;
  double vb = v0*bx + v1*By + v2*Bz;
  double bs = (bx*bx + By*By + Bz*Bz)*(1.0 - v2s) + vb*vb;
  return X[6] - bs - X[0]*sqrt(1.0 - v2s) - g_gamma/(g_gamma - 1.0)*(X[7] - 0.5*bs);
}

static void hx_fan (const double *z, const HX_Ctx *c, HX_Fan *f)
{
  int k;
  double sgn = c->bx >= 0.0 ? 1.0 : -1.0;
  double zL, zR, KL[3], KR[3], BtaL[2], BtaR[2], vaL[3], vaR[3], DaL, DaR;
  double UaL[7], FaL[7], UaR[7], FaR[7], RsL[7], RsR[7];

  hx_jump (c->RL, c->SfL, c->bx, z[0], f->XfL);
  hx_jump (c->RR, c->SfR, c->bx, z[1], f->XfR);
  zL = -sgn*sqrt(MAX(f->XfL[6], 0.0));
  zR =  sgn*sqrt(MAX(f->XfR[6], 0.0));
  hx_kvec (f->XfL, c->bx, zL, KL);
  hx_kvec (f->XfR, c->bx, zR, KR);
  f->SaL = KL[0];
  f->SaR = KR[0];
  BtaL[0] = z[3]; BtaL[1] = z[4];
  BtaR[0] = z[5]; BtaR[1] = z[6];
  hx_rotvel (KL, zL, BtaL, c->bx, vaL);
  hx_rotvel (KR, zR, BtaR, c->bx, vaR);
  DaL = f->XfL[0]*(f->SaL - f->XfL[1])/(f->SaL - vaL[0]);
  DaR = f->XfR[0]*(f->SaR - f->XfR[1])/(f->SaR - vaR[0]);
  f->XaL[0] = DaL; f->XaL[1] = vaL[0]; f->XaL[2] = vaL[1]; f->XaL[3] = vaL[2];
  f->XaL[4] = BtaL[0]; f->XaL[5] = BtaL[1]; f->XaL[6] = zL*zL; f->XaL[7] = z[0];
  f->XaR[0] = DaR; f->XaR[1] = vaR[0]; f->XaR[2] = vaR[1]; f->XaR[3] = vaR[2];
  f->XaR[4] = BtaR[0]; f->XaR[5] = BtaR[1]; f->XaR[6] = zR*zR; f->XaR[7] = z[1];
  hx_auxU (f->XaL, c->bx, UaL);  hx_auxF (f->XaL, c->bx, FaL);
  hx_auxU (f->XaR, c->bx, UaR);  hx_auxF (f->XaR, c->bx, FaR);
  for (k = 0; k < 7; k++){
    RsL[k] = c->SsL*UaL[k] - FaL[k];
    RsR[k] = c->SsR*UaR[k] - FaR[k];
  }
  hx_jump (RsL, c->SsL, c->bx, z[2], f->XsL);
  hx_jump (RsR, c->SsR, c->bx, z[2], f->XsR);
  for (k = 0; k < 3; k++) f->res[k] = f->XsL[1 + k] - f->XsR[1 + k];
  f->res[3] = f->XsL[4] - f->XsR[4];
  f->res[4] = f->XsL[5] - f->XsR[5];
  f->res[5] = hx_ceos (f->XsL, c->bx);
  f->res[6] = hx_ceos (f->XsR, c->bx);
  f->Sm = 0.5*(f->XsL[1] + f->XsR[1]);
}

static int hx_admissible (const HX_Fan *f, const HX_Ctx *c, const double *z)
{
  int k, j;
  const double *Xs[6] = {f->XfL, f->XaL, f->XsL, f->XsR, f->XaR, f->XfR};
  for (k = 0; k < 7; k++) if (!isfinite(z[k])) return 0;
  for (j = 0; j < 6; j++){
    const double *X = Xs[j];
    for (k = 0; k < 8; k++) if (!isfinite(X[k])) return 0;
    if (!(X[1]*X[1] + X[2]*X[2] + X[3]*X[3] < 1.0)) return 0;
    if (!(X[0] > 0.0)) return 0;
    if (!(X[6] > X[7])) return 0;
  }
  /* the fast and Alfven speeds coincide to round-off when B_t = 0 (a rotational front of
     zero width): those two orderings get a round-off slack, the others stay strict */
  if (!(f->SaL >= c->SfL - 1.e-12*(1.0 + fabs(c->SfL)))) return 0;
  if (!(c->SsL > f->SaL)) return 0;
  if (!(f->Sm > c->SsL))  return 0;
  if (!(c->SsR > f->Sm))  return 0;
  if (!(f->SaR > c->SsR)) return 0;
  if (!(c->SfR >= f->SaR - 1.e-12*(1.0 + fabs(c->SfR)))) return 0;
  return 1;
}

/* scaled residual norm; returns 0 if not finite */
static int hx_resnorm (const HX_Fan *f, const double *row, double *r, double *rn)
{
  int k;
  double s = 0.0;
  for (k = 0; k < 7; k++){
    r[k] = f->res[k]/row[k];
    s += r[k]*r[k];
  }
  *rn = sqrt(s);
  return isfinite(*rn);
}

/* Gaussian elimination with partial pivoting, A x = b, 7 x 7 (A, b destroyed) */
static int hx_lin7 (double A[7][7], double *b, double *x)
{
  int i, j, k, piv;
  double amax, t;
  for (k = 0; k < 7; k++){
    piv = k; amax = fabs(A[k][k]);
    for (i = k + 1; i < 7; i++) if (fabs(A[i][k]) > amax){ amax = fabs(A[i][k]); piv = i; }
    if (!(amax > 1.e-300)) return 1;
    if (piv != k){
      for (j = 0; j < 7; j++){ t = A[k][j]; A[k][j] = A[piv][j]; A[piv][j] = t; }
      t = b[k]; b[k] = b[piv]; b[piv] = t;
    }
    for (i = k + 1; i < 7; i++){
      t = A[i][k]/A[k][k];
      for (j = k; j < 7; j++) A[i][j] -= t*A[k][j];
      b[i] -= t*b[k];
    }
  }
  for (i = 6; i >= 0; i--){
    t = b[i];
    for (j = i + 1; j < 7; j++) t -= A[i][j]*x[j];
    x[i] = t/A[i][i];
  }
  for (i = 0; i < 7; i++) if (!isfinite(x[i])) return 1;
  return 0;
}

/* Damped Newton with LM retries on the scaled residual.  Returns 1 if converged
   (scaled residual < HLLX_TOL at an admissible point), 0 otherwise. z is updated
   in place to the best point reached; *iters counts the Newton steps taken. */
static int hx_newton (const HX_Ctx *c, double *z, const double *row, const double *col,
                      int *iters)
{
  int it, k, j, m, accepted, singular;
  double r[7], rn, J[7][7], A[7][7], rhs[7], dz_s[7], dz[7], zn[7], rnew[7], rnn, lam;
  double zp[7], res0[7];
  HX_Fan f, fn, fp;
  const double h = 1.e-7;
  const double mu_rel[5] = {1.e-6, 1.e-4, 1.e-2, 1.0, 1.e2};

  *iters = 0;
  hx_fan (z, c, &f);
  if (!hx_resnorm (&f, row, r, &rn)) return 0;
  if (!hx_admissible (&f, c, z)) return 0;
  if (rn < HLLX_TOL) return 1;

  for (it = 0; it < HLLX_MAX_NEWTON; it++){

    /* forward-difference Jacobian of the scaled residual, scaled columns */
    for (k = 0; k < 7; k++) res0[k] = r[k]*row[k];
    for (k = 0; k < 7; k++){
      for (j = 0; j < 7; j++) zp[j] = z[j];
      zp[k] += h*col[k];
      hx_fan (zp, c, &fp);
      for (j = 0; j < 7; j++){
        J[j][k] = (fp.res[j] - res0[j])/h/row[j];
        if (!isfinite(J[j][k])) J[j][k] = 0.0;
      }
    }

    /* Newton direction */
    for (j = 0; j < 7; j++){ rhs[j] = -r[j]; for (k = 0; k < 7; k++) A[j][k] = J[j][k]; }
    singular = hx_lin7 (A, rhs, dz_s);
    accepted = 0;
    if (!singular){
      for (k = 0; k < 7; k++) dz[k] = dz_s[k]*col[k];
      lam = 1.0;
      for (m = 0; m < 12; m++){
        for (k = 0; k < 7; k++) zn[k] = z[k] + lam*dz[k];
        hx_fan (zn, c, &fn);
        if (hx_resnorm (&fn, row, rnew, &rnn) && hx_admissible (&fn, c, zn)
            && rnn < (1.0 - 1.e-4*lam)*rn){
          accepted = 1; break;
        }
        lam *= 0.5;
      }
    }

    /* Levenberg-Marquardt retries */
    if (!accepted){
      int q;
      double fro = 0.0, mu;
      for (j = 0; j < 7; j++) for (k = 0; k < 7; k++) fro += J[j][k]*J[j][k];
      for (q = 0; q < 5 && !accepted; q++){
        mu = mu_rel[q]*fro;
        for (j = 0; j < 7; j++){
          rhs[j] = 0.0;
          for (k = 0; k < 7; k++){
            double s = 0.0;
            int i;
            for (i = 0; i < 7; i++) s += J[i][j]*J[i][k];
            A[j][k] = s + (j == k ? mu : 0.0);
          }
          for (k = 0; k < 7; k++) rhs[j] -= J[k][j]*r[k];
        }
        if (hx_lin7 (A, rhs, dz_s)) continue;
        for (k = 0; k < 7; k++) dz[k] = dz_s[k]*col[k];
        lam = 1.0;
        for (m = 0; m < 8; m++){
          for (k = 0; k < 7; k++) zn[k] = z[k] + lam*dz[k];
          hx_fan (zn, c, &fn);
          if (hx_resnorm (&fn, row, rnew, &rnn) && hx_admissible (&fn, c, zn)
              && rnn < (1.0 - 1.e-4*lam)*rn){
            accepted = 1; break;
          }
          lam *= 0.5;
        }
      }
    }

    (*iters)++;
    if (!accepted) return 0;                     /* stalled */
    for (k = 0; k < 7; k++){ z[k] = zn[k]; r[k] = rnew[k]; }
    rn = rnn;
    if (rn < HLLX_TOL) return 1;
  }
  return 0;
}

/* Residual (row) and unknown (column) scales from the seed, invariant under a change of
   units: velocity rows by the fan width S_fR - S_fL, field rows and columns by
   max(|B_a|, |B_x|, sqrt p*), pressure columns by p*, EOS rows by max(p*, HLLX_EOS_FLOOR w_T)
   (the floor keeps the round-off of C_EOS, ~eps w_T, below HLLX_TOL where p* << w_T: the
   scaled floor is <= eps/HLLX_EOS_FLOOR = 2e-11; the anchors are then enforced to
   HLLX_TOL HLLX_EOS_FLOOR w_T in absolute terms, far below any flux scale). The previous scales (velocity rows
   1, field and EOS scales floored at 1, absolute HLLX_TOL = 1e-10) accepted HLLD's seed as the
   root below p* ~ 1e-8 and lost convergence as v/c -> 0 (task 2, 2026-10-06 evening); PLUTO's
   p ~ 1 tests could not show it, SpECTRE's geometric units are entirely in that regime. */
static void hx_scales (const HX_Ctx *c, const double *z0, double *row, double *col)
{
  int k;
  HX_Fan f;
  double bsc = MAX(MAX(fabs(z0[3]), fabs(z0[4])), MAX(fabs(z0[5]), fabs(z0[6])));
  double psc = fabs(z0[0]), vsc = c->SfR - c->SfL, esc;
  bsc = MAX(bsc, fabs(c->bx));
  bsc = MAX(bsc, sqrt(psc));
  hx_fan (z0, c, &f);
  esc = MAX(psc, HLLX_EOS_FLOOR*fabs(f.XsL[6]));
  for (k = 0; k < 3; k++) row[k] = vsc;
  row[3] = row[4] = bsc;
  row[5] = row[6] = esc;
  col[0] = col[1] = col[2] = psc;
  col[3] = col[4] = col[5] = col[6] = bsc;
}

static void hx_clamp_gap (double *SsL, double *SsR, double SaL, double Sc, double SaR)
{
  double a = SaL + HLLX_THETA*(Sc - SaL), b = Sc - HLLX_THETA*(Sc - SaL), lo, hi;
  lo = MIN(a, b); hi = MAX(a, b);
  *SsL = MIN(MAX(*SsL, lo), hi);
  a = Sc + HLLX_THETA*(SaR - Sc); b = SaR - HLLX_THETA*(SaR - Sc);
  lo = MIN(a, b); hi = MAX(a, b);
  *SsR = MIN(MAX(*SsR, lo), hi);
}

/* The interface flux of a converged fan, by RH recursion from the nearer side. */
static void hx_flux (const HX_Fan *f, const HX_Ctx *c, double *F)
{
  int k;
  double UfL[7], UaL[7], UsL[7], UsR[7], UaR[7], UfR[7];
  double FfL[7], FaL[7], FsL[7], FfR[7], FaR[7], FsR[7];
  const double *pick, *pickU;   /* SPECTRE-MOD: + the picked region's state */
  hx_auxU (f->XfL, c->bx, UfL); hx_auxU (f->XaL, c->bx, UaL); hx_auxU (f->XsL, c->bx, UsL);
  hx_auxU (f->XsR, c->bx, UsR); hx_auxU (f->XaR, c->bx, UaR); hx_auxU (f->XfR, c->bx, UfR);
  for (k = 0; k < 7; k++){
    FfL[k] = c->FL[k] + c->SfL*(UfL[k] - c->UL[k]);
    FaL[k] = FfL[k] + f->SaL*(UaL[k] - UfL[k]);
    FsL[k] = FaL[k] + c->SsL*(UsL[k] - UaL[k]);
    FfR[k] = c->FR[k] + c->SfR*(UfR[k] - c->UR[k]);
    FaR[k] = FfR[k] + f->SaR*(UaR[k] - UfR[k]);
    FsR[k] = FaR[k] + c->SsR*(UsR[k] - UaR[k]);
  }
  /* SPECTRE-MOD: the region tests compare against the face speed c->w instead of 0, and the
     flux through the moving face is F - w U of the picked region (spectre_face_speed.h). */
  if      (c->SfL >= c->w) { pick = c->FL; pickU = c->UL; }
  else if (c->SfR <= c->w) { pick = c->FR; pickU = c->UR; }
  else if (f->SaL >= c->w) { pick = FfL;   pickU = UfL;   }
  else if (f->SaR <= c->w) { pick = FfR;   pickU = UfR;   }
  else if (c->SsL >= c->w) { pick = FaL;   pickU = UaL;   }
  else if (c->SsR <= c->w) { pick = FaR;   pickU = UaR;   }
  else if (f->Sm  >= c->w) { pick = FsL;   pickU = UsL;   }
  else                     { pick = FsR;   pickU = UsR;   }
  for (k = 0; k < 7; k++) F[k] = pick[k];
  if (c->w != 0.0) for (k = 0; k < 7; k++) F[k] -= c->w*pickU[k];
}

/* The seven-wave solve for one interface. Returns HX_HLLX, HX_HLLX_CONT or
   HX_HLLD (failed; nothing written to F). */
static int hx_solve_face (HX_Ctx *c, const double *z0, double SaL, double Sc, double SaR,
                          double *F)
{
  int k, s, iters, status = HX_HLLD;
  double row[7], col[7], z[7], tL = c->SsL, tR = c->SsR, sL0, sR0, t;
  double bestz[7], bestL = 0.0, bestR = 0.0;
  int have = 0;
  HX_Fan f;

  hx_scales (c, z0, row, col);
  for (k = 0; k < 7; k++) z[k] = z0[k];
  if (hx_newton (c, z, row, col, &iters)){
    hx_newton_iters += iters;
    hx_fan (z, c, &f);
    hx_flux (&f, c, F);
    return HX_HLLX;
  }
  hx_newton_iters += iters;

  /* continuation in the slow speeds from the middle of their intervals */
  sL0 = SaL + 0.5*(Sc - SaL);
  sR0 = SaR - 0.5*(SaR - Sc);
  for (k = 0; k < 7; k++) z[k] = z0[k];
  for (s = 0; s <= HLLX_CONT_STEPS; s++){
    t = (double)s/HLLX_CONT_STEPS;
    c->SsL = (1.0 - t)*sL0 + t*tL;
    c->SsR = (1.0 - t)*sR0 + t*tR;
    if (!hx_newton (c, z, row, col, &iters)){ hx_newton_iters += iters; break; }
    hx_newton_iters += iters;
    for (k = 0; k < 7; k++) bestz[k] = z[k];
    bestL = c->SsL; bestR = c->SsR; have = 1;
    status = (s == HLLX_CONT_STEPS) ? HX_HLLX : HX_HLLX_CONT;
  }
  if (!have){ c->SsL = tL; c->SsR = tR; return HX_HLLD; }
  c->SsL = bestL; c->SsR = bestR;
  hx_fan (bestz, c, &f);
  hx_flux (&f, c, F);
  return status;
}

void HLLX_Stats (long *out)
{
  int k;
  for (k = 0; k < HX_NSTAT; k++) out[k] = hx_stat[k];
  out[HX_NSTAT] = hx_newton_iters;
}

/* ===================================================================
   2b. The field-aligned branch (2026-10-26): ZPL26 section 4.2.2, the case c_f ~ c_a with
   c_s < c_a, in the relativistic variables. With v_t = B_t = 0 the longitudinal subsystem
   (D, m_n, E - B_n^2/2) is exactly relativistic hydrodynamics -- b^2 = B_n^2 and the
   magnetic terms are constants that drop out of every jump condition -- so the fan between
   the (coincident) fast-Alfven waves is Mignone & Bodo 2005's HLLC with the SLOW (acoustic)
   speeds as its outer speeds: closed form, no Newton. The seven-wave construction cannot
   form there (its unknowns are the inner states' transverse fields, 0 = 0 at B_t = 0), and
   HLLD's own fan degenerates (its a-states have zero width): on such faces the solver used
   to keep HLLD's flux. Transverse momenta and fields keep the HLL flux over the fast fan
   (they are Alfvenic and couple to the longitudinal system at O(B_t^2)); B_n's flux is zero;
   everything is sampled at x/t = w (0 in PLUTO). Design and derivation:
   notes/projects/active/hllx_rmhd/bt0_branch_design.md; numpy reference
   scripts/rmhd_py/hllx_rmhd.py (bt0_face), validated against an independent relativistic
   hydro HLLC to 1e-13 and against the converged seven-wave flux at B_t/B_n = 1e-2 to 1e-5.
   =================================================================== */
#ifndef HX_EPS_BT
  #define HX_EPS_BT 1.0e-2            /* |B_t| <= HX_EPS_BT |B_n| on both sides takes the branch */
#endif
#ifndef HX_EPS_BT_FALLBACK
  #define HX_EPS_BT_FALLBACK 5.0e-2   /* a nearly aligned face whose HLLD root or seven-wave Newton fails takes it too */
#endif
#ifndef HX_EPS_BTP
  #define HX_EPS_BTP 1.0e-2           /* and the transverse magnetic pressure B_t^2/2 must be below HX_EPS_BTP of the gas
                                 pressure on both sides: the hydro reduction drops B_t^2 terms, which are small
                                 against B_n^2 but need not be against p (the cold large-B_x tube: B_t/B_n = 0.02,
                                 B_t^2/2p = 0.6; taking the branch there as a fallback wrecked the run, 2026-10-26) */
#endif

static int hx_bt0_gate (const double *vL, const double *vR, double a2L, double hL,
                        double a2R, double hR, double bx, double eps)
{
  double bn = fabs(bx), bt, va2;
  if (!(bn > 0.0)) return 0;
  bt = sqrt(vL[BXt]*vL[BXt] + vL[BXb]*vL[BXb]); if (bt > eps*bn || 0.5*bt*bt > HX_EPS_BTP*vL[PRS]) return 0;
  bt = sqrt(vR[BXt]*vR[BXt] + vR[BXb]*vR[BXb]); if (bt > eps*bn || 0.5*bt*bt > HX_EPS_BTP*vR[PRS]) return 0;
  va2 = bn*bn/(vL[RHO]*hL + bn*bn); if (!(va2 > a2L)) return 0;   /* fast = Alfven needs v_A > c_s */
  va2 = bn*bn/(vR[RHO]*hR + bn*bn); if (!(va2 > a2R)) return 0;
  return 1;
}

static void hx_bt0_face (const double *vL, const double *vR,
                         const double *uL, const double *uR,
                         const double *fL, const double *fR,
                         double pTL, double pTR, double SL, double SR, double bx,
                         double a2L, double a2R, double w, const double *Fhll_i,
                         double *flux, double *press)
{
  const double bconst = 0.5*bx*bx;
  const double csL = sqrt(a2L), csR = sqrt(a2R), vnL = vL[VXn], vnR = vR[VXn];
  const double SsL = MIN((vnL - csL)/(1.0 - vnL*csL), (vnR - csR)/(1.0 - vnR*csR));
  const double SsR = MAX((vnL + csL)/(1.0 + vnL*csL), (vnR + csR)/(1.0 + vnR*csR));
  double UL[3], UR[3], FL[3], FR[3], Uh[3], Fh[3], RL[3], RR[3], Us[3], Fl[3];
  double dS, a, b, c, lam, disc, ps, pT, scrh, uh;
  int k, nv;

  /* longitudinal hydro variables: D, m_n, the TOTAL energy minus the magnetic constant; hydro fluxes */
  UL[0] = uL[RHO]; UL[1] = uL[MXn]; UL[2] = uL[ENG] + uL[RHO] - bconst;
  UR[0] = uR[RHO]; UR[1] = uR[MXn]; UR[2] = uR[ENG] + uR[RHO] - bconst;
  FL[0] = fL[RHO]; FL[1] = fL[MXn] + pTL + bconst; FL[2] = fL[ENG] + fL[RHO];
  FR[0] = fR[RHO]; FR[1] = fR[MXn] + pTR + bconst; FR[2] = fR[ENG] + fR[RHO];

  dS = 1.0/(SsR - SsL);
  for (k = 0; k < 3; k++){
    Uh[k] = (SsR*UR[k] - SsL*UL[k] + FL[k] - FR[k])*dS;
    Fh[k] = (SsR*FL[k] - SsL*FR[k] + SsL*SsR*(UR[k] - UL[k]))*dS;
    RL[k] = SsL*UL[k] - FL[k];
    RR[k] = SsR*UR[k] - FR[k];
  }
  /* Mignone & Bodo 2005: F_E^hll lam^2 - (E^hll + F_m^hll) lam + m^hll = 0, the subluminal root */
  a = Fh[2]; b = -(Uh[2] + Fh[1]); c = Uh[1];
  if (fabs(a) < 1.e-12*MAX(fabs(b), 1.e-300)){
    lam = -c/b;
  }else{
    disc = b*b - 4.0*a*c; disc = MAX(disc, 0.0);
    lam = (-b - sqrt(disc))/(2.0*a);
  }
  ps = 0.5*((RL[1] - RL[2]*lam)/(SsL*lam - 1.0) + (RR[1] - RR[2]*lam)/(SsR*lam - 1.0));

  if (SsL >= w){
    for (k = 0; k < 3; k++){ Us[k] = UL[k]; Fl[k] = FL[k]; }
    pT = pTL;
  }else if (lam >= w){
    Us[0] = RL[0]/(SsL - lam); Us[1] = (RL[1] + ps)/(SsL - lam); Us[2] = (RL[2] + Us[1])/SsL;
    for (k = 0; k < 3; k++) Fl[k] = FL[k] + SsL*(Us[k] - UL[k]);
    pT = ps + bconst;
  }else if (SsR > w){
    Us[0] = RR[0]/(SsR - lam); Us[1] = (RR[1] + ps)/(SsR - lam); Us[2] = (RR[2] + Us[1])/SsR;
    for (k = 0; k < 3; k++) Fl[k] = FR[k] + SsR*(Us[k] - UR[k]);
    pT = ps + bconst;
  }else{
    for (k = 0; k < 3; k++){ Us[k] = UR[k]; Fl[k] = FR[k]; }
    pT = pTR;
  }

  /* transverse components (and GLM's psi if present): HLL over the fast fan, sampled at w */
  scrh = 1.0/(SR - SL);
  for (nv = 0; nv < NFLX; nv++){
    if (nv == RHO || nv == MXn || nv == ENG || nv == BXn) continue;
    uh = (SR*uR[nv] - SL*uL[nv] + fL[nv] - fR[nv])*scrh;
    flux[nv] = Fhll_i[nv] - w*uh;
  }
  /* PLUTO's layout: total pressure outside the normal momentum flux, reduced energy; magnetic constant restored */
  flux[RHO] = Fl[0] - w*Us[0];
  flux[MXn] = Fl[1] - bconst - pT - w*Us[1];
  flux[ENG] = Fl[2] - Fl[0] - w*(Us[2] + bconst - Us[0]);
  flux[BXn] = -w*bx;
  *press = pT;
}

/* ===================================================================
   3. The solver: hlld.c's loop, with the seven-wave upgrade.
   =================================================================== */
void HLLX_Solver (const Sweep *sweep, int beg, int end,
                  double *cmax, Grid *grid)
{
  int    nv, i, k;
  int    switch_to_hll, quadratic_guess;
  double scrh;
  static _Thread_local double **Uhll, **Fhll, **Vhll;   /* SPECTRE-MOD */
  static _Thread_local double *lamL, *lamR;             /* SPECTRE-MOD */
  double *vL, *vR, *fL, *fR, *uL, *uR, *SL, *SR;
  double p0, p, dS_1;
  double Uc[NVAR];
  HX_Riemann_State PaL, PaR;
  const State   *stateL = &(sweep->stateL);
  const State   *stateR = &(sweep->stateR);
  double  *pL = stateL->prs,   *pR = stateR->prs;

  if (Uhll == NULL){
    Uhll = ARRAY_2D(NMAX_POINT, NVAR, double);
    Fhll = ARRAY_2D(NMAX_POINT, NVAR, double);
    Vhll = ARRAY_2D(NMAX_POINT, NVAR, double);
    lamL = ARRAY_1D(NFLX, double);
    lamR = ARRAY_1D(NFLX, double);
  }

#ifdef GLM_MHD
  GLM_Solve (sweep, beg, end, grid);
#endif

  SoundSpeed2 (stateL, beg, end, FACE_CENTER, grid);
  SoundSpeed2 (stateR, beg, end, FACE_CENTER, grid);

  Flux (stateL, beg, end);
  Flux (stateR, beg, end);

  SL = sweep->SL; SR = sweep->SR;
  HLL_Speed (stateL, stateR, SL, SR, beg, end);

  hx_calls++;

  for (i = beg; i <= end; i++) {

    scrh  = MAX(fabs(SL[i]), fabs(SR[i]));
    cmax[i] = scrh;

    vL = stateL->v[i]; uL = stateL->u[i]; fL = stateL->flux[i];
    vR = stateR->v[i]; uR = stateR->u[i]; fR = stateR->flux[i];
    /* SPECTRE-MOD: the fan is sampled on the ray of the (moving) face; 0 for PLUTO's own calls. */
    const double wf = SpectreFaceSpeed (sweep, i);

    if (SL[i] >= wf){   /* SPECTRE-MOD: 0 -> wf */
      NFLX_LOOP(nv) sweep->flux[i][nv] = fL[nv];
      sweep->press[i] = pL[i];
      if (wf != 0.0) SpectreMovingFace (sweep->flux[i], wf, uL);   /* SPECTRE-MOD */
      hx_stat[HX_SUPERSONIC]++;
      continue;
    }
    if (SR[i] <= wf){   /* SPECTRE-MOD: 0 -> wf */
      NFLX_LOOP(nv) sweep->flux[i][nv] = fR[nv];
      sweep->press[i] = pR[i];
      if (wf != 0.0) SpectreMovingFace (sweep->flux[i], wf, uR);   /* SPECTRE-MOD */
      hx_stat[HX_SUPERSONIC]++;
      continue;
    }

    dS_1 = 1.0/(SR[i] - SL[i]);
    NFLX_LOOP(nv){
      Uhll[i][nv]  = SR[i]*uR[nv] - SL[i]*uL[nv]  + fL[nv] - fR[nv];
      Uhll[i][nv] *= dS_1;
      Fhll[i][nv]  = SR[i]*fL[nv] - SL[i]*fR[nv] + SL[i]*SR[i]*(uR[nv] - uL[nv]);
      Fhll[i][nv] *= dS_1;
    }
    Uhll[i][MXn] += (pL[i] - pR[i])*dS_1;

    /* field-aligned faces: the closed-form branch (section 2b) */
    if (hx_bt0_gate (vL, vR, stateL->a2[i], stateL->h[i], stateR->a2[i], stateR->h[i], Uhll[i][BXn], HX_EPS_BT)){
      hx_bt0_face (vL, vR, uL, uR, fL, fR, pL[i], pR[i], SL[i], SR[i], Uhll[i][BXn],
                   stateL->a2[i], stateR->a2[i], wf, Fhll[i], sweep->flux[i], &sweep->press[i]);
      hx_stat[HX_BT0]++;
      continue;
    }

    memset (&PaL, 0, sizeof PaL);      /* deterministic: hlld.c leaves these on the stack */
    memset (&PaR, 0, sizeof PaR);
    PaL.S = SL[i];
    PaR.S = SR[i];
    hx_Bx = Uhll[i][BXn];
    NFLX_LOOP(nv){
      PaL.R[nv] = SL[i]*uL[nv] - fL[nv];
      PaR.R[nv] = SR[i]*uR[nv] - fR[nv];
    }
    PaL.R[MXn] -= pL[i];
    PaR.R[MXn] -= pR[i];
    #if RMHD_REDUCED_ENERGY == YES
    PaL.R[ENG] += PaL.R[RHO];
    PaR.R[ENG] += PaR.R[RHO];
    #endif

    scrh = MAX(pL[i], pR[i]);
    quadratic_guess = 0;
    if (hx_Bx*hx_Bx/scrh < 0.01) {
      double a,b,c;
      quadratic_guess = 1;
      a = SR[i] - SL[i];
      b = PaR.R[ENG] - PaL.R[ENG] + SR[i]*PaL.R[MXn] - SL[i]*PaR.R[MXn];
      c = PaL.R[MXn]*PaR.R[ENG] - PaR.R[MXn]*PaL.R[ENG];
      scrh = b*b - 4.0*a*c;
      scrh = MAX(scrh,0.0);
      p0 = 0.5*(- b + sqrt(scrh))*dS_1;
    }else{
      ConsToPrim(Uhll, Vhll, i, i, sweep->flag);
      p0    = HX_TotalPressure(Vhll[i]);
    }

    switch_to_hll = !hx_hlld_root (&PaL, &PaR, p0, &p);
    k = 0;

    if (switch_to_hll){
      if (hx_bt0_gate (vL, vR, stateL->a2[i], stateL->h[i], stateR->a2[i], stateR->h[i], hx_Bx, HX_EPS_BT_FALLBACK)){
        hx_bt0_face (vL, vR, uL, uR, fL, fR, pL[i], pR[i], SL[i], SR[i], hx_Bx,
                     stateL->a2[i], stateR->a2[i], wf, Fhll[i], sweep->flux[i], &sweep->press[i]);
        hx_stat[HX_BT0F]++;
        continue;
      }
      NFLX_LOOP(nv) sweep->flux[i][nv] = Fhll[i][nv];
      sweep->press[i]  = (SR[i]*pL[i] - SL[i]*pR[i])*dS_1;
      if (wf != 0.0) SpectreMovingFaceHll (sweep->flux[i], wf, uL, uR, fL, fR,   /* SPECTRE-MOD */
                                           pL[i], pR[i], SL[i], SR[i]);
      hx_stat[HX_HLL]++;
      continue;
    }

    g_maxRiemannIter = MAX(g_maxRiemannIter, k);

    /* ---- HLLD's flux, exactly as hlld.c assembles it ---- */
    {
      double SaL = PaL.Kx, SaR = PaR.Kx, Sc = hx_Sc;   /* before GetAState etc. */
      double Byc, Bzc, dK;

      if (PaL.Sa >= wf - 1.e-6){   /* SPECTRE-MOD: 0 -> wf */
        HX_GetAState (&PaL, p);
        #if RMHD_REDUCED_ENERGY == YES
        PaL.u[ENG] -= PaL.u[RHO];
        #endif
        NFLX_LOOP(nv) sweep->flux[i][nv] = fL[nv] + SL[i]*(PaL.u[nv] - uL[nv]);
        sweep->press[i] = pL[i];
        if (wf != 0.0) SpectreMovingFace (sweep->flux[i], wf, PaL.u);   /* SPECTRE-MOD */
      }else if (PaR.Sa <= wf + 1.e-6){   /* SPECTRE-MOD: 0 -> wf */
        HX_GetAState (&PaR, p);
        #if RMHD_REDUCED_ENERGY == YES
        PaR.u[ENG] -= PaR.u[RHO];
        #endif
        NFLX_LOOP(nv) sweep->flux[i][nv] = fR[nv] + SR[i]*(PaR.u[nv] - uR[nv]);
        sweep->press[i] = pR[i];
        if (wf != 0.0) SpectreMovingFace (sweep->flux[i], wf, PaR.u);   /* SPECTRE-MOD */
      }else{
        HX_GetCState (&PaL, &PaR, p, Uc, wf);   /* SPECTRE-MOD: + wf */
        if (hx_Sc > wf){   /* SPECTRE-MOD: 0 -> wf */
          #if RMHD_REDUCED_ENERGY == YES
          PaL.u[ENG] -= PaL.u[RHO];
          Uc[ENG]    -= Uc[RHO];
          #endif
          NFLX_LOOP(nv) sweep->flux[i][nv] = fL[nv] + SL[i]*(PaL.u[nv] - uL[nv])
                                                    + PaL.Sa*(Uc[nv] - PaL.u[nv]);
          sweep->press[i] = pL[i];
          if (wf != 0.0) SpectreMovingFace (sweep->flux[i], wf, Uc);   /* SPECTRE-MOD */
        }else{
          #if RMHD_REDUCED_ENERGY == YES
          PaR.u[ENG] -= PaR.u[RHO];
          Uc[ENG]    -= Uc[RHO];
          #endif
          NFLX_LOOP(nv) sweep->flux[i][nv] = fR[nv] + SR[i]*(PaR.u[nv] - uR[nv])
                                                    + PaR.Sa*(Uc[nv] - PaR.u[nv]);
          sweep->press[i] = pR[i];
          if (wf != 0.0) SpectreMovingFace (sweep->flux[i], wf, Uc);   /* SPECTRE-MOD */
        }
      }

      if (quadratic_guess){ hx_stat[HX_BX0]++; continue; }
      if (!hx_seven_waves){ hx_stat[HX_HLLX]++; continue; }   /* the hlldr solver stops here */

      /* ---- the seven-wave upgrade ---- */
      {
        HX_Ctx c;
        double z0[7], F[7], SsL, SsR;
        int status;

        /* contact field from the a-states (= GetCState's B_c) */
        dK  = (PaR.Kx - PaL.Kx) + 1.e-12;
        Byc = (PaR.By*(PaR.Kx - PaR.vx) - PaL.By*(PaL.Kx - PaL.vx) + hx_Bx*(PaR.vy - PaL.vy))/dK;
        Bzc = (PaR.Bz*(PaR.Kx - PaR.vx) - PaL.Bz*(PaL.Kx - PaL.vx) + hx_Bx*(PaR.vz - PaL.vz))/dK;

        c.bx = hx_Bx;
        c.SfL = SL[i]; c.SfR = SR[i];
        c.w = wf;   /* SPECTRE-MOD */
        c.UL[0] = uL[RHO]; c.UL[1] = uL[MXn]; c.UL[2] = uL[MXt]; c.UL[3] = uL[MXb];
        c.UL[4] = uL[ENG] + uL[RHO]; c.UL[5] = uL[BXt]; c.UL[6] = uL[BXb];
        c.UR[0] = uR[RHO]; c.UR[1] = uR[MXn]; c.UR[2] = uR[MXt]; c.UR[3] = uR[MXb];
        c.UR[4] = uR[ENG] + uR[RHO]; c.UR[5] = uR[BXt]; c.UR[6] = uR[BXb];
        c.FL[0] = fL[RHO]; c.FL[1] = fL[MXn] + pL[i]; c.FL[2] = fL[MXt]; c.FL[3] = fL[MXb];
        c.FL[4] = fL[ENG] + fL[RHO]; c.FL[5] = fL[BXt]; c.FL[6] = fL[BXb];
        c.FR[0] = fR[RHO]; c.FR[1] = fR[MXn] + pR[i]; c.FR[2] = fR[MXt]; c.FR[3] = fR[MXb];
        c.FR[4] = fR[ENG] + fR[RHO]; c.FR[5] = fR[BXt]; c.FR[6] = fR[BXb];
        for (k = 0; k < 7; k++){
          c.RL[k] = c.SfL*c.UL[k] - c.FL[k];
          c.RR[k] = c.SfR*c.UR[k] - c.FR[k];
        }
        /* slow speeds: the mean of the input states' slow eigenvalues */
        /* SPECTRE-MOD (2026-10-21): where either side's eigenvalues cannot be
           computed (MaxSignalSpeed then gave the face the light-cone bounds), keep
           HLLD's flux instead of seeding the slow speeds from the previous
           interface's lamL/lamR. */
        if (Magnetosonic (vL, stateL->a2[i], stateL->h[i], lamL) != 0 ||
            Magnetosonic (vR, stateR->a2[i], stateR->h[i], lamR) != 0) {
          hx_stat[HX_HLLD]++;
          continue;
        }
        SsL = 0.5*(lamL[KSLOWM] + lamR[KSLOWM]);
        SsR = 0.5*(lamL[KSLOWP] + lamR[KSLOWP]);
        hx_clamp_gap (&SsL, &SsR, SaL, Sc, SaR);
        c.SsL = SsL; c.SsR = SsR;

        z0[0] = z0[1] = z0[2] = p;
        z0[3] = Byc; z0[4] = Bzc; z0[5] = Byc; z0[6] = Bzc;

        status = hx_solve_face (&c, z0, SaL, Sc, SaR, F);
        hx_stat[status]++;
        if (status == HX_HLLD){                      /* HLLD's flux stays, unless the face is nearly aligned */
          if (hx_bt0_gate (vL, vR, stateL->a2[i], stateL->h[i], stateR->a2[i], stateR->h[i], hx_Bx, HX_EPS_BT_FALLBACK)){
            hx_bt0_face (vL, vR, uL, uR, fL, fR, pL[i], pR[i], SL[i], SR[i], hx_Bx,
                         stateL->a2[i], stateR->a2[i], wf, Fhll[i], sweep->flux[i], &sweep->press[i]);
            hx_stat[HX_HLLD]--; hx_stat[HX_BT0F]++;
          }
          continue;
        }

        /* back to PLUTO's layout: pressure apart, reduced energy */
        sweep->flux[i][RHO] = F[0];
        sweep->flux[i][MXn] = F[1] - sweep->press[i];
        sweep->flux[i][MXt] = F[2];
        sweep->flux[i][MXb] = F[3];
        sweep->flux[i][BXn] = 0.0;
        if (wf != 0.0) sweep->flux[i][BXn] -= wf*hx_Bx;   /* SPECTRE-MOD: as in every other region */
        sweep->flux[i][BXt] = F[5];
        sweep->flux[i][BXb] = F[6];
        #if RMHD_REDUCED_ENERGY == YES
        sweep->flux[i][ENG] = F[4] - F[0];
        #else
        sweep->flux[i][ENG] = F[4];
        #endif
      }
    }
  }

  #if HLLX_STATS_EVERY > 0
  if (hx_calls % HLLX_STATS_EVERY == 0){
    printLog ("> HLLX stats after %ld calls: supersonic %ld, hll %ld, bx0 %ld, hllx %ld, hllx-cont %ld, hlld %ld, bt0 %ld, bt0-fallback %ld; Newton iterations %ld\n",
              hx_calls, hx_stat[0], hx_stat[1], hx_stat[2], hx_stat[3], hx_stat[4], hx_stat[5], hx_stat[HX_BT0], hx_stat[HX_BT0F], hx_newton_iters);
  }
  #endif

#if DIVB_CONTROL == CONSTRAINED_TRANSPORT
  CT_Flux (sweep, beg, end, grid);
#endif
#if DIVB_CONTROL == EIGHT_WAVES
  HLL_DIVB_SOURCE (sweep, Uhll, beg + 1, end, grid);
#endif
}


/* HLLD with the converged total-pressure root and consistent states, no slow fronts:
   the base of HLLX_Solver on its own, for isolating the effect of the root finder. */
void HLLDR_Solver (const Sweep *sweep, int beg, int end, double *cmax, Grid *grid)
{
  hx_seven_waves = 0;
  HLLX_Solver (sweep, beg, end, cmax, grid);
  hx_seven_waves = 1;
}
