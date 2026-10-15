/* Independent refined fixed-bound HLLD for SR D2-R, v5.
 * Original native guess/selected secant path preserved. Final-state refresh
 * and bisection within its own pressure-history bracket remove hidden root
 * termination error. Unavailable refinement explicitly returns HLL.
 * Raw native hlld.c is a separate, byte-preserved control. */
/* ///////////////////////////////////////////////////////////////////// */
/*!
  \file
  \brief HLLD Riemann solver for the relativistic MHD equations.

  Solve the Riemann problem for the relativistic MHD equations 
  using the HLLD solver of Mignone, Ugliano & Bodo (2009).
  The solver requires the solution of a nonlinear equation (Eq. [48]) 
  and the maximum number of iterations before convergence can be 
  set using the macro HLLD_MAX_ITER (default 20).
  A physical relevant solution is accepted if it satisfies
  the constraints by Eq. [54], implemented in the function D2R_HLLD_Fstar().
  If this is not the case or if other failure conditions occur during 
  the iteration loop, we switch to the HLL Riemann solver.
  
  The macro ::COUNT_FAILURES can be set to YES if one wishes to count
  how many times the solver fails.
  
  On input, it takes left and right primitive sweep vectors 
  \c sweep->vL and \c sweep->vR at zone edge i+1/2;
  On output, return flux and pressure vectors at the same interface 
  \c i+1/2 (note that the \c i refers to \c i+1/2).
  
  Also during this step, compute maximum wave propagation speed (cmax) 
  for  explicit time step computation.
   
  \b Reference:
    - "A five wave Harte-Lax-van Leer Riemann solver for relativistic
       magnetohydrodynamics" Mignone et al, MNRAS (2009) 393,1141.

  \authors A. Mignone (andrea.mignone@unito.it)
  \date    Sep 28, 2021
*/
/* ///////////////////////////////////////////////////////////////////// */
#include"d2r_sr_face.h"
#include <string.h>
#if PHYSICS != RMHD || EOS != IDEAL || DIVB_CONTROL != NO || NSCL != 0
#error "D2R fixed HLLD v1 requires ideal RMHD with DIVB_CONTROL=NO"
#endif
/* SPECTRE-MOD: per-thread storage; fixed-v5 numerical algebra unchanged. */
static _Thread_local int glm_pair_on = -1;   /* GLM_PAIR switch, see the end of the solver */

#ifndef HLLD_MAX_ITER 
  #define HLLD_MAX_ITER  20
#endif

typedef struct RIEMANN_STATE{
  int    fail;
  double vx, vy, vz;
  double Bx, By, Bz;
  double Kx, Ky, Kz, K2;
  double w, sw, p, rho;
  double u[NFLX], R[NVAR];
  double S;    /* Fast magnetosonic speed */
  double Sa;   /* Alfven speed            */
/*
  double fun1, fun2, fun3; 
  double denL, denR;
*/
} D2R_Riemann_State;


static _Thread_local double Sc, Bx;
static double D2R_HLLD_Fstar (D2R_Riemann_State *, D2R_Riemann_State *, double);

static int  D2R_HLLD_GetRiemannState (D2R_Riemann_State *, double, int);
static void D2R_HLLD_GetAState (D2R_Riemann_State *,  double p);
static void D2R_HLLD_GetCState (D2R_Riemann_State *, D2R_Riemann_State *, double, double *);

static double D2R_HLLD_TotalPressure  (double *);

static void D2R_HLLD_PrintStates(double *, double *);
static void D2R_HLLD_PrintWhatsWrong(D2R_Riemann_State *, D2R_Riemann_State *, 
                                 double, double *, double *);


/* Exported construction marker: maps/native retain ABI3. */
int d2r_sr_fixed_precision(void) {return 5;}
/* Additional fixed-helper failure bits, local to v4 (native enum unchanged). */
#define D2R_REFINE_NO_BRACKET 32
#define D2R_REFINE_UNUSABLE 64
typedef struct {double p,f;} D2R_Pressure_Point;

static void D2R_SavePressure(D2R_Pressure_Point *history,int *count,
                           double p,double f,int failed) {
  if(!failed && isfinite(p) && p>0. && isfinite(f) &&
     *count<HLLD_MAX_ITER+4) {
    history[*count].p=p;history[*count].f=f;++*count;
  }
}

static int D2R_RefreshAndRefine(D2R_Riemann_State *l,D2R_Riemann_State *r,
 double *pressure,D2R_Pressure_Point *history,int *count,
 int *calls,D2R_SR_Record *rec) {
  double p=*pressure;
  if(!isfinite(p)||!(p>0.)) {
    rec->root_failure|=D2R_REFINE_UNUSABLE;return 0;
  }
  /* Fstar also prepares every PaL/PaR/Sc quantity consumed by flux assembly.
   * This MUST be evaluated at the actual returned pressure, not its previous
   * secant iterate. The original source omits that final-state refresh. */
  double f=D2R_HLLD_Fstar(l,r,p);++*calls;
  rec->last_root_residual=l->fail?NAN:f;
  if(l->fail||!isfinite(f)) {
    rec->root_failure|=D2R_REFINE_UNUSABLE;
    if(l->fail)rec->root_failure|=D2R_SR_NATIVE_STATE_FAIL;
    if(!isfinite(f))rec->root_failure|=D2R_SR_NONFINITE_ROOT;
    return 0;
  }
  D2R_SavePressure(history,count,p,f,0);
  if(f==0.)return 1;
  int best_i=-1,best_j=-1;
  double best_distance=INFINITY,best_width=INFINITY;
  /* Select the closest sign-changing interval already visited by the SAME
   * native root search; narrowest interval breaks a distance tie. No new
   * guess, bracket expansion, or unrelated pressure branch is introduced. */
  for(int j=1;j<*count;++j)for(int i=0;i<j;++i) {
    if(history[i].f==0.||history[j].f==0.)continue;
    if(signbit(history[i].f)==signbit(history[j].f))continue;
    const double low=fmin(history[i].p,history[j].p);
    const double high=fmax(history[i].p,history[j].p);
    if(!(low<high))continue;
    const double distance=p<low?low-p:p>high?p-high:0.;
    const double width=high-low;
    if(distance<best_distance ||
       (distance==best_distance && width<=best_width)) {
      best_i=i;best_j=j;best_distance=distance;best_width=width;
    }
  }
  if(best_i<0) {rec->root_failure|=D2R_REFINE_NO_BRACKET;return 0;}
  double low=history[best_i].p,high=history[best_j].p;
  double flow=history[best_i].f,fhigh=history[best_j].f;
  if(low>high) {
    double t=low;low=high;high=t;t=flow;flow=fhigh;fhigh=t;
  }
  /* Each accepted midpoint strictly shrinks a finite positive binary64
   * bracket. Termination uses adjacency or an exact represented zero, with
   * no problem-dependent residual or relative-pressure tolerance. */
  while(nextafter(low,high)<high) {
    const double mid=low+.5*(high-low);
    if(!(mid>low && mid<high)) {
      rec->root_failure|=D2R_REFINE_UNUSABLE;return 0;
    }
    const double fm=D2R_HLLD_Fstar(l,r,mid);++*calls;
    rec->last_root_residual=l->fail?NAN:fm;
    if(l->fail||!isfinite(fm)) {
      rec->root_failure|=D2R_REFINE_UNUSABLE;
      if(l->fail)rec->root_failure|=D2R_SR_NATIVE_STATE_FAIL;
      if(!isfinite(fm))rec->root_failure|=D2R_SR_NONFINITE_ROOT;
      return 0;
    }
    if(fm==0.) {*pressure=mid;return 1;}
    if(signbit(fm)==signbit(flow)) {low=mid;flow=fm;}
    else {high=mid;fhigh=fm;}
  }
  p=fabs(flow)<=fabs(fhigh)?low:high;
  f=D2R_HLLD_Fstar(l,r,p);++*calls;
  rec->last_root_residual=l->fail?NAN:f;
  if(l->fail||!isfinite(f)) {
    rec->root_failure|=D2R_REFINE_UNUSABLE;return 0;
  }
  *pressure=p;return 1;
}

#define HLLD_DEBUG  NO
/* ********************************************************************* */
void D2R_HLLD_Fixed_Solver (const Sweep *sweep, int beg, int end, 
                  double *cmax, Grid *grid, const double *fixed_sl,
                  const double *fixed_sr, D2R_SR_Record *records)
/*!
 * Solve the Riemann problem using the HLLD Riemann solver.
 *
 * \param[in,out] sweep   pointer to Sweep structure
 * \param[in]     beg     initial grid index
 * \param[out]    end     final grid index
 * \param[out]    cmax    1D array of maximum characteristic speeds
 * \param[in]     grid    pointer to array of Grid structures.
 *
 *********************************************************************** */
{
  int    nv, i, k;
  int    switch_to_hll;
  double   scrh;
  static _Thread_local double **Uhll, **Fhll, **Vhll;
  double *vL, *vR, *fL, *fR, *uL, *uR, *SL, *SR;
 
  double p0, f0, p, f, dp, dS_1;
  double Uc[NVAR];
  D2R_Riemann_State PaL, PaR;
  double pguess;
  const State   *stateL = &(sweep->stateL);
  const State   *stateR = &(sweep->stateR);
  double **fluxL = stateL->flux, **fluxR = stateR->flux;
  double *a2L = stateL->a2,   *a2R = stateR->a2;
  double  *hL = stateL->h,     *hR = stateR->h;
  double  *pL = stateL->prs,   *pR = stateR->prs;

/* ----------------------------------------------
   0. Count Riemann solver failures   ---------------------------------------------- */

  #if COUNT_FAILURES == YES
  static _Thread_local double totfail, totzones; 
  RiemannCheck (&totzones, &totfail);
  #endif
 
/* ----------------------------------------------
   1. Memory allocation
   ---------------------------------------------- */

  if (Uhll == NULL){
    Uhll = ARRAY_2D(NMAX_POINT, NVAR, double);
    Fhll = ARRAY_2D(NMAX_POINT, NVAR, double);
    Vhll = ARRAY_2D(NMAX_POINT, NVAR, double);
  }

/*
  #if DIVB_CONTROL == EIGHT_WAVES
   printLog ("! hlld Riemann solver does not work with Powell's 8-wave\n");
   QUIT_PLUTO(1);
  #endif
*/

#ifdef GLM_MHD
  GLM_Solve (sweep, beg, end, grid);
#endif

/* --------------------------------------------------------
   2. Compute sound speed & fluxes at zone interfaces
   -------------------------------------------------------- */

  SoundSpeed2 (stateL, beg, end, FACE_CENTER, grid);
  SoundSpeed2 (stateR, beg, end, FACE_CENTER, grid);

  Flux (stateL, beg, end);
  Flux (stateR, beg, end);

/* -- compute speeds -- */

  SL = sweep->SL; SR = sweep->SR;
  /* D2R: prepare this pair's flux/EOS data, retain ORIGINAL outer speeds. */
  for (i = beg; i <= end; ++i){
    SL[i] = fixed_sl[i-beg]; SR[i] = fixed_sr[i-beg];
  }

/* --------------------------------------------------------
   3. Compute HLLD flux - Begin main loop 
   -------------------------------------------------------- */

  for (i = beg; i <= end; i++) {
    D2R_Pressure_Point pressure_history[HLLD_MAX_ITER+4];
    int pressure_count=0;
    D2R_SR_Record scratch_record;
    D2R_SR_Record *rec = records ? &records[i-beg] : &scratch_record;
    memset(rec, 0, sizeof(*rec));
    rec->input_valid = rec->bounds_valid = 1;
    rec->sl = SL[i]; rec->sr = SR[i];
    rec->pressure_guess = rec->pressure_returned = NAN;
    rec->last_root_residual = NAN;
    rec->alfven_left = rec->alfven_right = rec->contact = NAN;
    rec->flags_before = sweep->flag[i];
    rec->flags_after = sweep->flag[i];

#if HLLD_DEBUG == YES
    if (!(grid[IDIR].x[i] < 0.5 && grid[IDIR].x[i+1] > 0.5)) continue;
#endif

    #if COUNT_FAILURES == YES
    totzones += 1.0;
    #endif

    scrh  = MAX(fabs(SL[i]), fabs(SR[i]));
    cmax[i] = scrh;

    vL = stateL->v[i]; uL = stateL->u[i]; fL = stateL->flux[i];
    vR = stateR->v[i]; uR = stateR->u[i]; fR = stateR->flux[i];

  /* --------------------------------------------
     3a. Handle different cases 
     -------------------------------------------- */
  
 
    if (SL[i] >= 0.0){   /* -- Supersonic case: SL > 0 -- */
      rec->outcome = D2R_SR_UPWIND_LEFT;

      NFLX_LOOP(nv) sweep->flux[i][nv] = fL[nv];
      sweep->press[i] = pL[i];

    }else if (SR[i] <= 0.0){   /* -- Supersonic right case: SR < 0 -- */
      rec->outcome = D2R_SR_UPWIND_RIGHT;

      NFLX_LOOP(nv) sweep->flux[i][nv] = fR[nv];
      sweep->press[i] = pR[i];

    }else{

  /* -------------------------------------------- 
     3b. Build the HLL average state & flux 
     -------------------------------------------- */

      dS_1 = 1.0/(SR[i] - SL[i]);
      NFLX_LOOP(nv){ 
        Uhll[i][nv]  = SR[i]*uR[nv] - SL[i]*uL[nv]  + fL[nv] - fR[nv];
        Uhll[i][nv] *= dS_1;

        Fhll[i][nv]  = SR[i]*fL[nv] - SL[i]*fR[nv] + SL[i]*SR[i]*(uR[nv] - uL[nv]);
        Fhll[i][nv] *= dS_1;
      }
      Uhll[i][MXn] += (pL[i] - pR[i])*dS_1;
      #if NSCL > 0 
      NSCL_LOOP(nv) {
        double vxR = vR[VXn];
        double vxL = vL[VXn];
        Uhll[i][nv]  = (SR[i] - vxR)*uR[nv] - (SL[i] - vxL)*uL[nv];
        Uhll[i][nv] *= dS_1;

        Fhll[i][nv]  =   SR[i]*uL[nv]*vL[VXn] - SL[i]*uR[nv]*vR[VXn] 
                       + SL[i]*SR[i]*(uR[nv] - uL[nv]);
        Fhll[i][nv] *= dS_1;
      }
      #endif

  /* -------------------------------------------- 
     3c. Switch to HLL at strong shocks 
     -------------------------------------------- */

      #if SHOCK_FLATTENING == MULTID
      if ((sweep->flag[i] & FLAG_HLL) || (sweep->flag[i+1] & FLAG_HLL)){
        rec->outcome = D2R_SR_HLL_FLATTEN;
        NFLX_LOOP(nv) sweep->flux[i][nv] = Fhll[i][nv];
         sweep->press[i] = (SR[i]*pL[i] - SL[i]*pR[i])*dS_1;
        continue;
      }
      #endif

  /* --------------------------------------------
     3d. Build the right hand sides 
         (lambda*U - F) for L/R states
     -------------------------------------------- */

      PaL.S = SL[i];
      PaR.S = SR[i];
      Bx    = Uhll[i][BXn];
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

  /* --------------------------------------------
     3e. Provide initial guess 
     -------------------------------------------- */

      scrh = MAX(pL[i], pR[i]);
      if (Bx*Bx/scrh < 0.01) { /* -- try the B->0 limit -- */

        double a,b,c;
        a = SR[i] - SL[i];
        b = PaR.R[ENG] - PaL.R[ENG] + SR[i]*PaL.R[MXn] - SL[i]*PaR.R[MXn];
        c = PaL.R[MXn]*PaR.R[ENG] - PaR.R[MXn]*PaL.R[ENG];
        scrh = b*b - 4.0*a*c;
        scrh = MAX(scrh,0.0);
        p0 = 0.5*(- b + sqrt(scrh))*dS_1;

      }else{  /* ----  use HLL average ---- */
                         
        rec->hll_recovery_called = 1;
        rec->hll_recovery_status = ConsToPrim(Uhll, Vhll, i, i, sweep->flag);
        rec->flags_after = sweep->flag[i];
        p0    = D2R_HLLD_TotalPressure(Vhll[i]);
      }
      
  /* --------------------------------------------
     3f. Check if guess is physically ok.
         If not, switch to HLL. 
     --------------------------------------------  */

      pguess = p0;
      rec->pressure_guess = p0;
      switch_to_hll = 0;
      f0 = D2R_HLLD_Fstar(&PaL, &PaR, p0);
      /* A failed native state check can follow an early-return scratch
       * path. Its residual is unavailable, not another root diagnosis. */
      rec->last_root_residual = PaL.fail ? NAN : f0;
      D2R_SavePressure(pressure_history,&pressure_count,p0,f0,PaL.fail);
      if (f0 != f0 || PaL.fail) switch_to_hll = 1;
      if (switch_to_hll) rec->root_failure |= D2R_SR_BAD_GUESS;
      if (!PaL.fail && f0 != f0) rec->root_failure |= D2R_SR_NONFINITE_ROOT;
      if (PaL.fail) rec->root_failure |= D2R_SR_NATIVE_STATE_FAIL;

  /* --------------------------------------------
     3g. Start root solver 
     -------------------------------------------- */

      k = 0;
      /* v5: a nonzero initial residual requires the SAME native
       * secant path, even when its old loose threshold would skip it. */
      if (f0 != 0.0 && !switch_to_hll){
        rec->root_attempted = 1;
        p  = 1.025*p0; f  = f0;
        for (k = 1; k < HLLD_MAX_ITER; k++){

          #if HLLD_DEBUG == YES
          printf ("k = %d, p = %12.6e,  f = %12.6e\n",k,p,f);
          #endif

          f  = D2R_HLLD_Fstar(&PaL, &PaR, p);
          rec->last_root_residual = PaL.fail ? NAN : f;
          D2R_SavePressure(pressure_history,&pressure_count,p,f,PaL.fail);
          rec->root_iterations = k;
          if (f != f  || PaL.fail || (k > 7) || (fabs(f) > fabs(f0) && k > 4)) {
            if (!PaL.fail && f != f) rec->root_failure |= D2R_SR_NONFINITE_ROOT;
            if (PaL.fail) rec->root_failure |= D2R_SR_NATIVE_STATE_FAIL;
            if (k > 7) rec->root_failure |= D2R_SR_ITERATION_LIMIT;
            if (!PaL.fail && fabs(f) > fabs(f0) && k > 4)
              rec->root_failure |= D2R_SR_GROWING_RESIDUAL;
            switch_to_hll = 1;
            break;
          }

          dp = (p - p0)/(f - f0)*f;

          p0 = p; f0 = f;
          p -= dp;
          if (p < 0.0) p = 1.e-6;
          if (fabs(dp) < 1.e-5*p || fabs(f) < 1.e-6) break;
        }
      }else p = p0;

      #if HLLD_DEBUG == YES
      D2R_HLLD_PrintWhatsWrong(&PaL, &PaR, phll, phll, p0Bx, p, vL, vR);
      #endif

    /* ----  too many iter ? --> use HLL ---- */

      if(!switch_to_hll && !PaL.fail) {
        int refine_calls=0;
        if(!D2R_RefreshAndRefine(&PaL,&PaR,&p,pressure_history,&pressure_count,
                                &refine_calls,rec)) switch_to_hll=1;
        k+=refine_calls;
      }
      if (PaL.fail) switch_to_hll = 1;
      if (PaL.fail) rec->root_failure |= D2R_SR_NATIVE_STATE_FAIL;
      rec->pressure_returned = p;
      rec->root_iterations = k;
      if (switch_to_hll){
        rec->outcome = D2R_SR_HLL_ROOT;

        #if COUNT_FAILURES == YES
        totfail += 1.0;
        #endif

        NFLX_LOOP(nv) sweep->flux[i][nv] = Fhll[i][nv];
        sweep->press[i]  = (SR[i]*pL[i] - SL[i]*pR[i])*dS_1;
        continue;
      }

  /* -- ok, solution should be reliable -- */

      rec->outcome = D2R_SR_HLLD;
      rec->alfven_left = PaL.Sa; rec->alfven_right = PaR.Sa;
      rec->contact = Sc;
      g_maxRiemannIter = MAX(g_maxRiemannIter, k);

      #ifdef GLM_MHD
      PaL.u[PSI_GLM] = PaR.u[PSI_GLM] = Uc[PSI_GLM] = uL[PSI_GLM];
      #endif

      if (PaL.Sa >= -1.e-6){      

        D2R_HLLD_GetAState (&PaL, p);
        #if RMHD_REDUCED_ENERGY == YES
         PaL.u[ENG] -= PaL.u[RHO];
        #endif
        NFLX_LOOP(nv) {
          sweep->flux[i][nv] = fL[nv] + SL[i]*(PaL.u[nv] - uL[nv]);
        }
        sweep->press[i] = pL[i];

      }else if (PaR.Sa <= 1.e-6){

        D2R_HLLD_GetAState (&PaR, p);
        #if RMHD_REDUCED_ENERGY == YES
         PaR.u[ENG] -= PaR.u[RHO];
        #endif
        NFLX_LOOP(nv) {
          sweep->flux[i][nv] = fR[nv] + SR[i]*(PaR.u[nv] - uR[nv]);
        }
        sweep->press[i] = pR[i];

      }else{

        D2R_HLLD_GetCState (&PaL, &PaR, p, Uc);
        if (Sc > 0.0){
          #if RMHD_REDUCED_ENERGY == YES
           PaL.u[ENG] -= PaL.u[RHO];
           Uc[ENG]    -= Uc[RHO];
          #endif
          NFLX_LOOP(nv) {
            sweep->flux[i][nv] = fL[nv] + SL[i]*(PaL.u[nv] - uL[nv]) 
                                        + PaL.Sa*(Uc[nv] - PaL.u[nv]);
          }
          sweep->press[i] = pL[i];

        }else{
          #if RMHD_REDUCED_ENERGY == YES
          PaR.u[ENG] -= PaR.u[RHO];
          Uc[ENG]    -= Uc[RHO];
          #endif
          NFLX_LOOP(nv) {
            sweep->flux[i][nv] = fR[nv] + SR[i]*(PaR.u[nv] - uR[nv]) 
                                        + PaR.Sa*(Uc[nv] - PaR.u[nv]);
          }
          sweep->press[i] = pR[i];
        }  
      }
    } /* --- end if on SL, SR -- */
    rec->flags_after = sweep->flag[i];
  } /* --- end loop on points -- */

/* --------------------------------------------------------
   4. Define point and diffusive fluxes for CT
   -------------------------------------------------------- */

#ifdef GLM_MHD
  /* GLM_PAIR (2026-10-08, default off; environment GLM_PAIR=1): the energy counterpart of the
     cleaning flux. PLUTO's RMHD GLM transports B_n with flux psi* and psi with c_h^2 B_n*, but
     the total-energy flux carries no matching term, so the non-solenoidal part of the field update
     changes magnetic energy at fixed total energy and the gas pays (at beta ~ 1e-3 that is a large
     fraction of the thermal energy). The relativistic entropy variables give the pairing
     dE = (b/W).dB with b/W = (1 - v^2) B + (v.B) v; in flux form on every face
     F_E += psi* [(1 - v^2) B_n* + (v.B) v_n], the analogue of the extended-GLM energy term. */
  if (glm_pair_on < 0){
    const char *e = getenv ("GLM_PAIR");
    glm_pair_on = (e != NULL ? atoi (e) : 0);
    if (glm_pair_on) print ("> %s: GLM_PAIR = 1 (energy flux paired with the cleaning flux)\n", __func__);
  }
  if (glm_pair_on){
    for (i = beg; i <= end; i++){
      double *vl = stateL->v[i], *vr = stateR->v[i];
      double psi = vl[PSI_GLM], Bn = vl[BXn];               /* GLM_Solve's common face values */
      double vn = 0.5*(vl[VXn] + vr[VXn]), vt = 0.5*(vl[VXt] + vr[VXt]), vb = 0.5*(vl[VXb] + vr[VXb]);
      double Bt = 0.5*(vl[BXt] + vr[BXt]), Bb = 0.5*(vl[BXb] + vr[BXb]);
      double v2 = vn*vn + vt*vt + vb*vb, vB = vn*Bn + vt*Bt + vb*Bb;
      sweep->flux[i][ENG] += psi*((1.0 - v2)*Bn + vB*vn);
    }
  }
#endif

#if DIVB_CONTROL == CONSTRAINED_TRANSPORT 
  CT_Flux (sweep, beg, end, grid);
#endif

/* --------------------------------------------------------
              initialize source term
   -------------------------------------------------------- */
 
  #if DIVB_CONTROL == EIGHT_WAVES
   HLL_DIVB_SOURCE (sweep, Uhll, beg + 1, end, grid);
  #endif

}

/* ********************************************************************* */
double D2R_HLLD_Fstar (D2R_Riemann_State *PaL, D2R_Riemann_State *PaR, double p)
/*!
 * Return the velocity difference across the contact mode as a 
 * function of the total pressure p.
 *
 *********************************************************************** */
{
  int    success = 1;
  double dK, Bxc, Byc, Bzc;
  double sBx, fun;
  double vxcL, KLBc;
  double vxcR, KRBc;

  sBx = Bx > 0.0 ? 1.0:-1.0;

  success *= D2R_HLLD_GetRiemannState (PaL, p, -1);
  success *= D2R_HLLD_GetRiemannState (PaR, p,  1);

/* -- compute B from average sweep -- */

  dK  = PaR->Kx - PaL->Kx + 1.e-12;

  Bxc = Bx*dK;

  Byc =   PaR->By*(PaR->Kx - PaR->vx) 
        - PaL->By*(PaL->Kx - PaL->vx)
        + Bx*(PaR->vy - PaL->vy);

  Bzc =   PaR->Bz*(PaR->Kx - PaR->vx) 
        - PaL->Bz*(PaL->Kx - PaL->vx)
        + Bx*(PaR->vz - PaL->vz);
  
  KLBc = PaL->Kx*Bxc + PaL->Ky*Byc + PaL->Kz*Bzc;
  KRBc = PaR->Kx*Bxc + PaR->Ky*Byc + PaR->Kz*Bzc;

  vxcL = PaL->Kx - dK*Bx*(1.0 - PaL->K2)/(PaL->sw*dK - KLBc);
  vxcR = PaR->Kx - dK*Bx*(1.0 - PaR->K2)/(PaR->sw*dK - KRBc);

  PaL->Sa = PaL->Kx;
  PaR->Sa = PaR->Kx;
  Sc      = 0.5*(vxcL + vxcR);
  fun     = vxcL - vxcR;   /* -- essentially, Eq. [48] -- */

/*
fun = dK*(1.0 - Bx*(  (1.0 - PaR->K2)/(PaR->sw*dK - KRBc)
                    - (1.0 - PaL->K2)/(PaL->sw*dK - KLBc)) );
*/

  /* -- check if state makes physically sense -- */

  success  = (vxcL - PaL->Kx)  > -1.e-6;
  success *= (PaR->Kx  - vxcR) > -1.e-6;

  success *= (PaL->S - PaL->vx) < 0.0;
  success *= (PaR->S - PaR->vx) > 0.0;

  success *= (PaR->w - p) > 0.0;
  success *= (PaL->w - p) > 0.0;
  success *= (PaL->Sa - PaL->S)  > -1.e-6;
  success *= (PaR->S  - PaR->Sa) > -1.e-6;

  PaL->fail = !success;

/*
scrh  = (1.0 - PaR->K2)*(PaL->sw*dK - KLBc);
scrh -= (1.0 - PaL->K2)*(PaR->sw*dK - KRBc);

PaL->fun1 = (PaR->sw*dK - KRBc)*(PaL->sw*dK - KLBc) - Bx*scrh; 
PaL->fun2 = scrh;
PaL->denL = (PaL->sw*dK - KLBc);
PaL->denR = (PaR->sw*dK - KRBc);
*/
  return (fun);
}

/* ********************************************************************* */
int D2R_HLLD_GetRiemannState (D2R_Riemann_State *Pv, double p, int side)
/*!
 * Express the sweep behind a wave as function of the total
 * pressure p and the right hand side on the other side of the wave.
 *  
 * On output, return 1 if succesful, 0 if w < 0 is encountered.
 * side = -1 : means left
 * side =  1 : means right
 * 
 *********************************************************************** */
{
  double A, C, G, X, s;
  double vx, vy, vz, scrh, S, *R;

  S = Pv->S;
  R = Pv->R;

  A = R[MXn] + p*(1.0 - S*S) - S*R[ENG];   /* Eq. [26] */      
  G = R[BXt]*R[BXt] + R[BXb]*R[BXb];       /* Eq. [27] */
  C = R[BXt]*R[MXt] + R[BXb]*R[MXb];       /* Eq. [28] */
  X = Bx*(A*S*Bx + C) - (A + G)*(S*p + R[ENG]);      /* Eq. [30] */

/* -- compute the numerators of Eqs. [23,23,45] -- */

  vx = ( Bx*(A*Bx + C*S) - (R[MXn] + p)*(G + A) );

  vy = ( - (A + G - Bx*Bx*(1.0 - S*S))*R[MXt]     
         + R[BXt]*(C + Bx*(S*R[MXn] - R[ENG])) ); 
 
  vz = ( - (A + G - Bx*Bx*(1.0 - S*S))*R[MXb]          
         + R[BXb]*(C + Bx*(S*R[MXn] - R[ENG])) );

  scrh = vx*R[MXn] + vy*R[MXt] + vz*R[MXb];
  scrh = X*R[ENG] - scrh;
  Pv->w = p + scrh/(X*S - vx);  /* Eq [31] */

  if (Pv->w < 0.0) return 0;  /* -- failure -- */

  Pv->vx = vx/X;
  Pv->vy = vy/X;
  Pv->vz = vz/X;
         
/*  -- original Eq. [21] -- */
/*
  Pv->Bx = Bx; 
  Pv->By = (R[BXt]*X - Bx*vy)/(X*S - vx); 
  Pv->Bz = (R[BXb]*X - Bx*vz)/(X*S - vx);
*/
/* --------------------------------------------------------------------- */
/*! When computing Bx, By and Bz, we use Eq. [21] with 
    vx, vy and vz replaced by by Eq. [23,24,25]. 
    The following short MAPLE script is used to verify the 
    correctness.
 \code
   restart;
   A    := R[mx] - lambda*R[E] + p*(1 - lambda^2):
   G    := R[By]*R[By] + R[Bz]*R[Bz]:
   C    := R[my]*R[By] + R[mz]*R[Bz]:
   Q    := - A - G + B[x]*B[x]*(1-lambda^2):
   X    := B[x]*(A*lambda*B[x]+ C) - (A + G)*(lambda*p + R[E]):
   v[x] := (B[x]*(A*B[x] + lambda*C) - (A + G)*(p + R[mx]))/X:
   v[y] := (Q*R[my] + R[By]*(C + B[x]*(lambda*R[mx] - R[E])))/X:
   B[y] := (R[By] - B[x]*v[y])/(lambda - v[x]):
   simplify(B[y]);
 \endcode
*/
/* --------------------------------------------------------------------- */

  Pv->Bx = Bx;
  Pv->By = -(R[BXt]*(S*p + R[ENG]) - Bx*R[MXt])/A;
  Pv->Bz = -(R[BXb]*(S*p + R[ENG]) - Bx*R[MXb])/A;
  
  s  = Bx > 0.0 ? 1.0:-1.0;
  if (side < 0) s *= -1.0;

  Pv->sw = s*sqrt(Pv->w);

  scrh = 1.0/(S*p +  R[ENG] + Bx*Pv->sw);
  Pv->Kx = scrh*(R[MXn] + p + R[BXn]*Pv->sw);
  Pv->Ky = scrh*(R[MXt]     + R[BXt]*Pv->sw);
  Pv->Kz = scrh*(R[MXb]     + R[BXb]*Pv->sw);

  Pv->K2 = Pv->Kx*Pv->Kx + Pv->Ky*Pv->Ky + Pv->Kz*Pv->Kz;
  return 1; /* -- success -- */
}
/* ********************************************************************* */
void D2R_HLLD_GetAState (D2R_Riemann_State *Pa,  double p)
/*!
 *  Compute states aL and aR behind fast waves.
 *
 *********************************************************************** */
{
  double vB, *ua, *R, S;
  double scrh;
  
  ua = Pa->u;
  S  = Pa->S;
  R  = Pa->R;

  scrh = 1.0/(S - Pa->vx);

  ua[RHO] =  R[RHO]*scrh;
  ua[BXn] =  Bx;
  ua[BXt] = (R[BXt] - Bx*Pa->vy)*scrh;
  ua[BXb] = (R[BXb] - Bx*Pa->vz)*scrh;

  vB     = Pa->vx*ua[BXn] + Pa->vy*ua[BXt] + Pa->vz*ua[BXb];
  ua[ENG] = (R[ENG] + p*Pa->vx - vB*Bx)*scrh;

  ua[MXn] = (ua[ENG] + p)*Pa->vx - vB*ua[BXn];
  ua[MXt] = (ua[ENG] + p)*Pa->vy - vB*ua[BXt];
  ua[MXb] = (ua[ENG] + p)*Pa->vz - vB*ua[BXb];
}

/* ********************************************************************* */
void D2R_HLLD_GetCState (D2R_Riemann_State *PaL, D2R_Riemann_State *PaR, double p,
                     double *Uc)
/*!
 *  Compute states cL and cR across contact mode.
 *
 *************************************************************** */
{
  double dK, *ua;
  double vxcL, vycL, vzcL, KLBc;
  double vxcR, vycR, vzcR, KRBc;
  double vxc, vyc, vzc, vBc;
  double Bxc, Byc, Bzc, Sa, vxa;
  double scrhL, scrhR;

  D2R_HLLD_GetAState (PaL, p);
  D2R_HLLD_GetAState (PaR, p);
  dK = (PaR->Kx - PaL->Kx) + 1.e-12;

  Bxc = Bx*dK;

  Byc =   PaR->By*(PaR->Kx - PaR->vx)    /* Eq. [45] */
        - PaL->By*(PaL->Kx - PaL->vx)
        + Bx*(PaR->vy - PaL->vy);

  Bzc =   PaR->Bz*(PaR->Kx - PaR->vx)   /* Eq.[45] */
        - PaL->Bz*(PaL->Kx - PaL->vx)
        + Bx*(PaR->vz - PaL->vz);
   
  Bxc  = Bx;
  Byc /= dK;
  Bzc /= dK;

  Uc[BXn] = Bxc;
  Uc[BXt] = Byc;
  Uc[BXb] = Bzc;

  KLBc = PaL->Kx*Bxc + PaL->Ky*Byc + PaL->Kz*Bzc;
  KRBc = PaR->Kx*Bxc + PaR->Ky*Byc + PaR->Kz*Bzc;

  scrhL = (1.0 - PaL->K2)/(PaL->sw - KLBc);
  scrhR = (1.0 - PaR->K2)/(PaR->sw - KRBc);

  vxcL = PaL->Kx - Uc[BXn]*scrhL;     /* Eq [47] */
  vxcR = PaR->Kx - Uc[BXn]*scrhR;

  vycL = PaL->Ky - Uc[BXt]*scrhL;  
  vycR = PaR->Ky - Uc[BXt]*scrhR;

  vzcL = PaL->Kz - Uc[BXb]*scrhL;
  vzcR = PaR->Kz - Uc[BXb]*scrhR;

  vxc = 0.5*(vxcL + vxcR);
  vyc = 0.5*(vycL + vycR);
  vzc = 0.5*(vzcL + vzcR);

  if (vxc > 0.0) {
    D2R_HLLD_GetAState (PaL, p);
    ua  = PaL->u;
    Sa  = PaL->Sa;
    vxa = PaL->vx;
  } else {
    D2R_HLLD_GetAState (PaR, p);
    ua  = PaR->u;
    Sa  = PaR->Sa;
    vxa = PaR->vx;
  }
  
  vBc = vxc*Uc[BXn] + vyc*Uc[BXt] + vzc*Uc[BXb];

  Uc[RHO] = ua[RHO]*(Sa - vxa)/(Sa - vxc);                      /* Eq [32] */
  Uc[ENG] = (Sa*ua[ENG] - ua[MXn] + p*vxc - vBc*Bx)/(Sa - vxc); /* Eq [33] */

  Uc[MXn] = (Uc[ENG] + p)*vxc - vBc*Bx;               /* Eq [34] */
  Uc[MXt] = (Uc[ENG] + p)*vyc - vBc*Uc[BXt];  
  Uc[MXb] = (Uc[ENG] + p)*vzc - vBc*Uc[BXb];
}

/* ********************************************************************* */
double D2R_HLLD_TotalPressure (double *v)
/*!
 * Compute total pressure
 *
 *********************************************************************** */
{
  double vel2, Bmag2, vB, lor2;
  double pt;

  Bmag2 = v[BX1]*v[BX1] + v[BX2]*v[BX2] + v[BX3]*v[BX3];
  vel2  = v[VX1]*v[VX1] + v[VX2]*v[VX2] + v[VX3]*v[VX3];
  vB    = v[VX1]*v[BX1] + v[VX2]*v[BX2] + v[VX3]*v[BX3];
  pt    = v[PRS] + 0.5*(Bmag2*(1.0 - vel2) + vB*vB);
  return(pt);  
}


void D2R_HLLD_PrintStates(double *vL, double *vR)
{
  printf ("RHO_LEFT  %18.9e\n",vL[RHO]);
  printf ("VX_LEFT   %18.9e\n",vL[VXn]);
  printf ("VY_LEFT   %18.9e\n",vL[VXt]);
  printf ("VZ_LEFT   %18.9e\n",vL[VXb]);
  printf ("BY_LEFT   %18.9e\n",vL[BXt]);
  printf ("BZ_LEFT   %18.9e\n",vL[BXb]);
  printf ("PR_LEFT   %18.9e\n",vL[PRS]);

  printf ("RHO_RIGHT  %18.9e\n",vR[RHO]);
  printf ("VX_RIGHT   %18.9e\n",vR[VXn]);
  printf ("VY_RIGHT   %18.9e\n",vR[VXt]);
  printf ("VZ_RIGHT   %18.9e\n",vR[VXb]);
  printf ("BY_RIGHT   %18.9e\n",vR[BXt]);
  printf ("BZ_RIGHT   %18.9e\n",vR[BXb]);
  printf ("PR_RIGHT   %18.9e\n",vR[PRS]);
  printf ("BX_CONST   %18.9e\n",vR[BXn]);
}


void D2R_HLLD_PrintWhatsWrong(D2R_Riemann_State *PaL, D2R_Riemann_State *PaR, 
                      double pguess, double *vL, double *vR)
{
  double f,p;
  FILE *fp;

  D2R_HLLD_PrintStates(vL, vR);

  D2R_HLLD_GetAState (PaL, pguess);
  D2R_HLLD_GetAState (PaR, pguess);

  printf ("pguess  = %f, F = %f\n",pguess, D2R_HLLD_Fstar(PaL, PaR, pguess));
  printf ("S = %f  %f  %f  %f  %f\n", PaL->S, PaL->Sa, Sc, PaR->Sa, PaR->S);

  fp = fopen("new.dat","w");
  for (p = 0.01*pguess; p <= 10.0*pguess; p += 1.e-4*pguess){
    f = D2R_HLLD_Fstar(PaL, PaR, p);
    fprintf (fp,"%f  %f %f  %f\n", p, f, Sc - PaL->Sa, PaR->Sa - Sc);
  }
  fclose(fp);
  exit(1);
}


