/* Private SpECTRE D2-R scalar bridge, adapted from source_v5 tau core.
 * Native hlld.c is linked from the existing vendor target, never redefined.
 * The fixed-v5 helper preserves its numerical algebra with TLS storage only.
 * No globals.h: pluto_hlld_shim.c already owns the vendor global definitions.
 * The C++ D2-R kernel serializes its complete call sequence. */
#include "d2r_sr_face.h"
#include "spectre_d2r_core.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if PHYSICS != RMHD || EOS != IDEAL || DIVB_CONTROL != NO || NSCL != 0
#error "tau core requires ideal seven-wave common-Bn SRMHD"
#endif
#if RMHD_REDUCED_ENERGY != YES
#error "tau core requires native PLUTO reduced energy"
#endif
/* Match the legacy 1024-interface shim plus its two-point headroom. Native
 * HLLD/mapper scratch is lazily allocated only once PER THREAD; letting this
 * scalar entrypoint allocate it with capacity16 first would make a later
 * legacy full-face call overrun it. All persistent data below are TLS. */
#define CORE_NP 1026
static _Thread_local Grid core_grid;
static _Thread_local D2R_SR_Context core_context[2];
static _Thread_local int core_initialized;

int d2r_sr_core_abi(void) { return D2R_CORE_ABI; }

static int core_setup(void) {
  /* These are TLS in the existing vendor embedding. Do not repeatedly write
   * process-global floors, simulation step/time, or diagnostic stage values. */
  if (NMAX_POINT < CORE_NP) NMAX_POINT = CORE_NP;
  SetVectorIndices(IDIR);
  if (core_initialized) return 1;
  for (int d=0; d<3; ++d) {
    if (!core_grid.x[d]) {
      core_grid.x[d] = calloc((size_t)NMAX_POINT, sizeof(double));
      if (!core_grid.x[d]) return 0;
    }
  }
  Where(0, &core_grid);
  if (!D2R_SR_InitContext(&core_context[0]) ||
      !D2R_SR_InitContext(&core_context[1])) return 0;
  core_initialized = 1;
  return 1;
}

static int check_w(const double *w, double bn, double gamma) {
  if (!isfinite(bn)||!isfinite(gamma)||gamma<=1.||gamma>2.) return 0;
  for (int j=0;j<7;++j) if (!isfinite(w[j])) return 0;
  if (!(w[0]>0.) || !(w[4]>0.)) return 0;
  const double v2=w[1]*w[1]+w[2]*w[2]+w[3]*w[3];
  const long double vl2=(long double)w[1]*w[1]+(long double)w[2]*w[2]
                        +(long double)w[3]*w[3];
  if ((v2<1.) != (vl2<1.L)) return -5;
  return v2<1. && vl2<1.L;
}

static int good_bounds(double sl,double sr) {
  return isfinite(sl)&&isfinite(sr)&&sl<sr&&sl>=-1.&&sr<=1.;
}

static void encode_side(State *s,const double *w,double bn) {
  double *v=s->v[0];
  for(int j=0;j<NVAR;++j) v[j]=0.;
  v[RHO]=w[0];v[VX1]=w[1];v[VX2]=w[2];v[VX3]=w[3];
  v[PRS]=w[4];v[BX1]=bn;v[BX2]=w[5];v[BX3]=w[6];
  PrimToCons(s->v,s->u,0,0); Enthalpy(s->v,s->h,0,0);
}

static void physical_tau(const State *s,double *u,double *f) {
  const double *p=s->u[0],*q=s->flux[0];
  u[0]=p[RHO];u[1]=p[MX1];u[2]=p[MX2];u[3]=p[MX3];u[4]=p[ENG];
  u[5]=p[BX2];u[6]=p[BX3];
  f[0]=q[RHO];f[1]=q[MX1]+s->prs[0];f[2]=q[MX2];f[3]=q[MX3];
  f[4]=q[ENG];f[5]=q[BX2];f[6]=q[BX3];
}

static int prepare_pair(D2R_SR_Context *c,const double *row,double *u,double *f) {
  const double *wl=row+2,*wr=row+9;
  int left=check_w(wl,row[1],row[0]),right=check_w(wr,row[1],row[0]);
  if (left!=1 || right!=1) return left==-5||right==-5 ? -5 : 0;
  SetVectorIndices(IDIR);g_gamma=row[0];
  c->sweep.flag[0]=c->sweep.flag[1]=0;
  encode_side(&c->sweep.stateL,wl,row[1]);
  encode_side(&c->sweep.stateR,wr,row[1]);
  State *sides[]={&c->sweep.stateL,&c->sweep.stateR};
  for(int side=0;side<2;++side) {
    State *s=sides[side];
    for(int j=0;j<NVAR;++j) if(!isfinite(s->u[0][j])) return -1;
    if(!isfinite(s->h[0]) || !(s->h[0]>0.)) return -1;
    SoundSpeed2(s,0,0,FACE_CENTER,NULL);Flux(s,0,0);
    physical_tau(s,u+7*side,f+7*side);
    for(int j=0;j<7;++j) if(!isfinite(u[7*side+j])||!isfinite(f[7*side+j])) return -1;
    if(!isfinite(s->a2[0]) || !(s->a2[0]>0.) || !(s->a2[0]<1.)) return -2;
    double waves[NFLX];
    if(Magnetosonic(s->v[0],s->a2[0],s->h[0],waves)) return -2;
    if(!good_bounds(waves[KFASTM],waves[KFASTP])) return -2;
  }
  return 1;
}

static void copy_record(const D2R_SR_Record *r,int *i,double *d) {
  i[1]=r->input_valid;i[2]=r->bounds_valid;i[3]=r->finite_flux;
  i[5]=r->outcome;i[6]=r->root_failure;i[7]=r->root_iterations;
  i[8]=r->root_attempted;i[9]=r->hll_recovery_called;
  i[10]=r->hll_recovery_status;i[11]=r->unresolved_velocity;
  i[12]=(int)r->flags_before;i[13]=(int)r->flags_after;
  d[0]=r->sl;d[1]=r->sr;d[2]=r->pressure_guess;d[3]=r->pressure_returned;
  d[4]=r->last_root_residual;d[5]=r->alfven_left;d[6]=r->alfven_right;
  d[7]=r->contact;
}

int d2r_sr_core_batch(size_t n,int mode,const double *faces,const double *fixed,
 double *numerical,double *states,double *physical,double *used_bounds,
 int *integers,double *doubles) {
  if(mode<0||mode>2||!faces||!numerical||!states||!physical||!used_bounds||
     !integers||!doubles||(mode==1&&!fixed)||!core_setup()) return 0;
  for(size_t k=0;k<n;++k) {
    double *num=numerical+7*k,*u=states+14*k,*f=physical+14*k;
    double *b=used_bounds+2*k,*d=doubles+D2R_CORE_ND*k;
    int *i=integers+D2R_CORE_NI*k;
    for(int j=0;j<7;++j) num[j]=NAN;
    for(int j=0;j<14;++j) u[j]=f[j]=NAN;
    for(int j=0;j<D2R_CORE_ND;++j) d[j]=NAN;
    for(int j=0;j<D2R_CORE_NI;++j) i[j]=-1;
    b[0]=b[1]=NAN;i[0]=0;i[1]=0;i[2]=0;i[3]=0;i[11]=0;
    D2R_SR_Context *c=&core_context[mode==1];
    const int prepared=prepare_pair(c,faces+16*k,u,f);
    if(prepared!=1) {i[0]=prepared;i[11]=prepared==-5;continue;}
    i[1]=1;
    if(mode==1) {
      b[0]=fixed[2*k];b[1]=fixed[2*k+1];
    } else {
      HLL_Speed(&c->sweep.stateL,&c->sweep.stateR,c->sweep.SL,c->sweep.SR,0,0);
      b[0]=c->sweep.SL[0];b[1]=c->sweep.SR[0];
    }
    d[0]=b[0];d[1]=b[1];
    if(!good_bounds(b[0],b[1])) {i[0]=-3;continue;}
    i[2]=1;
    if(mode==2) {i[0]=1;continue;}
    if(mode==0) {
      g_maxRiemannIter=0;
      HLLD_Solver(&c->sweep,0,0,c->cmax,NULL);
      i[4]=g_maxRiemannIter;
      b[0]=c->sweep.SL[0];b[1]=c->sweep.SR[0];d[0]=b[0];d[1]=b[1];
    } else {
      D2R_SR_Record r;
      memset(&r,0,sizeof(r));
      r.input_valid=r.bounds_valid=1;
      r.pressure_guess=r.pressure_returned=r.last_root_residual=NAN;
      r.alfven_left=r.alfven_right=r.contact=NAN;
      double sl[1]={b[0]},sr[1]={b[1]};
      D2R_HLLD_Fixed_Solver(&c->sweep,0,0,c->cmax,NULL,sl,sr,&r);
      copy_record(&r,i,d);
    }
    const double *q=c->sweep.flux[0];
    num[0]=q[RHO];num[1]=q[MX1]+c->sweep.press[0];num[2]=q[MX2];
    num[3]=q[MX3];num[4]=q[ENG];num[5]=q[BX2];num[6]=q[BX3];
    i[3]=1;
    for(int j=0;j<7;++j) if(!isfinite(num[j])) i[3]=0;
    i[2]=good_bounds(b[0],b[1]);
    i[0]=!i[2] ? -3 : i[3] ? 1 : -4;
    /* Return the same encoded states and physical flux arrays actually used
     * by the core, rather than an independent analytic surrogate. */
    physical_tau(&c->sweep.stateL,u,f);
    physical_tau(&c->sweep.stateR,u+7,f+7);
  }
  return 1;
}
