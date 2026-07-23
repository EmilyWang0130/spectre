! Distributed under the MIT License.
! See LICENSE.txt for details.
!
! C-callable diagnostic wrapper around the WHISKY Marquina kernel, used by
! bench_marquina_crosscheck.cpp to compare SpECTRE's Marquina flux against the
! WHISKY reference on *identical* input states. The kernel math is transcribed
! verbatim from whisky_reference/Whisky_Eigenproblem_Marquina.F90
! (ANALYTICAL=.TRUE., FAST=.TRUE.), specialized to flat metric, unit normal
! along x, alpha=1, beta=0.
!
! For a left/right state pair it returns, for the LEFT state:
!   - the three characteristic speeds (lambda_+, lambda_-, lambda_0 = v_x),
!   - the two acoustic right eigenvectors R+, R- (order D,Sx,Sy,Sz,tau),
!   - the two acoustic left eigenvectors L+, L-,
! and the 5-component Marquina dissipation  diss = rflux_r - rflux_l  (the term
! WHISKY subtracts in f_marquina = 0.5*(F_L + F_R - diss)).

module whisky_crosscheck
  use, intrinsic :: iso_c_binding, only: c_double
  implicit none
  integer, parameter :: dp = kind(1.0d0)
contains

  subroutine whisky_marquina_diag( &
       rhoL, vxL, vyL, vzL, epsL, pressL, cs2L, dpdepsL, &
       densL, sxL, syL, szL, tauL, &
       rhoR, vxR, vyR, vzR, epsR, pressR, cs2R, dpdepsR, &
       densR, sxR, syR, szR, tauR, &
       lam_l, reivecp_l, reivecm_l, leivecp_l, leivecm_l, diss) &
       bind(c, name='whisky_marquina_diag')
    real(c_double), intent(in), value :: rhoL, vxL, vyL, vzL, epsL
    real(c_double), intent(in), value :: pressL, cs2L, dpdepsL
    real(c_double), intent(in), value :: densL, sxL, syL, szL, tauL
    real(c_double), intent(in), value :: rhoR, vxR, vyR, vzR, epsR
    real(c_double), intent(in), value :: pressR, cs2R, dpdepsR
    real(c_double), intent(in), value :: densR, sxR, syR, szR, tauR
    real(c_double), intent(out) :: lam_l(3)
    real(c_double), intent(out) :: reivecp_l(5), reivecm_l(5)
    real(c_double), intent(out) :: leivecp_l(5), leivecm_l(5)
    real(c_double), intent(out) :: diss(5)

    real(dp) :: one, two, u
    real(dp) :: enthalpyl, wl, v2l, vlowxl, vlowyl, vlowzl
    real(dp) :: enthalpyr, wr, v2r, vlowxr, vlowyr, vlowzr
    real(dp) :: lamp_nobetal, lamm_nobetal, lamp_nobetar, lamm_nobetar
    real(dp) :: lam1, lamp, lamm
    real(dp) :: axpl, axml, vxpl, vxml, kappal, xsil, dltl
    real(dp) :: axpr, axmr, vxpr, vxmr, kappar, xsir, dltr
    real(dp) :: cxx, cxy, cxz, gam, tmp1, tmp2, sump, summ, vxa, vxb
    real(dp) :: leivecpr(5), leivecmr(5)
    real(dp) :: du(5), rfluxl(5), rfluxr(5)
    integer :: i

    one = 1.0d0
    two = 2.0d0
    u = 1.0d0   ! g^xx for flat metric

    ! ---- LEFT state ----
    enthalpyl = one + epsL + pressL / rhoL
    vlowxl = vxL
    vlowyl = vyL
    vlowzl = vzL
    v2l = vxL*vxL + vyL*vyL + vzL*vzL
    wl = one / sqrt(one - v2l)
    lamp_nobetal = (vxL*(one-cs2L) + sqrt(cs2L*(one-v2l)* &
         (u*(one-v2l*cs2L) - vxL**2*(one-cs2L))))/(one-v2l*cs2L)
    lamm_nobetal = (vxL*(one-cs2L) - sqrt(cs2L*(one-v2l)* &
         (u*(one-v2l*cs2L) - vxL**2*(one-cs2L))))/(one-v2l*cs2L)

    ! ---- RIGHT state ----
    enthalpyr = one + epsR + pressR / rhoR
    vlowxr = vxR
    vlowyr = vyR
    vlowzr = vzR
    v2r = vxR*vxR + vyR*vyR + vzR*vzR
    wr = one / sqrt(one - v2r)
    lamp_nobetar = (vxR*(one-cs2R) + sqrt(cs2R*(one-v2r)* &
         (u*(one-v2r*cs2R) - vxR**2*(one-cs2R))))/(one-v2r*cs2R)
    lamm_nobetar = (vxR*(one-cs2R) - sqrt(cs2R*(one-v2r)* &
         (u*(one-v2r*cs2R) - vxR**2*(one-cs2R))))/(one-v2r*cs2R)

    ! ---- Componentwise-max abs speeds (Marquina) ----
    lam1 = dmax1(dabs(vxL), dabs(vxR))
    lamp = dmax1(dabs(lamp_nobetal), dabs(lamp_nobetar))
    lamm = dmax1(dabs(lamm_nobetal), dabs(lamm_nobetar))

    ! L-state per-mode speeds returned for the eigenvalue comparison.
    lam_l(1) = lamp_nobetal
    lam_l(2) = lamm_nobetal
    lam_l(3) = vxL

    ! ---- Auxiliary quantities ----
    cxx = one   ! gyy*gzz - gyz^2 (flat)
    cxy = 0.0d0
    cxz = 0.0d0
    gam = one   ! det of flat metric

    ! LEFT acoustic eigenvectors
    axpl = (u - vxL*vxL)/(u - vxL*lamp_nobetal)
    axml = (u - vxL*vxL)/(u - vxL*lamm_nobetal)
    vxpl = (vxL - lamp_nobetal)/(u - vxL * lamp_nobetal)
    vxml = (vxL - lamm_nobetal)/(u - vxL * lamm_nobetal)
    kappal = dpdepsL / (dpdepsL - rhoL * cs2L)
    xsil = cxx - gam * vxL * vxL
    dltl = enthalpyl**3 * wl * (kappal - one) * (vxml - vxpl) * xsil

    reivecp_l(1) = one
    reivecp_l(2) = enthalpyl * wl * (vlowxl - vxpl)
    reivecp_l(3) = enthalpyl * wl * vlowyl
    reivecp_l(4) = enthalpyl * wl * vlowzl
    reivecp_l(5) = enthalpyl * wl * axpl - one
    reivecm_l(1) = one
    reivecm_l(2) = enthalpyl * wl * (vlowxl - vxml)
    reivecm_l(3) = enthalpyl * wl * vlowyl
    reivecm_l(4) = enthalpyl * wl * vlowzl
    reivecm_l(5) = enthalpyl * wl * axml - one

    tmp1 = enthalpyl * enthalpyl / dltl
    tmp2 = wl * wl * xsil
    leivecp_l(1) = - (enthalpyl * wl * vxml * xsil + (one - kappal) * &
      (vxml * (tmp2 - cxx) - gam * vxL) - kappal * tmp2 * vxml) * tmp1
    leivecp_l(2) = - (cxx * (one - kappal * axml) + (two * kappal - one) * &
      vxml * (tmp2 * vxL - cxx * vxL)) * tmp1
    leivecp_l(3) = - (cxy * (one - kappal * axml) + (two * kappal - one) * &
      vxml * (tmp2 * vyL - cxy * vxL)) * tmp1
    leivecp_l(4) = - (cxz * (one - kappal * axml) + (two * kappal - one) * &
      vxml * (tmp2 * vzL - cxz * vxL)) * tmp1
    leivecp_l(5) = - ((one - kappal) * (vxml * (tmp2 - cxx) - gam * vxL) - &
      kappal * tmp2 * vxml) * tmp1
    leivecm_l(1) = (enthalpyl * wl * vxpl * xsil + (one - kappal) * &
      (vxpl * (tmp2 - cxx) - gam * vxL) - kappal * tmp2 * vxpl) * tmp1
    leivecm_l(2) = (cxx * (one - kappal * axpl) + (two * kappal - one) * &
      vxpl * (tmp2 * vxL - cxx * vxL)) * tmp1
    leivecm_l(3) = (cxy * (one - kappal * axpl) + (two * kappal - one) * &
      vxpl * (tmp2 * vyL - cxy * vxL)) * tmp1
    leivecm_l(4) = (cxz * (one - kappal * axpl) + (two * kappal - one) * &
      vxpl * (tmp2 * vzL - cxz * vxL)) * tmp1
    leivecm_l(5) = ((one - kappal) * (vxpl * (tmp2 - cxx) - gam * &
      vxL) - kappal * tmp2 * vxpl) * tmp1

    ! RIGHT acoustic left eigenvectors (needed for the FAST dissipation)
    axpr = (u - vxR*vxR)/(u - vxR*lamp_nobetar)
    axmr = (u - vxR*vxR)/(u - vxR*lamm_nobetar)
    vxpr = (vxR - lamp_nobetar)/(u - vxR * lamp_nobetar)
    vxmr = (vxR - lamm_nobetar)/(u - vxR * lamm_nobetar)
    kappar = dpdepsR / (dpdepsR - rhoR * cs2R)
    xsir = cxx - gam * vxR * vxR
    dltr = enthalpyr**3 * wr * (kappar - one) * (vxmr - vxpr) * xsir
    tmp1 = enthalpyr * enthalpyr / dltr
    tmp2 = wr * wr * xsir
    leivecpr(1) = - (enthalpyr * wr * vxmr * xsir + (one - kappar) * &
      (vxmr * (tmp2 - cxx) - gam * vxR) - kappar * tmp2 * vxmr) * tmp1
    leivecpr(2) = - (cxx * (one - kappar * axmr) + (two * kappar - one) * &
      vxmr * (tmp2 * vxR - cxx * vxR)) * tmp1
    leivecpr(3) = - (cxy * (one - kappar * axmr) + (two * kappar - one) * &
      vxmr * (tmp2 * vyR - cxy * vxR)) * tmp1
    leivecpr(4) = - (cxz * (one - kappar * axmr) + (two * kappar - one) * &
      vxmr * (tmp2 * vzR - cxz * vxR)) * tmp1
    leivecpr(5) = - ((one - kappar) * (vxmr * (tmp2 - cxx) - gam * vxR) - &
      kappar * tmp2 * vxmr) * tmp1
    leivecmr(1) = (enthalpyr * wr * vxpr * xsir + (one - kappar) * &
      (vxpr * (tmp2 - cxx) - gam * vxR) - kappar * tmp2 * vxpr) * tmp1
    leivecmr(2) = (cxx * (one - kappar * axpr) + (two * kappar - one) * &
      vxpr * (tmp2 * vxR - cxx * vxR)) * tmp1
    leivecmr(3) = (cxy * (one - kappar * axpr) + (two * kappar - one) * &
      vxpr * (tmp2 * vyR - cxy * vxR)) * tmp1
    leivecmr(4) = (cxz * (one - kappar * axpr) + (two * kappar - one) * &
      vxpr * (tmp2 * vzR - cxz * vxR)) * tmp1
    leivecmr(5) = ((one - kappar) * (vxpr * (tmp2 - cxx) - gam * &
      vxR) - kappar * tmp2 * vxpr) * tmp1

    ! ---- LEFT flux reconstruction (FAST), du = left conserved ----
    du(1) = densL; du(2) = sxL; du(3) = syL; du(4) = szL; du(5) = tauL
    sump = 0.0d0
    summ = 0.0d0
    do i = 1, 5
      sump = sump + (lamp - lam1) * leivecp_l(i) * du(i)
      summ = summ + (lamm - lam1) * leivecm_l(i) * du(i)
    end do
    vxa = sump + summ
    vxb = -(sump * vxpl + summ * vxml)
    rfluxl(1) = lam1 * du(1) + vxa
    rfluxl(2) = lam1 * du(2) + enthalpyl * wl * (vlowxl * vxa + vxb)
    rfluxl(3) = lam1 * du(3) + enthalpyl * wl * (vlowyl * vxa)
    rfluxl(4) = lam1 * du(4) + enthalpyl * wl * (vlowzl * vxa)
    rfluxl(5) = lam1 * du(5) + enthalpyl * wl * (vxL * vxb + vxa) - vxa

    ! ---- RIGHT flux reconstruction (FAST), du = right conserved ----
    du(1) = densR; du(2) = sxR; du(3) = syR; du(4) = szR; du(5) = tauR
    sump = 0.0d0
    summ = 0.0d0
    do i = 1, 5
      sump = sump + (lamp - lam1) * leivecpr(i) * du(i)
      summ = summ + (lamm - lam1) * leivecmr(i) * du(i)
    end do
    vxa = sump + summ
    vxb = -(sump * vxpr + summ * vxmr)
    rfluxr(1) = lam1 * du(1) + vxa
    rfluxr(2) = lam1 * du(2) + enthalpyr * wr * (vlowxr * vxa + vxb)
    rfluxr(3) = lam1 * du(3) + enthalpyr * wr * (vlowyr * vxa)
    rfluxr(4) = lam1 * du(4) + enthalpyr * wr * (vlowzr * vxa)
    rfluxr(5) = lam1 * du(5) + enthalpyr * wr * (vxR * vxb + vxa) - vxa

    do i = 1, 5
      diss(i) = rfluxr(i) - rfluxl(i)
    end do
  end subroutine whisky_marquina_diag

end module whisky_crosscheck
