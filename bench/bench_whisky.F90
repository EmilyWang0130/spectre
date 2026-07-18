! Distributed under the MIT License.
! See LICENSE.txt for details.
!
! Standalone microbenchmark of the WHISKY Marquina per-interface kernel
! `eigenproblem_marquina_general` (Elias Most's reference), extracted for the
! production configuration ANALYTICAL=.TRUE., FAST=.TRUE. (Aloy et al. 1999 CPC
! shortcut). The kernel math is transcribed verbatim from
! whisky_reference/Whisky_Eigenproblem_Marquina.F90; only the Cactus scaffolding
! (cctk headers, Whisky_Scalars module, CCTK_REAL typedef) is stripped so the
! file compiles standalone with gfortran.
!
! WHISKY-Marquina is hydro-only (5 conserved vars: D, S_i, tau). We time the
! kernel over N random, physically valid left/right state pairs in the same
! hydro-only, flat-metric, ideal-gas (Gamma=2) limit as the SpECTRE harness.
!
! NOTE: this harness is NOT bit-for-bit input-identical to the C++ harness
! (std::mt19937 vs Fortran's PRNG differ); it draws from the same distributions
! and same N so the *throughput* (ns/interface) is an apples-to-apples measure
! of the two kernels. Correctness of the SpECTRE side is guarded separately by
! bench_marquina against reference_output.dat.

module marquina_kernel
  implicit none
  integer, parameter :: dp = kind(1.0d0)
contains

  ! Production WHISKY Marquina kernel: ANALYTICAL=.TRUE., FAST=.TRUE.
  ! Returns the upwind dissipation correction flux(1:5) = rfluxr - rfluxl.
  subroutine eigenproblem_marquina_general_fast( &
       rhor, velxr, velyr, velzr, epsr, pressr, cs2r, dpdepsr, &
       rhol, velxl, velyl, velzl, epsl, pressl, cs2l, dpdepsl, &
       gxx, gxy, gxz, gyy, gyz, gzz, u, alp, beta, &
       densl, sxl, syl, szl, taul, &
       densr, sxr, syr, szr, taur, &
       flux1, flux2, flux3, flux4, flux5)
    real(dp), intent(in) :: rhor, velxr, velyr, velzr, epsr
    real(dp), intent(in) :: pressr, cs2r, dpdepsr
    real(dp), intent(in) :: rhol, velxl, velyl, velzl, epsl
    real(dp), intent(in) :: pressl, cs2l, dpdepsl
    real(dp), intent(in) :: gxx, gxy, gxz, gyy, gyz, gzz, u, alp, beta
    real(dp), intent(in) :: densl, sxl, syl, szl, taul
    real(dp), intent(in) :: densr, sxr, syr, szr, taur
    real(dp), intent(out) :: flux1, flux2, flux3, flux4, flux5

    real(dp) :: lam(5), du(5)
    real(dp) :: rfluxr(5), rfluxl(5)
    real(dp) :: one, two
    real(dp) :: vlowxr, vlowyr, vlowzr, v2r, wr
    real(dp) :: vlowxl, vlowyl, vlowzl, v2l, wl
    real(dp) :: enthalpyl, kappal, enthalpyr, kappar
    real(dp) :: axpl, axml, vxpl, vxml, xsil, dltl
    real(dp) :: axpr, axmr, vxpr, vxmr, xsir, dltr
    real(dp) :: cxx, cxy, cxz, gam
    real(dp) :: lam1l, lamml, lampl, lamm_nobetal, lamp_nobetal
    real(dp) :: lam1r, lammr, lampr, lamm_nobetar, lamp_nobetar
    real(dp) :: lam1, lamm, lamp
    real(dp) :: leivecpl(5), leivecml(5), leivecpr(5), leivecmr(5)
    real(dp) :: tmp1, tmp2, sump, summ, vxa, vxb
    integer :: i

    one = 1.0d0
    two = 2.0d0

    ! ---- LEFT fluid quantities ----
    enthalpyl = one + epsl + pressl / rhol
    vlowxl = gxx*velxl + gxy*velyl + gxz*velzl
    vlowyl = gxy*velxl + gyy*velyl + gyz*velzl
    vlowzl = gxz*velxl + gyz*velyl + gzz*velzl
    v2l = vlowxl*velxl + vlowyl*velyl + vlowzl*velzl
    wl = one / sqrt(one - v2l)

    lam1l = velxl - beta/alp
    lamp_nobetal = (velxl*(one-cs2l) + sqrt(cs2l*(one-v2l)* &
         (u*(one-v2l*cs2l) - velxl**2*(one-cs2l))))/(one-v2l*cs2l)
    lamm_nobetal = (velxl*(one-cs2l) - sqrt(cs2l*(one-v2l)* &
         (u*(one-v2l*cs2l) - velxl**2*(one-cs2l))))/(one-v2l*cs2l)
    lampl = lamp_nobetal - beta/alp
    lamml = lamm_nobetal - beta/alp

    ! ---- RIGHT fluid quantities ----
    enthalpyr = one + epsr + pressr / rhor
    vlowxr = gxx*velxr + gxy*velyr + gxz*velzr
    vlowyr = gxy*velxr + gyy*velyr + gyz*velzr
    vlowzr = gxz*velxr + gyz*velyr + gzz*velzr
    v2r = vlowxr*velxr + vlowyr*velyr + vlowzr*velzr
    wr = one / sqrt(one - v2r)

    lam1r = velxr - beta/alp
    lamp_nobetar = (velxr*(one-cs2r) + sqrt(cs2r*(one-v2r)* &
         (u*(one-v2r*cs2r) - velxr**2*(one-cs2r))))/(one-v2r*cs2r)
    lamm_nobetar = (velxr*(one-cs2r) - sqrt(cs2r*(one-v2r)* &
         (u*(one-v2r*cs2r) - velxr**2*(one-cs2r))))/(one-v2r*cs2r)
    lampr = lamp_nobetar - beta/alp
    lammr = lamm_nobetar - beta/alp

    ! ---- FINAL (componentwise max abs) ----
    lam1 = dmax1(dabs(lam1l), dabs(lam1r))
    lamp = dmax1(dabs(lampl), dabs(lampr))
    lamm = dmax1(dabs(lamml), dabs(lammr))
    lam(1) = lamm
    lam(5) = lam1
    lam(3) = lam1
    lam(4) = lam1
    lam(2) = lamp

    ! ---- Auxiliary quantities (ANALYTICAL) ----
    cxx = gyy * gzz - gyz * gyz
    cxy = gxz * gyz - gxy * gzz
    cxz = gxy * gyz - gxz * gyy
    gam = gxx * cxx + gxy * cxy + gxz * cxz

    ! LEFT acoustic left-eigenvectors
    axpl = (u - velxl*velxl)/(u - velxl*lamp_nobetal)
    axml = (u - velxl*velxl)/(u - velxl*lamm_nobetal)
    vxpl = (velxl - lamp_nobetal)/(u - velxl * lamp_nobetal)
    vxml = (velxl - lamm_nobetal)/(u - velxl * lamm_nobetal)
    kappal = dpdepsl / (dpdepsl - rhol * cs2l)
    xsil = cxx - gam * velxl * velxl
    dltl = enthalpyl**3 * wl * (kappal - one) * (vxml - vxpl) * xsil
    tmp1 = enthalpyl * enthalpyl / dltl
    tmp2 = wl * wl * xsil
    leivecpl(1) = - (enthalpyl * wl * vxml * xsil + (one - kappal) * &
      (vxml * (tmp2 - cxx) - gam * velxl) - kappal * tmp2 * vxml) * tmp1
    leivecpl(2) = - (cxx * (one - kappal * axml) + (two * kappal - one) * &
      vxml * (tmp2 * velxl - cxx * velxl)) * tmp1
    leivecpl(3) = - (cxy * (one - kappal * axml) + (two * kappal - one) * &
      vxml * (tmp2 * velyl - cxy * velxl)) * tmp1
    leivecpl(4) = - (cxz * (one - kappal * axml) + (two * kappal - one) * &
      vxml * (tmp2 * velzl - cxz * velxl)) * tmp1
    leivecpl(5) = - ((one - kappal) * (vxml * (tmp2 - cxx) - gam * velxl) - &
      kappal * tmp2 * vxml) * tmp1
    leivecml(1) = (enthalpyl * wl * vxpl * xsil + (one - kappal) * &
      (vxpl * (tmp2 - cxx) - gam * velxl) - kappal * tmp2 * vxpl) * tmp1
    leivecml(2) = (cxx * (one - kappal * axpl) + (two * kappal - one) * &
      vxpl * (tmp2 * velxl - cxx * velxl)) * tmp1
    leivecml(3) = (cxy * (one - kappal * axpl) + (two * kappal - one) * &
      vxpl * (tmp2 * velyl - cxy * velxl)) * tmp1
    leivecml(4) = (cxz * (one - kappal * axpl) + (two * kappal - one) * &
      vxpl * (tmp2 * velzl - cxz * velxl)) * tmp1
    leivecml(5) = ((one - kappal) * (vxpl * (tmp2 - cxx) - gam * &
      velxl) - kappal * tmp2 * vxpl) * tmp1

    ! RIGHT acoustic left-eigenvectors
    axpr = (u - velxr*velxr)/(u - velxr*lamp_nobetar)
    axmr = (u - velxr*velxr)/(u - velxr*lamm_nobetar)
    vxpr = (velxr - lamp_nobetar)/(u - velxr * lamp_nobetar)
    vxmr = (velxr - lamm_nobetar)/(u - velxr * lamm_nobetar)
    kappar = dpdepsr / (dpdepsr - rhor * cs2r)
    xsir = cxx - gam * velxr * velxr
    dltr = enthalpyr**3 * wr * (kappar - one) * (vxmr - vxpr) * xsir
    tmp1 = enthalpyr * enthalpyr / dltr
    tmp2 = wr * wr * xsir
    leivecpr(1) = - (enthalpyr * wr * vxmr * xsir + (one - kappar) * &
      (vxmr * (tmp2 - cxx) - gam * velxr) - kappar * tmp2 * vxmr) * tmp1
    leivecpr(2) = - (cxx * (one - kappar * axmr) + (two * kappar - one) * &
      vxmr * (tmp2 * velxr - cxx * velxr)) * tmp1
    leivecpr(3) = - (cxy * (one - kappar * axmr) + (two * kappar - one) * &
      vxmr * (tmp2 * velyr - cxy * velxr)) * tmp1
    leivecpr(4) = - (cxz * (one - kappar * axmr) + (two * kappar - one) * &
      vxmr * (tmp2 * velzr - cxz * velxr)) * tmp1
    leivecpr(5) = - ((one - kappar) * (vxmr * (tmp2 - cxx) - gam * velxr) - &
      kappar * tmp2 * vxmr) * tmp1
    leivecmr(1) = (enthalpyr * wr * vxpr * xsir + (one - kappar) * &
      (vxpr * (tmp2 - cxx) - gam * velxr) - kappar * tmp2 * vxpr) * tmp1
    leivecmr(2) = (cxx * (one - kappar * axpr) + (two * kappar - one) * &
      vxpr * (tmp2 * velxr - cxx * velxr)) * tmp1
    leivecmr(3) = (cxy * (one - kappar * axpr) + (two * kappar - one) * &
      vxpr * (tmp2 * velyr - cxy * velxr)) * tmp1
    leivecmr(4) = (cxz * (one - kappar * axpr) + (two * kappar - one) * &
      vxpr * (tmp2 * velzr - cxz * velxr)) * tmp1
    leivecmr(5) = ((one - kappar) * (vxpr * (tmp2 - cxx) - gam * &
      velxr) - kappar * tmp2 * vxpr) * tmp1

    ! ---- LEFT flux reconstruction (FAST path) ----
    du(1) = densl
    du(2) = sxl
    du(3) = syl
    du(4) = szl
    du(5) = taul
    sump = 0.0d0
    summ = 0.0d0
    do i = 1, 5
      sump = sump + (lamp - lam1) * leivecpl(i) * du(i)
      summ = summ + (lamm - lam1) * leivecml(i) * du(i)
    end do
    vxa = sump + summ
    vxb = -(sump * vxpl + summ * vxml)
    rfluxl(1) = lam1 * du(1) + vxa
    rfluxl(2) = lam1 * du(2) + enthalpyl * wl * (vlowxl * vxa + vxb)
    rfluxl(3) = lam1 * du(3) + enthalpyl * wl * (vlowyl * vxa)
    rfluxl(4) = lam1 * du(4) + enthalpyl * wl * (vlowzl * vxa)
    rfluxl(5) = lam1 * du(5) + enthalpyl * wl * (velxl * vxb + vxa) - vxa

    ! ---- RIGHT flux reconstruction (FAST path) ----
    du(1) = densr
    du(2) = sxr
    du(3) = syr
    du(4) = szr
    du(5) = taur
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
    rfluxr(5) = lam1 * du(5) + enthalpyr * wr * (velxr * vxb + vxa) - vxa

    flux1 = rfluxr(1) - rfluxl(1)
    flux2 = rfluxr(2) - rfluxl(2)
    flux3 = rfluxr(3) - rfluxl(3)
    flux4 = rfluxr(4) - rfluxl(4)
    flux5 = rfluxr(5) - rfluxl(5)
  end subroutine eigenproblem_marquina_general_fast

end module marquina_kernel


program bench_whisky
  use marquina_kernel
  implicit none

  integer, parameter :: N = 1000000
  real(dp), parameter :: gamma_eos = 2.0d0

  ! Stored primitive state pairs.
  real(dp), allocatable :: rhoL(:), vxL(:), vyL(:), vzL(:), epsL(:)
  real(dp), allocatable :: rhoR(:), vxR(:), vyR(:), vzR(:), epsR(:)
  ! Derived conserved states.
  real(dp), allocatable :: dL(:), sxL(:), syL(:), szL(:), tauL(:)
  real(dp), allocatable :: dR(:), sxR(:), syR(:), szR(:), tauR(:)
  real(dp), allocatable :: pL(:), csL(:), dpL(:)
  real(dp), allocatable :: pR(:), csR(:), dpR(:)

  integer, allocatable :: seed(:)
  integer :: nseed, k, rep
  real(dp) :: r(5)
  real(dp) :: h, w, v2, vlx, vly, vlz, p, cs2, dpde
  real(dp) :: f1, f2, f3, f4, f5, checksum
  integer(kind=8) :: c0, c1, crate
  real(dp) :: best_ns, ns
  integer, parameter :: NREP = 3

  allocate(rhoL(N), vxL(N), vyL(N), vzL(N), epsL(N))
  allocate(rhoR(N), vxR(N), vyR(N), vzR(N), epsR(N))
  allocate(dL(N), sxL(N), syL(N), szL(N), tauL(N))
  allocate(dR(N), sxR(N), syR(N), szR(N), tauR(N))
  allocate(pL(N), csL(N), dpL(N), pR(N), csR(N), dpR(N))

  ! Deterministic seed.
  call random_seed(size=nseed)
  allocate(seed(nseed))
  seed = 20240716
  call random_seed(put=seed)

  ! Generate states: rho in [0.1,10], v_i in [-0.3,0.3], eps in [0.01,1.0].
  do k = 1, N
    call random_number(r)
    rhoL(k) = 0.1d0 + r(1) * (10.0d0 - 0.1d0)
    vxL(k)  = -0.3d0 + r(2) * 0.6d0
    vyL(k)  = -0.3d0 + r(3) * 0.6d0
    vzL(k)  = -0.3d0 + r(4) * 0.6d0
    epsL(k) = 0.01d0 + r(5) * (1.0d0 - 0.01d0)
    call random_number(r)
    rhoR(k) = 0.1d0 + r(1) * (10.0d0 - 0.1d0)
    vxR(k)  = -0.3d0 + r(2) * 0.6d0
    vyR(k)  = -0.3d0 + r(3) * 0.6d0
    vzR(k)  = -0.3d0 + r(4) * 0.6d0
    epsR(k) = 0.01d0 + r(5) * (1.0d0 - 0.01d0)
  end do

  ! Derive EOS quantities and conserved variables (flat metric, alpha=1,
  ! beta=0). Ideal gas: p=(G-1) rho eps, dp/deps=(G-1) rho, h=1+eps+p/rho,
  ! cs2 = (G-1) p / (rho h) * G  (== chi + p/rho^2 kappa, all over h).
  do k = 1, N
    ! LEFT
    p = (gamma_eos - 1.0d0) * rhoL(k) * epsL(k)
    dpde = (gamma_eos - 1.0d0) * rhoL(k)
    h = 1.0d0 + epsL(k) + p / rhoL(k)
    cs2 = ((gamma_eos - 1.0d0) * epsL(k) + p / (rhoL(k)*rhoL(k)) * dpde) / h
    pL(k) = p; csL(k) = cs2; dpL(k) = dpde
    vlx = vxL(k); vly = vyL(k); vlz = vzL(k)
    v2 = vlx*vlx + vly*vly + vlz*vlz
    w = 1.0d0 / sqrt(1.0d0 - v2)
    dL(k)   = rhoL(k) * w
    sxL(k)  = rhoL(k) * h * w*w * vlx
    syL(k)  = rhoL(k) * h * w*w * vly
    szL(k)  = rhoL(k) * h * w*w * vlz
    tauL(k) = rhoL(k) * h * w*w - p - dL(k)
    ! RIGHT
    p = (gamma_eos - 1.0d0) * rhoR(k) * epsR(k)
    dpde = (gamma_eos - 1.0d0) * rhoR(k)
    h = 1.0d0 + epsR(k) + p / rhoR(k)
    cs2 = ((gamma_eos - 1.0d0) * epsR(k) + p / (rhoR(k)*rhoR(k)) * dpde) / h
    pR(k) = p; csR(k) = cs2; dpR(k) = dpde
    vlx = vxR(k); vly = vyR(k); vlz = vzR(k)
    v2 = vlx*vlx + vly*vly + vlz*vlz
    w = 1.0d0 / sqrt(1.0d0 - v2)
    dR(k)   = rhoR(k) * w
    sxR(k)  = rhoR(k) * h * w*w * vlx
    syR(k)  = rhoR(k) * h * w*w * vly
    szR(k)  = rhoR(k) * h * w*w * vlz
    tauR(k) = rhoR(k) * h * w*w - p - dR(k)
  end do

  call system_clock(count_rate=crate)
  best_ns = huge(1.0d0)

  do rep = 1, NREP
    checksum = 0.0d0
    call system_clock(c0)
    do k = 1, N
      call eigenproblem_marquina_general_fast( &
           rhoR(k), vxR(k), vyR(k), vzR(k), epsR(k), pR(k), csR(k), dpR(k), &
           rhoL(k), vxL(k), vyL(k), vzL(k), epsL(k), pL(k), csL(k), dpL(k), &
           1.0d0, 0.0d0, 0.0d0, 1.0d0, 0.0d0, 1.0d0, 1.0d0, 1.0d0, 0.0d0, &
           dL(k), sxL(k), syL(k), szL(k), tauL(k), &
           dR(k), sxR(k), syR(k), szR(k), tauR(k), &
           f1, f2, f3, f4, f5)
      checksum = checksum + f1 + f2 + f3 + f4 + f5
    end do
    call system_clock(c1)
    ns = real(c1 - c0, dp) / real(crate, dp) * 1.0d9 / real(N, dp)
    if (ns < best_ns) best_ns = ns
  end do

  write(*,'(A,I0)')      'WHISKY N            = ', N
  write(*,'(A,ES16.8)')  'WHISKY checksum     = ', checksum
  write(*,'(A,F12.4)')   'WHISKY ns/interface = ', best_ns
  write(*,'(A,F12.4)')   'WHISKY_NS_PER_CALL ', best_ns

  deallocate(rhoL, vxL, vyL, vzL, epsL, rhoR, vxR, vyR, vzR, epsR)
  deallocate(dL, sxL, syL, szL, tauL, dR, sxR, syR, szR, tauR)
  deallocate(pL, csL, dpL, pR, csR, dpR, seed)
end program bench_whisky
