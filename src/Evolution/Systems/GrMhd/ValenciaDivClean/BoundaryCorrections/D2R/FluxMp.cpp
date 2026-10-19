// Distributed under the MIT License.
// See LICENSE.txt for details.

// The 160-digit oracle build of the D2-R kernel: the same source compiled with
// D2R_MULTIPRECISION into namespace d2r_detail_mp (see Types.hpp). Selected at
// run time by SPECTRE_D2R_PRECISION=mp; the default kernel is the double build.
#define D2R_MULTIPRECISION 1
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/D2R/Flux.cpp"
