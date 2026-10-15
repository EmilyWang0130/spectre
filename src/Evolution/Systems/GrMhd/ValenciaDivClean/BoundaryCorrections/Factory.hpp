// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hllc.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/HllcGr.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hlld.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hllem.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/HllemHydroYe.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Marquina.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoD2R.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoHllc.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoHlld.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoHllx.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
using standard_boundary_corrections =
    tmpl::list<Hll, Hllc, HllcGr, Hlld, Hllem, HllemHydroYe, Marquina,
               PlutoD2R, PlutoHllc, PlutoHlld, PlutoHllx, Rusanov>;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
