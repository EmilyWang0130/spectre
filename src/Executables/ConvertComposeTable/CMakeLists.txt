# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE ConvertComposeTable)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  ConvertComposeTable.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Boost::boost
  Boost::program_options
  DataStructures
  H5
  Hydro
  Informer
  IO
  Printf
  )

# 1D beta-equilibrium slice converter (CompOSE eos.beta -> SpECTRE 1D EosTable).
# Produces an h5 readable by EquationsOfState::Tabulated1D.
set(EXECUTABLE_BETA_1D ConvertComposeBetaTo1D)

add_spectre_executable(
  ${EXECUTABLE_BETA_1D}
  EXCLUDE_FROM_ALL
  ConvertComposeBetaTo1D.cpp
  )

target_link_libraries(
  ${EXECUTABLE_BETA_1D}
  PRIVATE
  Boost::boost
  Boost::program_options
  DataStructures
  H5
  Hydro
  Informer
  IO
  Printf
  )

# Legacy converter — a verbatim copy of the original ConvertComposeTable kept
# as a fallback. The "live" ConvertComposeTable above replaces CompOSE Q12
# (sound speed squared) with a finite-difference reconstruction from pressure
# and entropy; if a downstream consumer needs the unmodified Q12 it can build
# this target instead.
set(EXECUTABLE_LEGACY ConvertComposeTableLegacy)

add_spectre_executable(
  ${EXECUTABLE_LEGACY}
  EXCLUDE_FROM_ALL
  ConvertComposeTableLegacy.cpp
  )

target_link_libraries(
  ${EXECUTABLE_LEGACY}
  PRIVATE
  Boost::boost
  Boost::program_options
  DataStructures
  H5
  Informer
  IO
  Printf
  )
