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
