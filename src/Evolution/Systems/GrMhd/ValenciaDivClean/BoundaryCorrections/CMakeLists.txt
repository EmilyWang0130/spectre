#
# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Hll.cpp
  Hllc.cpp
  HllcGr.cpp
  Hlld.cpp
  Hllem.cpp
  HllemHydroYe.cpp
  Marquina.cpp
  Rusanov.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  Hll.hpp
  Hllc.hpp
  HllcGr.hpp
  Hlld.hpp
  HlldImpl.hpp
  Hllem.hpp
  HllemHydroYe.hpp
  Marquina.hpp
  NamespaceDocs.hpp
  Rusanov.hpp
  )
