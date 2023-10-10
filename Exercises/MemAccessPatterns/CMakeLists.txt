# Kokkos minimally requires 3.16 right now,
# but your project can set it higher
cmake_minimum_required (VERSION 3.16)

project (KokkosExercisesMemAccessPatterns)

# Look for an installed Kokkos
#find_package(KOKKOS REQUIRED)
include(../common.cmake)

add_executable(layout layout.cpp)
target_link_libraries(layout Kokkos::kokkos)

# TO BUILD, RUN:
# cmake . -DKokkos_ROOT=$HOME/KOKKOS/install/kokkos_release/lib64/cmake/Kokkos -DCMAKE_CXX_COMPILER=$HOME/KOKKOS/install/kokkos_release/bin/nvcc_wrapper -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON