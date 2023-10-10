# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-8.5.0/cmake-3.24.3-iy5kxxsrqsx5xom5uptyirzgbr36app4/bin/cmake

# The command to remove a file.
RM = /leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-8.5.0/cmake-3.24.3-iy5kxxsrqsx5xom5uptyirzgbr36app4/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /leonardo/home/userinternal/mredenti/test_kokkos/Exercises/ViewsAndSpaces

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /leonardo/home/userinternal/mredenti/test_kokkos/Exercises/ViewsAndSpaces

# Include any dependencies generated for this target.
include CMakeFiles/views.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/views.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/views.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/views.dir/flags.make

CMakeFiles/views.dir/views.cpp.o: CMakeFiles/views.dir/flags.make
CMakeFiles/views.dir/views.cpp.o: views.cpp
CMakeFiles/views.dir/views.cpp.o: CMakeFiles/views.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/leonardo/home/userinternal/mredenti/test_kokkos/Exercises/ViewsAndSpaces/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/views.dir/views.cpp.o"
	/leonardo/home/userinternal/mredenti/KOKKOS/install/kokkos_release/bin/nvcc_wrapper $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/views.dir/views.cpp.o -MF CMakeFiles/views.dir/views.cpp.o.d -o CMakeFiles/views.dir/views.cpp.o -c /leonardo/home/userinternal/mredenti/test_kokkos/Exercises/ViewsAndSpaces/views.cpp

CMakeFiles/views.dir/views.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/views.dir/views.cpp.i"
	/leonardo/home/userinternal/mredenti/KOKKOS/install/kokkos_release/bin/nvcc_wrapper $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /leonardo/home/userinternal/mredenti/test_kokkos/Exercises/ViewsAndSpaces/views.cpp > CMakeFiles/views.dir/views.cpp.i

CMakeFiles/views.dir/views.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/views.dir/views.cpp.s"
	/leonardo/home/userinternal/mredenti/KOKKOS/install/kokkos_release/bin/nvcc_wrapper $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /leonardo/home/userinternal/mredenti/test_kokkos/Exercises/ViewsAndSpaces/views.cpp -o CMakeFiles/views.dir/views.cpp.s

# Object files for target views
views_OBJECTS = \
"CMakeFiles/views.dir/views.cpp.o"

# External object files for target views
views_EXTERNAL_OBJECTS =

views: CMakeFiles/views.dir/views.cpp.o
views: CMakeFiles/views.dir/build.make
views: /leonardo/home/userinternal/mredenti/KOKKOS/install/kokkos_release/lib64/libkokkoscontainers.a
views: /leonardo/home/userinternal/mredenti/KOKKOS/install/kokkos_release/lib64/libkokkoscore.a
views: /usr/lib64/libcuda.so
views: /leonardo/prod/opt/compilers/cuda/11.8/none/lib64/libcudart.so
views: /leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-8.5.0/gcc-11.3.0-tm6phj7wkcw7cuy6gjixemkvh5x2mhza/lib64/libgomp.so
views: /leonardo/home/userinternal/mredenti/KOKKOS/install/kokkos_release/lib64/libkokkossimd.a
views: CMakeFiles/views.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/leonardo/home/userinternal/mredenti/test_kokkos/Exercises/ViewsAndSpaces/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable views"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/views.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/views.dir/build: views
.PHONY : CMakeFiles/views.dir/build

CMakeFiles/views.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/views.dir/cmake_clean.cmake
.PHONY : CMakeFiles/views.dir/clean

CMakeFiles/views.dir/depend:
	cd /leonardo/home/userinternal/mredenti/test_kokkos/Exercises/ViewsAndSpaces && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /leonardo/home/userinternal/mredenti/test_kokkos/Exercises/ViewsAndSpaces /leonardo/home/userinternal/mredenti/test_kokkos/Exercises/ViewsAndSpaces /leonardo/home/userinternal/mredenti/test_kokkos/Exercises/ViewsAndSpaces /leonardo/home/userinternal/mredenti/test_kokkos/Exercises/ViewsAndSpaces /leonardo/home/userinternal/mredenti/test_kokkos/Exercises/ViewsAndSpaces/CMakeFiles/views.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/views.dir/depend

