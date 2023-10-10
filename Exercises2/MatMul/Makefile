# Set the paths to Kokkos library and headers
KOKKOS_PATH=$(WORK)/mredenti/Software/KOKKOS/kokkos-4.0.00/cudaompinstall
KOKKOS_INCLUDE_PATH = $(KOKKOS_PATH)/include
KOKKOS_LIB_PATH = $(KOKKOS_PATH)/lib64

# KERNELS
# DOUBLE_LOOP = 

MAKEFILE_PATH := $(subst Makefile,,$(abspath $(lastword $(MAKEFILE_LIST))))

SRC = $(wildcard $(MAKEFILE_PATH)main.cpp)
HEADERS = $(wildcard $(MAKEFILE_PATH)*.hpp)

default: build
	echo "Start Build"

CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper --extended-lambda
EXE = main.x

CXXFLAGS ?= -O3 -g -fopenmp
override CXXFLAGS += -I$(MAKEFILE_PATH) -I$(KOKKOS_INCLUDE_PATH) 

DEPFLAGS = -M
LINK = ${CXX}
LINKFLAGS =

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(LINKFLAGS) $(CXXFLAGS) $(OBJ) -L$(KOKKOS_LIB_PATH) -lkokkoscontainers -lkokkoscore -lkokkossimd $(LIB) -o $(EXE) 

clean: 
	rm -f *.o *.cuda *.host *.x

# Compilation rules
%.o:%.cpp $(HEADERS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)

run:
	OMP_PROC_BIND=spread OMP_PLACES=threads ./$(EXE)

profile:
	OMP_PROC_BIND=spread OMP_PLACES=threads nsys nvprof -o ./nvidia/ ./$(EXE)