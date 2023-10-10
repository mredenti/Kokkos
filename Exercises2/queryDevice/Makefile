# Set the paths to Kokkos library and headers
KOKKOS_PATH=$(WORK)/mredenti/Software/KOKKOS/kokkos-4.0.00/cudaompinstall
KOKKOS_INCLUDE_PATH = $(KOKKOS_PATH)/include
KOKKOS_LIB_PATH = $(KOKKOS_PATH)/lib64


MAKEFILE_PATH := $(subst Makefile,,$(abspath $(lastword $(MAKEFILE_LIST))))

SRC = $(wildcard $(MAKEFILE_PATH)*.cpp)
HEADERS = $(wildcard $(MAKEFILE_PATH)*.hpp)

default: build
	echo "Start Build"

CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper 
EXE = query_device.x

CXXFLAGS ?= -O3 -g -fopenmp 
override CXXFLAGS += -I$(MAKEFILE_PATH) -I$(KOKKOS_INCLUDE_PATH) 

ifeq ($(USE_MPI), 1)
    override CXXFLAGS += -DUSE_MPI
	CXX += -ccbin $(shell which mpicxx)
endif

DEPFLAGS = -M
LINK = ${CXX}
LINKFLAGS =

OBJ = $(notdir $(SRC:.cpp=.o))
LIB = -L$(KOKKOS_LIB_PATH) -lkokkoscontainers -lkokkoscore -lkokkossimd

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(LINKFLAGS) $(CXXFLAGS) $(OBJ) $(LIB) -o $(EXE) 

clean: 
	rm -f *.o *.cuda *.host *.x

# Compilation rules
%.o:%.cpp $(HEADERS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)