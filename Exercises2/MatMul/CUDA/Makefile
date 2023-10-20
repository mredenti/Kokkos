MAKEFILE_PATH := $(subst Makefile,,$(abspath $(lastword $(MAKEFILE_LIST))))

SRC = $(wildcard $(MAKEFILE_PATH)*.cu)
HEADERS = $(wildcard $(MAKEFILE_PATH)*.hpp)

default: build
	echo "Start Build"

EXE = main.cux

HOST_COMPILER ?= g++
NVCC          := $(CUDA_HOME)/bin/nvcc -ccbin $(HOST_COMPILER)
NVCCFLAGS ?= -O2 -g 
override NVCCFLAGS += -I$(MAKEFILE_PATH) -I$(CUDA_HOME)/samples/Common

DEPFLAGS = -M
LINK = 
LINKFLAGS =

OBJ = $(notdir $(SRC:.cu=.o))
LIB =

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(NVCC) $(LINKFLAGS) $(NVCCFLAGS) $(OBJ) $(LIB) -o $(EXE) 

clean: 
	rm -f *.o *.cuda *.host *.x

# Compilation rules
%.o:%.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)

run:
	OMP_PROC_BIND=spread OMP_PLACES=threads ./$(EXE)