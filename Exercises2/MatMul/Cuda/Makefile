MAKEFILE_PATH := $(subst Makefile,,$(abspath $(lastword $(MAKEFILE_LIST))))

SRC = $(wildcard $(MAKEFILE_PATH)*.cu)
HEADERS = $(wildcard $(MAKEFILE_PATH)*.hpp)

default: build
	echo "Start Build"

CC = nvcc
EXE = outer_loop.cux

CFLAGS ?= -O2 -g 
override CXXFLAGS += -I$(MAKEFILE_PATH) 

DEPFLAGS = -M
LINK = 
LINKFLAGS =

OBJ = $(notdir $(SRC:.cu=.o))
LIB =

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(CC) $(LINKFLAGS) $(CFLAGS) $(OBJ) $(LIB) -o $(EXE) 

clean: 
	rm -f *.o *.cuda *.host *.x

# Compilation rules
%.o:%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)

run:
	OMP_PROC_BIND=spread OMP_PLACES=threads ./$(EXE)