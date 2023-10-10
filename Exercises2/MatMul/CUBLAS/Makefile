
EXE = matrixMulCUBLAS

run:
	OMP_NUM_THREADS=1 ./$(EXE)

nvprof:
	OMP_NUM_THREADS=1 nsys nvprof ./$(EXE)	
