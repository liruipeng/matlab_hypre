HYPRE_DIR=../hypre
HYPRE_INC = -I${HYPRE_DIR}/include
HYPRE_LIB = -L${HYPRE_DIR}/lib -lHYPRE

default:
	mex -O -largeArrayDims $(HYPRE_INC) hypre_amg_setup.cpp -DLINUX -DUNIX $(HYPRE_LIB)

clean:
	rm -f hypre_amg_setup.mexa64
