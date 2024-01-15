HYPRE_DIR=../hypre_v2290/src/hypre
EIGEN_DIR=/usr/local/include/eigen3/Eigen

HYPRE_INC = -I${HYPRE_DIR}/include
HYPRE_LIB = -L${HYPRE_DIR}/lib -lHYPRE
EIGEN_INC = -I${EIGEN_DIR}

default:
	mex -O -largeArrayDims $(HYPRE_INC) hypre_amg_setup.cpp -DLINUX -DUNIX $(HYPRE_LIB)
	mex -O -largeArrayDims $(EIGEN_INC) eigen_ProjSSOR.cpp -DLINUX -DUNIX 

clean:
	rm -f hypre_amg_setup.mexmaci64
	rm -f eigen_ProjSSOR.mexmaci64
