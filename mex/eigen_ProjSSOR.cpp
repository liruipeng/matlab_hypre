/* From matlab call
$ mex projectedSSOR.cpp -I/usr/local/include/eigen3/Eigen
*/
#include "matrix.h"
#include "mex.h"
#include <Eigen>
#include <iostream>
#include <limits>
#include <type_traits>

// -----------------------------------------------------------------------------
typedef Eigen::SparseMatrix<double, Eigen::ColMajor,
                            std::make_signed<mwIndex>::type>
    MatlabSparse;

// -----------------------------------------------------------------------------
Eigen::Map<MatlabSparse> matlab_to_eigen_sparse(const mxArray *mat) {
  mxAssert(mxGetClassID(mat) == mxDOUBLE_CLASS,
           "Type of the input matrix isn't double");
  mwSize m = mxGetM(mat);
  mwSize n = mxGetN(mat);
  mwSize nz = mxGetNzmax(mat);

  /*Theoretically fails in very very large matrices*/
  mxAssert(nz <= std::numeric_limits<std::make_signed<mwIndex>::type>::max(),
           "Unsupported Data size.");

  double *pr = mxGetPr(mat);

  MatlabSparse::StorageIndex *ir =
      reinterpret_cast<MatlabSparse::StorageIndex *>(mxGetIr(mat));
  MatlabSparse::StorageIndex *jc =
      reinterpret_cast<MatlabSparse::StorageIndex *>(mxGetJc(mat));

  Eigen::Map<MatlabSparse> result(m, n, nz, jc, ir, pr);

  return result;
}

// -----------------------------------------------------------------------------
mxArray *eigen_to_matlab_sparse(
    const Eigen::Ref<const MatlabSparse, Eigen::StandardCompressedFormat>
        &mat) {
  mxArray *result =
      mxCreateSparse(mat.rows(), mat.cols(), mat.nonZeros(), mxREAL);
  const MatlabSparse::StorageIndex *ir = mat.innerIndexPtr();
  const MatlabSparse::StorageIndex *jc = mat.outerIndexPtr();
  const double *pr = mat.valuePtr();

  mwIndex *ir2 = mxGetIr(result);
  mwIndex *jc2 = mxGetJc(result);
  double *pr2 = mxGetPr(result);

  for (mwIndex i = 0; i < mat.nonZeros(); i++) {
    pr2[i] = pr[i];
    ir2[i] = ir[i];
  }
  for (mwIndex i = 0; i < mat.cols() + 1; i++) {
    jc2[i] = jc[i];
  }
  return result;
}

// -----------------------------------------------------------------------------
/* the gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // ---------------------------------------------------------------------------
  /* Check for proper number of input and output arguments */
  if (nrhs != 9) {
    mexErrMsgIdAndTxt("MATLAB:mxislogical:invalidNumInputs",
                      "One input argument required.");
  }

  if (!(mxIsSparse(prhs[0]))) {
    mexErrMsgIdAndTxt("MATLAB: ", "First input matrix is not sparse.\n");
  }

  // ---------------------------------------------------------------------------
  // figure out dimensions of A
  const size_t mrowsA = mxGetM(prhs[0]);
  const size_t ncolsA = mxGetN(prhs[0]);

  // figure out dimensions of b
  const size_t mrowsb = mxGetM(prhs[1]);
  const size_t ncolsb = mxGetN(prhs[1]);

  // figure out dimensions of x
  const size_t mrowsx = mxGetM(prhs[2]);
  const size_t ncolsx = mxGetN(prhs[2]);

  // figure out dimensions of lb
  const size_t mrowslb = mxGetM(prhs[3]);
  const size_t ncolslb = mxGetN(prhs[3]);

  // figure out dimensions of ub
  const size_t mrowsub = mxGetM(prhs[4]);
  const size_t ncolsub = mxGetN(prhs[4]);

  // ---------------------------------------------------------------------------
  // verify if dimesions matches
  if (ncolsx != 1) {
    mexPrintf("x should be a column vector\n");
  }
  if (ncolslb != 1) {
    mexPrintf("lb should be a column vector\n");
  }
  if (ncolsub != 1) {
    mexPrintf("ub should be a column vector\n");
  }

  if (mrowsx != mrowslb) {
    mexPrintf("size of x and lb should be same\n");
  }
  if (mrowsx != mrowsub) {
    mexPrintf("size of x and ub should be same\n");
  }

  if (mrowsx != mrowsA) {
    mexPrintf("size of x and A should match\n");
  }
  if (mrowsA != ncolsA) {
    mexPrintf("A should be a square matrix\n");
  }

  // ---------------------------------------------------------------------------
  // declare variables and map it to Eigen
  const Eigen::Map<MatlabSparse> A = matlab_to_eigen_sparse(prhs[0]);

  const Eigen::Map<Eigen::VectorXd> b(mxGetPr(prhs[1]), mrowsb, ncolsb);
  const Eigen::Map<Eigen::VectorXd> lb(mxGetPr(prhs[3]), mrowslb, ncolslb);
  const Eigen::Map<Eigen::VectorXd> ub(mxGetPr(prhs[4]), mrowsub, ncolsub);

  Eigen::Map<Eigen::VectorXd> x(mxGetPr(prhs[2]), mrowsx, ncolsx);

  const double omega = mxGetScalar(prhs[5]);
  const uint max_iter = mxGetScalar(prhs[6]);
  const double atol = mxGetScalar(prhs[7]);
  const bool show = mxGetScalar(prhs[8]);

  plhs[0] = mxCreateDoubleMatrix(mrowsx, ncolsx, mxREAL);
  Eigen::Map<Eigen::VectorXd> x_out(mxGetPr(plhs[0]), mrowsx, ncolsx);

  plhs[1] = mxCreateLogicalMatrix(mrowsx, ncolsx);
  mxLogical *marker = mxGetLogicals(plhs[1]);

  // ----------------------------------------------------------------------
  // declare variables and for GS
  double sum;
  uint curr_row;
  double A_ij;
  double x_j;
  double A_ii;

  Eigen::VectorXd res;
  Eigen::VectorXd ProjConst;
  Eigen::VectorXd ProjRes;

  int iter = 0;

  // compute projected gradient -------------------------------------------
  res = (A * x - b);

  ProjConst = x - res;

  for (int curr_col = 0; curr_col < A.outerSize(); ++curr_col) {
    if (ProjConst(curr_col) <= lb(curr_col))
      ProjConst(curr_col) = lb(curr_col);
    if (ProjConst(curr_col) >= ub(curr_col))
      ProjConst(curr_col) = ub(curr_col);
  }
  ProjRes = ProjConst - x;

  while (iter < max_iter && ProjRes.norm() > atol) {
    // std::cout << "iter " << iter << std::endl;

    // this iterates over columns
    for (int curr_col = 0; curr_col < A.outerSize(); ++curr_col) {
      // std::cout << "Here: :-> col " << curr_col << std::endl;
      sum = 0.0;

      // this loop iterates over non-zero entries in given columns
      // hence, the values of the rows

      for (Eigen::Map<MatlabSparse>::InnerIterator it(A, curr_col); it; ++it) {
        curr_row = it.row();
        A_ij = it.value();
        x_j = x(curr_row);

        if (curr_col != curr_row) {
          sum += A_ij * x_j;
        } else {
          A_ii = A_ij;
        }
      }
      // std::cout << "sum:" << sum << std::endl;
      // std::cout << "b: " << b(curr_col) << " ";
      x(curr_col) =
          (1 - omega) * x(curr_col) + (b(curr_col) - sum) * (omega / A_ii);

      // check for contraints
      if ((x(curr_col) >= ub(curr_col)) || (x(curr_col) <= lb(curr_col))) {
        x(curr_col) =
            std::max(lb(curr_col), std::min(x(curr_col), ub(curr_col)));
        marker[curr_col] = true;
      }
    }

    // this iterates over columns
    for (int curr_col = A.outerSize() - 1; curr_col >= 0; --curr_col) {
      // std::cout << "Here: :-> col " << curr_col << std::endl;
      sum = 0.0;

      // this loop iterates over non-zero entries in given columns
      // hence, the values of the rows

      for (Eigen::Map<MatlabSparse>::InnerIterator it(A, curr_col); it; ++it) {
        curr_row = it.row();

        A_ij = it.value();
        x_j = x(curr_row);

        if (curr_col != curr_row) {
          sum += A_ij * x_j;
        } else {
          A_ii = A_ij;
        }
      }
      // std::cout << "sum:" << sum << std::endl;
      // std::cout << "b: " << b(curr_col) << " ";
      x(curr_col) =
          (1 - omega) * x(curr_col) + (b(curr_col) - sum) * (omega / A_ii);
      // check for contraints
      if ((x(curr_col) >= ub(curr_col)) || (x(curr_col) <= lb(curr_col))) {
        x(curr_col) =
            std::max(lb(curr_col), std::min(x(curr_col), ub(curr_col)));
        marker[curr_col] = true;
      }
    }

    // std::cout<< x.transpose() << std::endl;
    // std::cout<< marker.transpose() << std::endl;

    // compute projected gradient -----------------------------------------
    res = (A * x - b);

    ProjConst = x - res;

    for (int curr_col = 0; curr_col < A.outerSize(); ++curr_col) {
      if (ProjConst(curr_col) <= lb(curr_col))
        ProjConst(curr_col) = lb(curr_col);
      if (ProjConst(curr_col) >= ub(curr_col))
        ProjConst(curr_col) = ub(curr_col);
    }
    ProjRes = ProjConst - x;
    iter += 1;
    if (show == true)
      std::cout << "ProjSSOR iter = " << iter << "  |r| = " << ProjRes.norm()
                << std::endl;
  }
  plhs[2] = mxCreateDoubleScalar(ProjRes.norm());
  x_out = x;

  return;
}
