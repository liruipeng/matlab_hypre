#include "mex.h"
#include "matrix.h"
#include <math.h>
#include "_hypre_utilities.h"
#include "_hypre_parcsr_ls.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include <vector>

using namespace std;

mxArray* convert_hypreCSR_to_mxSparse(int ii, hypre_CSRMatrix *A);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   /* input:
    *   A     : a square sparse matrix
    *   setup :
    * Output:
    *   amg   : array of structures of {cf, A, P}
    */
   // default parameters
   int print_level  = 3,
       agg_levels   = 0,
       coarsen_type = 10,
       relax_type   = 3,
       relax_sweeps = 1,
       max_level    = 20,
       interp_type  = 6,
       Pmax         = 4,
       num_func     = 1,
       max_iter     = 50,
       solver_id    = 1,
       elast        = 0;

   mxArray *mx_dof_func = NULL;

   double sol_tol   = 1e-8,
          theta     = 0.25,
          maxrowsum = 1.0;

   const mxArray *setup = prhs[2];

   if (nrhs >= 3)
   {
      if (mxIsStruct(setup))
      {
         mxArray *mxTmp;
         mxTmp = mxGetField(setup, 0, "print_level");
         if (mxTmp)
         {
            print_level = (int) (*mxGetPr(mxTmp));
            if (print_level >= 4)
            {
               mexPrintf("print_level %d\n", print_level);
            }
         }
         mxTmp = mxGetField(setup, 0, "coarsen_type");
         if (mxTmp)
         {
            coarsen_type = (int) (*mxGetPr(mxTmp));
            if (print_level >= 4)
            {
               mexPrintf("coarsen_type %d\n", coarsen_type);
            }
         }
         mxTmp = mxGetField(setup, 0, "interp_type");
         if (mxTmp)
         {
            interp_type = (int) (*mxGetPr(mxTmp));
            if (print_level >= 4)
            {
               mexPrintf("interp_type %d\n", interp_type);
            }
         }
         mxTmp = mxGetField(setup, 0, "relax_type");
         if (mxTmp)
         {
            relax_type = (int) (*mxGetPr(mxTmp));
            if (print_level >= 4)
            {
               mexPrintf("relax_type %d\n", relax_type);
            }
         }
         mxTmp = mxGetField(setup, 0, "max_level");
         if (mxTmp)
         {
            max_level = (int) (*mxGetPr(mxTmp));
            if (print_level >= 4)
            {
               mexPrintf("max_level %d\n", max_level);
            }
         }
         mxTmp = mxGetField(setup, 0, "num_func");
         if (mxTmp)
         {
            num_func = (int) (*mxGetPr(mxTmp));
            if (print_level >= 4)
            {
               mexPrintf("num_func %d\n", num_func);
            }
         }

         mx_dof_func = mxGetField(setup, 0, "dof_func");

         mxTmp = mxGetField(setup, 0, "sol_tol");
         if (mxTmp)
         {
            sol_tol = (double) (*mxGetPr(mxTmp));
            if (print_level >= 4)
            {
               mexPrintf("sol_tol %e\n", sol_tol);
            }
         }
         mxTmp = mxGetField(setup, 0, "max_iter");
         if (mxTmp)
         {
            max_iter = (int) (*mxGetPr(mxTmp));
            if (print_level >= 4)
            {
               mexPrintf("max_iter %d\n", max_iter);
            }
         }
         mxTmp = mxGetField(setup, 0, "solver_id");
         if (mxTmp)
         {
            solver_id = (int) (*mxGetPr(mxTmp));
            if (print_level >= 4)
            {
               mexPrintf("solver_id %d\n", solver_id);
            }
         }
         mxTmp = mxGetField(setup, 0, "elast");
         if (mxTmp)
         {
            elast = (int) (*mxGetPr(mxTmp));
            if (print_level >= 4)
            {
               mexPrintf("elast %d\n", elast);
            }
         }
         mxTmp = mxGetField(setup, 0, "theta");
         if (mxTmp)
         {
            theta = (double) (*mxGetPr(mxTmp));
            if (print_level >= 4)
            {
               mexPrintf("theta %f\n", theta);
            }
         }
         mxTmp = mxGetField(setup, 0, "maxrowsum");
         if (mxTmp)
         {
            maxrowsum = (double) (*mxGetPr(mxTmp));
            if (print_level >= 4)
            {
               mexPrintf("maxrowsum %f\n", maxrowsum);
            }
         }
      }
   }

   const mxArray *mxA = prhs[0];
   /* A matrix: must be row-wise, so it should be A^T on input from matlab */
   if (!mxIsSparse(mxA))
   {
      mexErrMsgTxt("A must be a sparse matrix");
   }
   size_t m = mxGetM(mxA);
   size_t n = mxGetN(mxA);
   if (m != n)
   {
      mexErrMsgTxt("A must be a square matrix");
   }
   /* 3 CSR arrays */
   mwIndex *ia = mxGetJc(mxA);
   mwIndex *ja = mxGetIr(mxA);
   double *a = mxGetPr(mxA);
   mwIndex nnz = ia[m];

   /* create matrix */
   HYPRE_IJMatrix A;
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, m - 1, 0, m - 1, &A);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(A);
   vector<int> ncols(m), rows(m), cols(nnz);
   //mexPrintf("sizeof(mwIndex) %ld, sizeof(int) %ld\n", sizeof(mwIndex), sizeof(int));
   for (int i = 0; i < m; i++)
   {
      ncols[i] = ia[i + 1] - ia[i];
      rows[i] = i;
   }
   for (int i = 0; i < nnz; i++)
   {
      cols[i] = ja[i];
   }
   HYPRE_IJMatrixSetValues(A, m, ncols.data(), rows.data(), cols.data(), a);
   HYPRE_IJMatrixAssemble(A);
   //mexPrintf("A: %d x %d, nnz %d\n", m, n, nnz);

   /* create rhs */
   const mxArray *mxb = prhs[1];
   if (mxIsSparse(mxb) || mxGetM(mxb) != m || mxGetN(mxb) != 1)
   {
      mexPrintf("b must be a dense vector of size %d\n", m);
      mexErrMsgTxt("b error");
   }
   double *b = mxGetPr(mxb);

   /* Create the rhs and solution */
   HYPRE_IJVector B, X;
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, m - 1, &B);
   HYPRE_IJVectorSetObjectType(B, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(B);
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, m - 1, &X);
   HYPRE_IJVectorSetObjectType(X, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(X);

   double dzero = 0.0;
   for (int i = 0; i < m; i++)
   {
      HYPRE_IJVectorSetValues(B, 1, &i, b + i);
      HYPRE_IJVectorSetValues(X, 1, &i, &dzero);
   }
   HYPRE_IJVectorAssemble(B);
   HYPRE_IJVectorAssemble(X);

   /* AMG precond */
   HYPRE_Solver precond, solver;
   HYPRE_BoomerAMGCreate(&precond);
   /* Set some parameters (See Reference Manual for more parameters) */
   HYPRE_BoomerAMGSetCoarsenType(precond, coarsen_type);
   HYPRE_BoomerAMGSetAggNumLevels(precond, agg_levels);

   // coarsest grid solver
   // par_amg_setup.c: hypre_GaussElimSetup
   //HYPRE_BoomerAMGSetMaxCoarseSize(precond, m); // in order to force coarset solver to be GE
   HYPRE_BoomerAMGSetCycleRelaxType(precond, 9, 3);  // GE

   // Set relax
   HYPRE_BoomerAMGSetRelaxType(precond, relax_type);
   if (relax_type == 3)
   {
      HYPRE_BoomerAMGSetCycleRelaxType(precond, 13, 1);  // hybrid GS forward
      HYPRE_BoomerAMGSetCycleRelaxType(precond, 14, 2);  // hybrid GS backward
      HYPRE_BoomerAMGSetRelaxOrder(precond, 0); // 0: lexicographic order, 1: C-F relax
   }
   HYPRE_BoomerAMGSetNumSweeps(precond, relax_sweeps);
   HYPRE_BoomerAMGSetStrongThreshold(precond, theta);
   HYPRE_BoomerAMGSetInterpType(precond, interp_type);
   HYPRE_BoomerAMGSetPMaxElmts(precond, Pmax);
   HYPRE_BoomerAMGSetPrintLevel(precond, print_level);
   HYPRE_BoomerAMGSetMaxLevels(precond, max_level);  /* maximum number of levels */

   /* settings used in mfem for elasity problem */
   if (elast)
   {
      HYPRE_BoomerAMGSetNodal(precond, 4);  // 4: row-sum norm
      HYPRE_BoomerAMGSetNodalDiag(precond,
                                  1); // 1: the diagonal entry is set to the negative sum of all off diagonal entries
      //HYPRE_BoomerAMGSetCycleRelaxType(precond, 8, 3); // set the relax type for the coarsest level
      //HYPRE_BoomerAMGSetInterpRefine(precond, interp_refine);
   }

   if (num_func > 1)
   {
      // mexPrintf("num_func %d\n", num_func);
      HYPRE_BoomerAMGSetNumFunctions(precond, num_func);
      HYPRE_BoomerAMGSetAggNumLevels(precond, 0);
      //HYPRE_BoomerAMGSetStrongThreshold(precond, 0.5);

      if (mx_dof_func)
      {
         const mwSize *doflen = mxGetDimensions(mx_dof_func);
         if (print_level >= 4)
         {
            mexPrintf("dof_func len %d\n", doflen[0]);
         }
         if (doflen[0] == m)
         {
            if (print_level >= 4)
            {
               mexPrintf("setting the given dof func\n");
            }
            int *dof_func = (int*) malloc(m * sizeof(int));
            double *tmp = mxGetPr(mx_dof_func);
            for (size_t i = 0; i < m; i++)
            {
               dof_func[i] = (int) tmp[i];
            }
            HYPRE_BoomerAMGSetDofFunc(precond, dof_func);
         }
      }
   }

   if (maxrowsum > 0.0)
   {
      HYPRE_BoomerAMGSetMaxRowSum(precond, maxrowsum);
   }

   HYPRE_ParCSRMatrix parcsr_A;
   HYPRE_ParVector par_b, par_x;
   HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
   HYPRE_IJVectorGetObject(B, (void **) &par_b);
   HYPRE_IJVectorGetObject(X, (void **) &par_x);

   if (solver_id < 0)
   {
      // setup AMG
      HYPRE_BoomerAMGSetup(precond, parcsr_A, par_b, par_x);
   }
   else
   {
      int num_iterations;
      double final_res_norm;
      if (solver_id == 0)
      {
         int rtype;
         // setup AMG
         HYPRE_BoomerAMGSetup(precond, parcsr_A, par_b, par_x);
         // solve
         HYPRE_BoomerAMGSetTol(precond, sol_tol);      /* conv. tolerance */
         HYPRE_BoomerAMGSetMaxIter(precond, max_iter);
         //HYPRE_BoomerAMGGetCycleRelaxType(precond, &rtype, 3);
         //mexPrintf("rtype = %d\n", rtype);
         HYPRE_BoomerAMGSolve(precond, parcsr_A, par_b, par_x);
         /* Run info - needed logging turned on */
         HYPRE_BoomerAMGGetNumIterations(precond, &num_iterations);
         HYPRE_BoomerAMGGetFinalRelativeResidualNorm(precond, &final_res_norm);
      }
      else
      {
         // Use as a preconditioner (one V-cycle, zero tolerance)
         HYPRE_BoomerAMGSetTol(precond, 0.0);
         HYPRE_BoomerAMGSetMaxIter(precond, 1);
         /* Create PCG solver */
         HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);
         /* Set some parameters (See Reference Manual for more parameters) */
         HYPRE_PCGSetMaxIter(solver, max_iter); /* max iterations */
         HYPRE_PCGSetTol(solver, sol_tol); /* conv. tolerance */
         //HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
         HYPRE_PCGSetPrintLevel(solver, 2); /* prints out the iteration info */
         HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

         /* Set the PCG preconditioner */
         HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);
         /* Now setup and solve! */
         HYPRE_ParCSRPCGSetPrintLevel(solver, 2);
         HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
         HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);
         HYPRE_PCGGetNumIterations(solver, &num_iterations);
         HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         HYPRE_ParCSRPCGDestroy(solver);
      }

      // print solve info
      HYPRE_ParCSRMatrixMatvec (1.0, parcsr_A, par_x, -1.0, par_b);
      double rnrm = sqrt(hypre_ParVectorInnerProd(par_b, par_b));
      mexPrintf("\n");
      mexPrintf("Iter %d, ", num_iterations);
      mexPrintf("Relres %e [res %e]\n", final_res_norm, rnrm);
      mexPrintf("\n");
   }

   // extract AMG data
   hypre_ParAMGData *amg_data = (hypre_ParAMGData *) precond;
   int nlev = hypre_ParAMGDataNumLevels(amg_data);
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParCSRMatrix **P_array = hypre_ParAMGDataPArray(amg_data);
   //hypre_ParCSRMatrix **R_array = hypre_ParAMGDataRArray(amg_data);
   hypre_IntArray **cf = hypre_ParAMGDataCFMarkerArray(amg_data);
   // mexPrintf("amg num of levels %d\n", nlev);

   // create MatLab datatypes and copy
   const char *fieldnames[] = {"cf", "A", "P"};
   plhs[0] = mxCreateStructMatrix(nlev, 1, 3, fieldnames);
   for (int i = 0; i < nlev; i++)
   {
      // dof of this level
      int nAi = hypre_ParCSRMatrixGlobalNumRows(A_array[i]);
      // cf marker of this level
      int ncf = i < nlev - 1 ?  nAi : 1;
      mxArray *mxcf = mxCreateDoubleMatrix(ncf, 1, mxREAL);
      double *mxcf_pr = mxGetPr(mxcf);
      for (int j = 0; i < nlev - 1 && j < nAi; j++)
      {
         mxcf_pr[j] = hypre_IntArrayData(cf[i])[j];
      }
      mxSetFieldByNumber(plhs[0], i, 0, mxcf);
      // Ai of this level
      int nnzAi = hypre_ParCSRMatrixNumNonzeros(A_array[i]);
      hypre_CSRMatrix *Ai = hypre_ParCSRMatrixDiag(A_array[i]);
      if (hypre_CSRMatrixNumCols(Ai) != nAi || hypre_CSRMatrixNumRows(Ai) != nAi ||
          hypre_CSRMatrixNumNonzeros(Ai) != nnzAi)
      {
         mexErrMsgTxt("Ai size/nnz wrong ");
      }
      mxArray *mxAi = convert_hypreCSR_to_mxSparse(i, Ai);
      mxSetFieldByNumber(plhs[0], i, 1, mxAi);

      // except for the last level
      if (i < nlev - 1)
      {
         // interpolation matrix of this level
         hypre_CSRMatrix *Pi = hypre_ParCSRMatrixDiag(P_array[i]);
         // dof of the next level
         int nAi1 = hypre_ParCSRMatrixGlobalNumRows(A_array[i + 1]);
         if (hypre_CSRMatrixNumCols(Pi) != nAi1 || hypre_CSRMatrixNumRows(Ai) != nAi)
         {
            mexErrMsgTxt("Interpolation matrix P size/nnz wrong ");
         }
         mxArray *mxPi = convert_hypreCSR_to_mxSparse(i, Pi);
         mxSetFieldByNumber(plhs[0], i, 2, mxPi);
         // R = P
         //hypre_CSRMatrix *Ri = hypre_ParCSRMatrixDiag(R_array[i]);
         //mxArray *mxRi = convert_hypreCSR_to_mxSparse(i, Ri);
         //mxSetFieldByNumber(plhs[0], i, 3, mxRi);
      }
   }

   HYPRE_BoomerAMGDestroy(precond);
   HYPRE_IJMatrixDestroy(A);

   mxArray *mX = mxCreateDoubleMatrix(m, 1, mxREAL);
   double *mX_pr = mxGetPr(mX);
   for (int j = 0; j < m; j++)
   {
      mX_pr[j] = par_x->local_vector->data[j];
   }

   plhs[1] = mX;
}


template <int OUTINDEX>
void csrcsc(int nrow, int ncol, int job,
            double *a, int *ja, int *ia,
            double *ao, int *jao, int *iao)
{

   for (int i = 0; i < ncol + 1; i++)
   {
      iao[i] = 0;
   }
   // compute nnz of columns of A
   for (int i = 0; i < nrow; i++)
   {
      for (int k = ia[i]; k < ia[i + 1]; k++)
      {
         iao[ja[k] + 1] ++;
      }
   }
   // compute pointers from lengths
   for (int i = 0; i < ncol; i++)
   {
      iao[i + 1] += iao[i];
   }
   // now do the actual copying
   for (int i = 0; i < nrow; i++)
   {
      for (int k = ia[i]; k < ia[i + 1]; k++)
      {
         int j = ja[k];
         if (job)
         {
            ao[iao[j]] = a[k];
         }
         jao[iao[j]++] = i + OUTINDEX;
      }
   }
   /*---- reshift iao and leave */
   for (int i = ncol; i > 0; i--)
   {
      iao[i] = iao[i - 1] + OUTINDEX;
   }
   iao[0] = OUTINDEX;
}

mxArray* convert_hypreCSR_to_mxSparse(int ii, hypre_CSRMatrix *A)
{
   // A is m x n CSR
   mwSize m = hypre_CSRMatrixNumRows(A);
   mwSize n = hypre_CSRMatrixNumCols(A);
   mwSize nnz = hypre_CSRMatrixNumNonzeros(A);
   //mexPrintf("%d %d %d(%d)\n", m,n,nnz,A->i[m]);
   // note mxA is CSC
   mxArray *mxA = mxCreateSparse(m, n, nnz, mxREAL);
   // convert A to m x n CSC
   vector<int> ib(n + 1), jb(nnz);
   vector<double> b(nnz);

   csrcsc<0>(m, n, 1, A->data, A->j, A->i, b.data(), jb.data(), ib.data());
   double *a = mxGetPr(mxA);
   mwIndex *ja = mxGetIr(mxA);
   mwIndex *ia = mxGetJc(mxA);
   for (int i = 0; i < nnz; i++)
   {
      a[i] = b[i];
      ja[i] = jb[i];
   }
   for (int i = 0; i <= n; i++)
   {
      ia[i] = ib[i];
   }
   return mxA;
}

