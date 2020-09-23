#ifndef KERNEL_H
#define KERNEL_H
/*
 * Header file for API 
 */

#include <dgl/array.h>
#include <dgl/bcast.h>
#include <math.h>
#include <iostream>
#include "../selector.h"
#define INDEXTYPE int

#ifdef __cplusplus 
   extern "C"
   {
#endif
/* 
 * INT type can be changed here, implemenetation not dependent on int  
 */

//#ifndef DREAL 
   #define BCL_TYPE double 
// double function prototypes  
void dgsddmm_csr (const char tkern, const INDEXTYPE m, const INDEXTYPE n, 
      const INDEXTYPE k,const double alpha, const INDEXTYPE nnz, 
      const INDEXTYPE rows, const INDEXTYPE cols, const INDEXTYPE *indx, 
      const INDEXTYPE *pntrb, const INDEXTYPE *pntre, const double *A, 
      const INDEXTYPE lda, const double *B, const INDEXTYPE ldb, 
      const double beta, double *C, const INDEXTYPE ldc);

void trusted_dgsddmm_csr (const char tkern, const INDEXTYPE m, const INDEXTYPE n, 
      const INDEXTYPE k,const double alpha, const INDEXTYPE nnz, 
      const INDEXTYPE rows, const INDEXTYPE cols, const INDEXTYPE *indx, 
      const INDEXTYPE *pntrb, const INDEXTYPE *pntre, const double *A, 
      const INDEXTYPE lda, const double *B, const INDEXTYPE ldb, 
      const double beta, double *C, const INDEXTYPE ldc);
//#else
/*   #define BCL_TYPE float 
void sgsddmm_csr (const char tkern, const INDEXTYPE m, const INDEXTYPE n, 
      const INDEXTYPE k,const float alpha, const INDEXTYPE nnz, 
      const INDEXTYPE rows, const INDEXTYPE cols, const INDEXTYPE *indx, 
      const INDEXTYPE *pntrb, const INDEXTYPE *pntre, const float *A, 
      const INDEXTYPE lda, const float *B, const INDEXTYPE ldb, 
      const float beta, float *C, const INDEXTYPE ldc);

void trusted_sgsddmm_csr (const char tkern, const INDEXTYPE m, const INDEXTYPE n, 
      const INDEXTYPE k,const float alpha, const INDEXTYPE nnz, 
      const INDEXTYPE rows, const INDEXTYPE cols, const INDEXTYPE *indx, 
      const INDEXTYPE *pntrb, const INDEXTYPE *pntre, const float *A, 
      const INDEXTYPE lda, const float *B, const INDEXTYPE ldb, 
      const float beta, float *C, const INDEXTYPE ldc);
*/
//#endif

#ifdef __cplusplus 
   }  // extern "C"
#endif

#endif
