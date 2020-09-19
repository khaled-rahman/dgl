#ifndef DGL_ARRAY_CPU_SDDMMSPMM_H_
#define DGL_ARRAY_CPU_SDDMMSPMM_H_
#include <dgl/array.h>
#include <dgl/bcast.h>
#include <math.h>
#include <iostream>
#include "../selector.h"
#include "sddmm.h"

#define SM_TABLE_SIZE 2048
#define SM_BOUND 5.0
#define SM_RESOLUTION SM_TABLE_SIZE/(2.0 * SM_BOUND)

using namespace std;

namespace dgl {
namespace aten {
namespace cpu {

template <typename DType>
DType scale(DType v){
	if(v > SM_BOUND) return SM_BOUND;
        else if(v < -SM_BOUND) return -SM_BOUND;
        return v;
}

template <typename DType>
DType fast_SM(DType v, DType *sm_table){
        if(v > SM_BOUND) return 1.0;
        else if(v < -SM_BOUND) return 0.0;
        return sm_table[(int)((v + SM_BOUND) * SM_RESOLUTION)];
}

template <typename IdType, typename DType>
void init_SM_TABLE(DType *sm_table){
        DType x;
        for(IdType i = 0; i < SM_TABLE_SIZE; i++){
                x = 2.0 * SM_BOUND * i / SM_TABLE_SIZE - SM_BOUND;
                sm_table[i] = 1.0 / (1 + exp(-x));
        }
}

template <typename IdType, typename DType>
void SDDMMSPMMCsrTdist(const IdType *indptr, const IdType *indices, const IdType *edges,
                const DType *X, const DType *Y, DType *O, const IdType N, const int64_t dim) {

#pragma omp parallel for
for (IdType rid = 0; rid < N; ++rid) {
        const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
        const IdType iindex = rid * dim;
	DType T[dim];
        for (IdType j = row_start; j < row_end; ++j){
                const IdType cid = indices[j];
                const IdType jindex = cid * dim;
                DType attrc = 0;
                for (int64_t k = 0; k < dim; ++k) {
			T[k] = X[iindex + k] - Y[jindex + k];
                        attrc += T[k] * T[k];
                }
                DType d1 = -2.0 / (1.0 + attrc);
                for (int64_t k = 0; k < dim; ++k) {
			T[k] = scale<DType>(T[k] * d1);
                        O[iindex+k] = O[iindex+k]  + T[k];
                }
        }
}

}


template <typename IdType, typename DType>
void SDDMMSPMMCsrSigmoid(const IdType *indptr, const IdType *indices, const IdType *edges, 
		const DType *X, const DType *Y, DType *O, const IdType N, const int64_t dim) {

DType *sm_table;
sm_table = static_cast<DType *> (::operator new (sizeof(DType[SM_TABLE_SIZE])));
init_SM_TABLE<IdType, DType>(sm_table);

//for(IdType i = 0; i < SM_TABLE_SIZE; i++) cout << sm_table[i] << " "; cout << endl;

#pragma omp parallel for
for (IdType rid = 0; rid < N; ++rid){
        const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
        const IdType iindex = rid * dim;
	for (IdType j = row_start; j < row_end; ++j){
		const IdType cid = indices[j];
                const IdType jindex = cid * dim;
                DType attrc = 0;
                for (int64_t k = 0; k < dim; ++k) {
                        attrc += X[iindex + k] * Y[jindex + k];
                }
                //DType d1 = 1.0 / (1.0 + exp(-attrc));
                DType d1 = fast_SM<DType>(attrc, sm_table);
		//printf("");
		for (int64_t k = 0; k < dim; ++k) {
                        O[iindex+k] = O[iindex+k]  + (1.0 - d1) * Y[jindex + k];
                }
        }
}		

}

template <typename IdType, typename DType, typename Op,
int LhsTarget = 0, int RhsTarget = 2>
void SDDMMSPMMCsr(const BcastOff& bcast,
const CSRMatrix& csr,
              NDArray lhs, NDArray rhs, NDArray out) {

const IdType* indptr = csr.indptr.Ptr<IdType>();
const IdType* indices = csr.indices.Ptr<IdType>();
const IdType* edges = csr.data.Ptr<IdType>();
const DType* X = lhs.Ptr<DType>();
const DType* Y = rhs.Ptr<DType>();
const int64_t dim = bcast.out_len;
// lhs_dim = bcast.lhs_len, rhs_dim = bcast.rhs_len, reduce_size = bcast.reduce_size;
DType* O = out.Ptr<DType>();
//SDDMMSPMMCsrSigmoid<IdType, DType>(indptr, indices, edges, X, Y, O, csr.num_rows, dim);
SDDMMSPMMCsrTdist<IdType, DType>(indptr, indices, edges, X, Y, O, csr.num_rows, dim);
/*
#pragma omp parallel for
for (IdType rid = 0; rid < csr.num_rows; ++rid) {
	const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
	const IdType iindex = rid * dim;
	//DType* out_off = O + cid * dim;
	for (IdType j = row_start; j < row_end; ++j){
		//cout << rid << ", " << indices[j] << endl;
		const IdType cid = indices[j];
		const IdType jindex = cid * dim;
		DType attrc = 0;
		for (int64_t k = 0; k < dim; ++k) {
        		attrc += X[iindex + k] * Y[jindex + k];
      		}
		DType d1 = 1.0 / (1.0 + exp(-attrc));
		for (int64_t k = 0; k < dim; ++k) {
			O[iindex+k] = O[iindex+k]  + (1.0 - d1) * Y[jindex + k];
		}
    	}
  }
*/
}	
}
}
}
#endif 
