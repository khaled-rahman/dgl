#ifndef DGL_ARRAY_CPU_SDDMMSPMM_H_
#define DGL_ARRAY_CPU_SDDMMSPMM_H_
#include <dgl/array.h>
#include <dgl/bcast.h>
#include <math.h>
#include <iostream>
#include "../selector.h"
#include "sddmm.h"

using namespace std;

namespace dgl {
namespace aten {
namespace cpu {
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
}	
}
}
}
#endif 
