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

auto lsize = lhs.GetSize();
cout << "LHS Size:" << lsize << endl;
auto rsize = rhs.GetSize();
cout << "RHS Size:" << rsize << endl;
auto osize = out.GetSize();
cout << "Out Size:" << osize << endl;

const bool has_idx = !IsNullArray(csr.data);
const IdType* indptr = csr.indptr.Ptr<IdType>();
const IdType* indices = csr.indices.Ptr<IdType>();
const IdType* edges = csr.data.Ptr<IdType>();
const DType* X = lhs.Ptr<DType>();
const DType* Y = rhs.Ptr<DType>();
const int64_t dim = bcast.out_len;
// lhs_dim = bcast.lhs_len, rhs_dim = bcast.rhs_len, reduce_size = bcast.reduce_size;
DType* O = out.Ptr<DType>();
//#pragma omp parallel for
for (IdType rid = 0; rid < csr.num_rows; ++rid) {
	const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
	const IdType iindex = rid * dim;
	for (IdType j = row_start; j < row_end; ++j){
		const IdType cid = indices[j];
		const IdType jindex = cid * dim;
		//const IdType eid = has_idx? edges[j] : j;
		DType* out_off = O + cid * dim;
		DType attrc = 0;
		for (int64_t k = 0; k < dim; ++k) {
			//const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
			//const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
			//const DType* lhs_off = Op::use_lhs?
			//X + Selector<LhsTarget>::Call(rid, eid, cid) * lhs_dim + lhs_add * reduce_size : nullptr;
			//const DType* rhs_off = Op::use_rhs?
          		//Y + Selector<RhsTarget>::Call(rid, eid, cid) * rhs_dim + rhs_add * reduce_size : nullptr;
        		//out_off[k] = Op::Call(lhs_off, rhs_off, reduce_size);
        		attrc += X[iindex + k] * Y[jindex + k];
      		}
		DType d1 = 1.0 / (1.0 + exp(-attrc));
		for (int64_t k = 0; k < dim; ++k) {
			out_off[k] = out_off[k]  + (1.0 - d1) * Y[jindex + k];
		}
    	}
  }
}	
}
}
}
#endif 
