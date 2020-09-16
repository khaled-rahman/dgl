import dgl,torch
from dgl.sparse import _gsddmmspmm, _gsddmm, _gspmm
from dgl.data.citation_graph import load_cora
import sys, time
import numpy as np

def cacheflush():
	print("Cache Refreshing...")
	BIGNUM = 100000
	L1 = []
	L2 = []
	for i in range(BIGNUM):
		L1.insert(0, i)
		L2.insert(0, i+1)
	print("Done!")
	
def batch_process(graph, batchsize=64):
	batches = []
	b = 0
	N = len(graph.nodes())
	edges = [np.array(graph.edges()[0]),np.array(graph.edges()[1])]
	while b < N:
		bedges = []
		for i in range(b, min(b+batchsize, N)):
			bedges += [[edges[0][row], edges[1][row]] for row in range(len(edges[0])) if edges[0][row]==i]
		bedge_left = [i for [i,j] in bedges]
		bedge_right = [j for [i,j] in bedges]
		bgraph = dgl.heterograph(torch.tensor(bedge_left), torch.tensor(bedge_right))
		batches.append([bgraph, b , min(b+batchsize, N)])
		b += batchsize
	return batches

#combined sddmmspmm kernel
def f2valgo(batchgraphs, embed, iterations = 5):
	it = 0
	#remember to add learning rate ...
	totalktime = 0
	while it < iterations:
		start = time.time()
		output = _gsddmmspmm(batchgraphs._graph, "mul", embed, embed, "u", "v")
		end = time.time()
		totalktime += end - start
		it += 1
	print("Total F2V Kernel Time:", totalktime, "seconds")
	return embed

#gdl:sddmm+transformation+spmm kernel
def f2vusingdefault(batchgraphs, batchgraphsT, embed, iterations=5):
	it = 0
	totalktime = 0
	while it < iterations:
		start = time.time()
		#SDDMM operation
		X = _gsddmm(batchgraphs._graph, 'dot', embed, embed, lhs_target='u', rhs_target='v')
		#non-linear transformation
		Y = 1. / (1 + np.exp(-X))
		#SPMM operation
		output = _gspmm(batchgraphsT._graph, "mul", "sum", embed, Y)[0]
		end = time.time()
		totalktime += end - start
		it += 1
	print("Total GDL Kernel Time:", totalktime, "seconds")
	return embed

#manual result verification
def gsddmmspmmkerneltest(graph, A):
	coo = graph.adj()._indices().numpy()
	N = graph.adj().size()[0]
	S = [[0 for i in range(N)] for j in range(N)]
	nnz = graph.adj()._nnz()
	for e in range(nnz):
		u = coo[0][e]
		v = coo[1][e]
		S[u][v] = 1
	B = np.transpose(A)
	X = np.array([[0.0 for i in range(N)] for j in range(N)])
	for e in range(nnz):
		u = coo[0][e]
		v = coo[1][e]
		dp = np.dot(A[u,].numpy(), B[:, v].numpy())
		X[u][v] = dp
	#X = np.matmul(np.matmul(A, B), S)
	#print(X)
	Y = 1.0 / (1.0 + np.exp(-X))
	Z = np.array([[0 if S[i][j] == 0 else 1-Y[i][j] for j in range(N)] for i in range(N)])
	#print(Z)
	O = np.matmul(Z, A)
	return O

def transposeGraph(graph):
	edges = graph.edges()
	source = edges[0]
	dest = edges[1]
	tGraph = dgl.graph((dest, source))
	return tGraph

if __name__ == "__main__":
	if len(sys.argv) > 1:
		data = load_cora(".")
		graph = data[0]
		tgraph = transposeGraph(graph)
		N = len(graph.nodes())
		embed = torch.rand(N, 128)
		#need to check batch processing ...
		#bgraphs = batch_process(graph)
		#print(graph)
		cacheflush()
		#output = f2valgo(graph, embed)
		#print("SDDMMSPMM Kernel:")
		#print(output)
		#cacheflush()
		dgloutput = f2vusingdefault(graph, tgraph, embed)
		print("DGL: SDDMMSPMM+Transformation+SPMM")
		print(dgloutput)
		#cacheflush()
		#f2valgo(graph, embed)
	else:
		print("Simple SDDMMSPMM Test:")
		g = dgl.graph(([0, 0, 1, 1, 2, 3], [1, 2, 2, 4, 3, 4]))
		gt = dgl.graph(([1, 2, 2, 4, 3, 4], [0, 0, 1, 1, 2, 3]))
		#g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
		embed = torch.rand(5, 5)
		#print(embed)
		#output = _gsddmmspmm(g._graph, "mul", embed, embed, "u", "v")
		cacheflush()
		output = f2valgo(g, embed)
		print("SDDMMSPMM Kernel:")
		print(output)
		cacheflush()
		dgloutput = f2vusingdefault(g, gt, embed) 
		print("DGL: SDDMMSPMM+Transformation+SPMM")
		print(dgloutput)
		#output = f2valgo(g, embed)
		#print("Manual SPDDMM+SPMM:")
		#print(embed)
		#mout = gsddmmspmmkerneltest(g, embed)
		#print(mout)
