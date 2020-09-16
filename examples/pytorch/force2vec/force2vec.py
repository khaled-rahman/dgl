import dgl,torch
from dgl.sparse import _gsddmmspmm, _gsddmm, _gspmm
from dgl.data.citation_graph import load_cora
import sys, time
import numpy as np

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
	print("Total Kernel Time:", totalktime, "seconds")
	return embed

def f2vusingdefault(batchgraphs, embed, iterations=5):
	it = 0
	totaltime = 0
	while it < iterations:
		start = time.time()
		#SDDMM operation
		X = _gsddmm(batchgraphs._graph, 'dot', embed, embed, lhs_target='u', rhs_target='v')
		#non-linear transformation
		Y = 1. / (1 + np.exp(-X))
		#SPMM operation
		end = time.time()
		totalktime += end - start
		it += 1
	print("Total Kernel Time:", totalktime, "seconds")
	return embed

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

if __name__ == "__main__":
	if len(sys.argv) > 1:
		data = load_cora(".")
		graph = data[0]
		N = len(graph.nodes())
		embed = torch.rand(N, N)
		#need to check batch processing ...
		bgraphs = batch_process(graph)
		#print(graph)
		#f2valgo(graph, embed)
	else:
		print("Simple SDDMMSPMM Test:")
		g = dgl.graph(([0, 0, 1, 1, 2, 3], [1, 2, 2, 4, 3, 4]))
		#g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
		embed = torch.rand(5, 5)
		#print(embed)
		output = _gsddmmspmm(g._graph, "mul", embed, embed, "u", "v")
		print("SPDDMMSPMM Kernel:")
		print(output)
		print("Manual SPDDMM+SPMM:")
		#print(embed)
		mout = gsddmmspmmkerneltest(g, embed)
		print(mout)
