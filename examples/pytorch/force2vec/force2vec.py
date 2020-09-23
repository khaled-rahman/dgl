import dgl,torch
from dgl.sparse import _gsddmmspmm, _gsddmm, _gspmm
from dgl.data.citation_graph import load_cora, load_citeseer, load_pubmed
import sys, time, random
import numpy as np
from math import log, exp, isnan
from scipy.io import mmread,mminfo
import networkx as nx
import argparse

def cacheflush():
	print("Cache Refreshing...")
	BIGNUM = 100000
	L1 = []
	L2 = []
	for i in range(BIGNUM):
		L1.insert(0, i)
		L2.insert(0, i+1)
	print("Done!")

SM_BOUND = 5.0
SM_TABLE_SIZE = 2048
SM_RESOLUTION = SM_TABLE_SIZE / (2.0 * SM_BOUND)

#create sigmoid table to make it equivalent
def init_SM_TABLE():
	sm_table = [0.0 for i in range(SM_TABLE_SIZE)]
	for i in range(SM_TABLE_SIZE):
		x = 2.0 * SM_BOUND * i / SM_TABLE_SIZE - SM_BOUND
		sm_table[i] = 1.0 / (1.0 + exp(-x))
	return sm_table

sm_table = init_SM_TABLE()

def FastSigmoid(v, stable):
	if isnan(v):
		return 1.0
	if v > SM_BOUND:
		return 1.0
	elif v < -SM_BOUND:
		return 0.0
	return stable[int((v + SM_BOUND) * SM_RESOLUTION)]

def allFastSigmoid(T, stable, add = 1.0):
	for i in range(len(T)):
		T[i] = add - FastSigmoid(T[i], stable)
	return T

	
def batch_process(graph, batchsize=256):
	batches = []
	b = 0
	N = len(graph.nodes())
	edges = [np.array(graph.edges()[0]),np.array(graph.edges()[1])]
	while b < N:
		bedges = []
		for i in range(b, min(b+batchsize, N)):
			bedges += [[edges[0][row], edges[1][row]] for row in range(len(edges[0])) if edges[0][row]==i]
		bedge_left = [int(i) for [i,j] in bedges]
		bedge_right = [int(j) for [i,j] in bedges]
		bgraph = dgl.graph((bedge_left,bedge_right))
		
		#discard subgraph with no edge
		if len(bgraph.edges()[0]) > 0:
			batches.append([bgraph, b , min(b+batchsize, N)])
		
		#batches.append(bgraph)
		b += batchsize
	return batches

def negativeSamples(graph, start, end, nsamples = 2):
	random.seed(1)
	left = []
	right = []
	N = len(graph.nodes())
	for i in range(start, end):
		for j in range(nsamples):
			left.append(i)
			v = random.randint(0, N-1)
			right.append(v)
	ngraph = dgl.graph((left, right))
	return ngraph

#commonRepulsive function
def calcRepulsive(graph, embed, sm_table, V):
	X = _gsddmm(graph._graph, 'dot', embed[:V], embed[:V], lhs_target='u', rhs_target='v')
	Y = allFastSigmoid(X, sm_table, 0)
	output = _gspmm(graph._graph.reverse(), "mul", "sum", embed[:V], Y)[0]
	return output

#update function
def update(s, e, embed, upembed):
	for i in range(s, e):
		for j in range(len(embed[i])):
			embed[i][j] = upembed[i][j].clone()
	return embed

#combined sddmmspmm kernel
def f2vusingsigmoid(batchgraphs, embed, iterations = 1, lrate=1.0):
	it = 0
	#for repulsive force
	#sm_table = init_SM_TABLE()
	totaltime = 0
	kerneltime = 0
	while it < iterations:
		start = time.time()
		for [graph, s, e] in batchgraphs:
			ngraph = negativeSamples(graph, s, e)
			nV = len(ngraph.nodes())
			#print(ngraph, ngraph.edges())
			kstart = time.time()
			outputa = _gsddmmspmm(graph._graph, "mul", embed, embed, "u", "v", 1)
			kend = time.time()
			outputr = calcRepulsive(ngraph, embed, sm_table, nV)
			embed[s:e] = lrate * outputa[s:e].clone()
			embed[s:e] = embed[s:e] + lrate * outputr[s:e]
			kerneltime += kend - kstart
		end = time.time()
		totaltime += end - start
		it += 1
	print("Total F2V Time:", totaltime, "seconds", " and Total Kernel Time:", kerneltime)
	return embed

def f2vusingtdistribution(batchgraphs, embed, iterations = 1, lrate=1.0):
	it = 0
	totalktime = 0
	while it < iterations:
		for [graph, s, e] in batchgraphs:
			start = time.time()
			outputa = _gsddmmspmm(graph._graph, "mul", embed, embed, "u", "v", 2)
			end = time.time()
			embed = embed + lrate * outputa
			totalktime += end - start
		it += 1
	print("Total F2V Kernel Time:", totalktime, "seconds")
	return embed
	
def scale(v):
	if v > SM_BOUND:
		return SM_BOUND
	elif v < -SM_BOUND:
		return -SM_BOUND
	return v

def allscale(T):
	for i in range(len(T)):
		for j in range(len(T[i])):
			T[i][j] = scale(T[i][j])
	return T

#gdl:sddmm+transformation+spmm kernel
def dglusingsigmoid(batchgraphs, embed, iterations=1, lrate=1.0):
	it = 0
	totaltime = 0
	kerneltime = 0
	#sm_table = init_SM_TABLE()
	#print(sm_table)
	while it < iterations:
		start = time.time()
		for [graph, s, e] in batchgraphs:
			ngraph = negativeSamples(graph, s, e)
			nV = len(ngraph.nodes())
			#just to make sure: dim(sparse graph) == dim(embedding)
			pV = len(graph.nodes())
			#SDDMM operation
			kstart = time.time()
			X = _gsddmm(graph._graph, 'dot', embed[:pV], embed[:pV], lhs_target='u', rhs_target='v')
			#non-linear transformation
			#Y = 1.0 - 1.0 / (1 + torch.exp(-X))
			Y = allFastSigmoid(X, sm_table)
			#SPMM operation
			outputa = _gspmm(graph._graph.reverse(), "mul", "sum", embed[:pV], Y)[0]
			kend = time.time()
			outputr = calcRepulsive(ngraph, embed, sm_table, nV)
			embed[s:e] = lrate * outputa[s:e]
			embed[s:e] = embed[s:e] + lrate * outputr[s:e]
			kerneltime += kend - kstart
		end = time.time()
		totaltime += end - start
		it += 1
	print("Total GDL Total Time:", totaltime, "seconds", " and Total Kernel Time:", kerneltime)
	return embed

def dglusingtdistribution(batchgraphs, embed, iterations=1, lrate = 1.0):
	it = 0
	totalktime = 0
	while it < iterations:
		for [graph, s, e] in batchgraphs:
			#just to make sure: dim(sparse graph) == dim(embedding)
			pV = len(graph.nodes())
			start = time.time()
			#SDDMM
			D = _gsddmm(graph._graph, "sub", embed[:pV], embed[:pV], lhs_target='u', rhs_target='v')
			d = _gsddmm(graph._graph, 'dot', D, D, lhs_target='e', rhs_target='e')
			#NonlinearTransformation
			E = - 2.0 / (1.0 + d)
			#SDDMM
			E = _gsddmm(graph._graph, "mul", D, E, lhs_target='e', rhs_target='e')
			#scaling
			E = allscale(E)
			#SPMM
			outputa = _gspmm(graph._graph.reverse(), "copy_rhs", "sum", E, E)[0]
			end = time.time()
			embed[:pV] = embed[:pV] + lrate * outputa[:pV]
			totalktime += end - start
		it += 1
	print("Total GDL Kernel Time:", totalktime, "seconds")
	return embed

#manual result verification
def gsddmmspmmkerneltest(graph, A):
	sm_table = init_SM_TABLE()
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
	#Y = 1.0 / (1.0 + np.exp(-X))
	#Z = np.array([[0 if S[i][j] == 0 else 1-Y[i][j] for j in range(N)] for i in range(N)])
	Z = allscale(X)
	#print(Z)
	O = np.matmul(Z, A)
	return O

def transposeGraph(graph):
	edges = graph.edges()
	source = edges[0]
	dest = edges[1]
	tGraph = dgl.graph((dest, source))
	return tGraph

def f2vfunctions(graph, embed, ftype, it, lr):
	if ftype == 1:
		f2voutput = f2vusingsigmoid(graph, embed, it, lr)
	else:
		f2voutput = f2vusingtdistribution(graph, embed, it, lr)
	return f2voutput


def dglfunctions(graph, embed, ftype, it, lr):
	if ftype == 1:
		dgloutput = dglusingsigmoid(graph, embed, it, lr)
	else:
		dgloutput = dglusingtdistribution(graph, embed, it, lr)
	return dgloutput

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Demonstration of Force2Vec using fused kernels and DGL SDDMM and SpMM kernels', add_help=True)
	parser.add_argument('-g', '--g', required=False, type=str, default="simple", help='Input graph name: cora, citeseer, pubmed')
	parser.add_argument('-d', '--d', required=False, type=int, default=128, help='Embedding dimension, default:128')
	parser.add_argument('-t', '--t', required=False, type=int, default=1, help='Similarity function type, default:1(sigmoid)')
	parser.add_argument('-r', '--r', required=False, type=float, default=1.0, help='Learning rate, default:1.0')
	parser.add_argument('-it', '--it', required=False, type=int, default=1, help='Iterations, default:1')
	parser.add_argument('-b', '--b', required=False, type=int, default=256, help='Batch Size, default:256')
	parser.add_argument('-p', '--p', required=False, type=str, default="", help='Path to mtx graph format, default:None')

	args = parser.parse_args()
	graph = args.g
	dim = args.d
	ftype = args.t
	lrate = args.r
	bsize = args.b
	it = args.it
	path = args.p
	
	if len(path) > 0:
		G = mmread(path)
		nxgraph = nx.Graph(G)
		graph = dgl.from_networkx(nxgraph)
	elif graph == "simple":
		graph = dgl.graph(([0, 0, 1, 1, 2, 3], [1, 2, 2, 4, 3, 4]))
	elif graph == "citeseer":
		data = load_citeseer(".")
		graph = data[0]
	elif graph == "pubmed":
		data = load_pubmed(".")
		graph = data[0]
	else:
		data = load_cora(".")
		graph = data[0]
	N = len(graph.nodes())
	print("#Nodes:", N, "#Edges:", len(graph.edges()[0]))
	embed = torch.rand(N, dim)
	#print(embed)
	#need to check batch processing ...
	print("Creating batch graphs...")
	bgraphs = batch_process(graph, bsize)
	print("Done!")
	#cacheflush()
	f2voutput = f2vfunctions(bgraphs, embed.clone(), ftype, it, lrate)
	print("Fused SDDMM+SPMM Kernel:")
	print(f2voutput)
	#dgloutput = dglfunctions(bgraphs, embed.clone(), ftype, it, lrate)
	#print("DGL: SDDMMSPMM+Transformation+SPMM")
	#print(dgloutput)
	#out = gsddmmspmmkerneltest(graph, embed)
	#print(out)
