import dgl,torch
from dgl.sparse import _gsddmmspmm
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

if __name__ == "__main__":
	if len(sys.argv) > 1:
		data = load_cora(".")
		graph = data[0]
		N = len(graph.nodes())
		embed = torch.rand(N, N)
		#need to check batch processing ...
		#bgraphs = batch_process(graph)
		#print(graph)
		f2valgo(graph, embed)
	else:
		print("Simple SDDMMSPMM Test:")
		g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
		embed = torch.rand(5, 5)
		output = _gsddmmspmm(g._graph, "mul", embed, embed, "u", "v")
		print(output)
