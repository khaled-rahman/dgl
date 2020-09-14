import dgl,torch
from dgl.sparse import _gsddmmspmm

def batch_process(graphs):
	batches = []
	return batches

if __name__ == "__main__":
	g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
	embed = torch.rand(5, 5)
	output = _gsddmmspmm(g._graph, "mul", embed, embed, "u", "v")
	print(output)
