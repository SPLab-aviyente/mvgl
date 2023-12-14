from mvgl.data import graph
from mvgl import utils

def test_graph_vectorization_shape():
    n_nodes = 50
    G = graph.gen_consensus_graph(n_nodes, graph_generator="er", p=0.3)
    
    w = utils.vectorize_graphs(G)

    # check dimension
    n_nodes = 50
    n_pairs = n_nodes*(n_nodes-1)//2
    assert w.shape == (n_pairs,), "Graph vectorization gives wrong shape."