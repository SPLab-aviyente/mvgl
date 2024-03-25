import numpy as np
import networkx as nx

from mvgl.graphlearning import utils

def test_rowsum_mat_construction():
    A = np.array([
        [0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 1, 0]
    ])
    n = A.shape[0]
    a = A[np.triu_indices_from(A, k=1)]

    S = utils.rowsum_mat(n)

    expected = np.sum(A, axis=1)
    actual = S@a

    msg = "Row-sum matrix returns wrong sums."
    assert np.sum(np.abs(actual - expected)) < 1e-8, msg

def test_graph_vectorization():
    A = np.array([
        [0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 1, 0]
    ])
    G = nx.from_numpy_array(A, edge_attr=False)
    
    expected = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    actual = utils.vectorize_a_graph(G)

    msg = "Graph vectorization doesn't match with expected output."
    assert np.sum(np.abs(actual - expected)) < 1e-8, msg