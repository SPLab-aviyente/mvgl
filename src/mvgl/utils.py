import networkx as nx
import numpy as np

def vectorize_graphs(G):
    """Returns the upper triangular part of the adjacency matrix of a graph as 
    a vector.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    w : np.array
        Output vector as a numpy array.
    """
    n_nodes = G.number_of_nodes()

    w = nx.to_numpy_array(G)[np.triu_indices(n_nodes, k=1)]

    return w