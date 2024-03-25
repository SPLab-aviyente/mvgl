import numpy as np
import networkx as nx

from scipy.sparse import csr_array

def rowsum_mat(n):
    """Construct row-sum matrix for a symmetric zero-diagonal matrix.

    For a symmetric zero-diagonal matrix :math:`\mathbf{A} \in \mathbb{R}^{n \\times n}`, let 
    :math:`\mathbf{a} \in \mathbb{R}^{n(n-1)/2}` be its upper triangular part in a vector form. 
    Row sum matrix :math:`\mathbf{S} \in \mathbb{R}^{n \\times n(n-1)/2}` can be used as 
    :math:`\mathbf{S}\mathbf{a}` to calculate :math:`\mathbf{A}\mathbf{1}`, where 
    :math:`\mathbf{1}` is :math:`n` dimensional all-ones vector.
    
    Parameters
    ----------
    n : int
        Dimension of the matrix.

    Returns
    -------
    S : scipy.sparse.csr_array
        Matrix to be used in row-sum calculation.
    """

    i, j = np.triu_indices(n, k=1)
    M = len(i)
    rows = np.concatenate((i, j))
    cols = np.concatenate((np.arange(M), np.arange(M)))

    return csr_array((np.ones((2*M, )), (rows, cols)), shape=(n, M))

def vectorize_a_graph(G):
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