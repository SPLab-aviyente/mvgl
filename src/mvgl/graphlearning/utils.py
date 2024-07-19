"""This module includes utility functions used during learning graph structures.
"""

import numpy as np
import networkx as nx

from scipy.sparse import csr_array

def rowsum_mat(n):
    r"""Construct row-sum matrix for a symmetric zero-diagonal matrix.

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
    scipy.sparse.csr_array
        Matrix to be used in row-sum calculation.
    """

    i, j = np.triu_indices(n, k=1)
    M = len(i)
    rows = np.concatenate((i, j))
    cols = np.concatenate((np.arange(M), np.arange(M)))

    return csr_array((np.ones((2*M, )), (rows, cols)), shape=(n, M))

def vectorize_a_graph(G):
    """Get the upper triangular part of the adjacency of a graph as a vector. 

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    np.array
        Output vector as a numpy array.
    """
    n_nodes = G.number_of_nodes()

    w = nx.to_numpy_array(G)[np.triu_indices(n_nodes, k=1)]

    return w

def calc_data_vecs(X, normalize=False, S=None):
    r"""Calculate data vectors of a set of graph signal data matrices.

    Let :math:`\mathbf{X} \in \mathbb{R}^{n \\times p}` be a data matrix whose
    columns are graph signals on a graph with :math:`n` nodes. This function
    calculates the vector :math:`\mathbf{x} \in \mathbb{R}^{n(n-1)/2}`:

    .. math::
        \mathbf{x} = 2\mathrm{upper}(\mathbf{K}) 
            - \mathbf{S}^\\top \mathrm{diag}(\mathbf{K}),
    
    where :math:`\mathbf{K} = \mathbf{X}\mathbf{X}^\\top`, :math:`\mathbf{S}` is
    the row-sum matrix, :math:`\mathrm{upper}` is an operator returning upper
    triangular of part of the input matrix as a vector, and
    :math:`\mathrm{diag}` is an operator returning the diagonal part of the
    input matrix as a vector. :math:`\mathbf{x}` can be used to calculate the
    smoothness of :math:`\mathbf{X}` in a vectorized form: 

    .. math::
        \mathrm{tr}(\mathbf{X}^\\top\mathbf{\mathbf{L}}\mathbf{X}) = 
            \mathbf{x}^\\top\mathrm{upper}(\mathbf{L}),
    where :math:`\mathbf{L}` is the Laplacian of the graph.

    Parameters
    ----------
    X : np.array or list of np.array
        The matrix or a set of matrices whose data vectors will be calculated.
        If list, data vectors calculated for each matrix independently.
    normalize : bool, optional
        Whether to normalize the calculated data vectors before returning them.
        Normalization is done by dividing the calculated data vector to its
        infinity norm. This can be helpful when smoothness terms is used with
        other terms (*e.g.* regularization terms, smoothness terms calculated
        from other data matrices) such that normalization may ensure scale of
        the terms are in a similar range. By default False.
    S : scipy.sparse.csr_array, optional
        Preconstructed row-sum matrix. If None, it is constructed by
        :meth:`rowsum_mat`. By default None

    Returns
    -------
    np.array or list of np.array
        Calculated data vectors.
    """

    # Input check
    if not isinstance(X, list):
        X = [X]

    if S is None:
        S = rowsum_mat(X[0].shape[0])
    
    normalizer = -1
    n_views = len(X)
    data_vecs = []
    for v in range(n_views):
        # Calculate data vector for each data matrix in X
        K = X[v]@X[v].T
        k = K[np.triu_indices_from(K, k=1)]
        d = K[np.diag_indices_from(K)]
        data_vecs.append(2*k - S.T@d)

        if np.ndim(data_vecs[-1]) == 1:
            data_vecs[-1] = data_vecs[-1][:, None]

        data_vec_inf_norm = np.max(np.abs(data_vecs[-1]))
        if data_vec_inf_norm > normalizer:
            normalizer = data_vec_inf_norm

    # Normalize the data vector by its infinity norm
    if normalize:
        for v in range(n_views):
            data_vecs[v] /= normalizer

    return data_vecs if n_views > 1 else data_vecs[0]