import numpy as np

from sklearn.metrics import f1_score, average_precision_score

def _to_mv(data):
    if not isinstance(data, list):
        data = [data]

    return data

def _one(func, data):
    data = _to_mv(data)

    n_views = len(data)
    result = np.zeros(n_views)

    for v in range(n_views):
        result[v] = func(data[v])

    return result if n_views>1 else result[0]

def _one_to_one(func, data1, data2):
    data1 = _to_mv(data1)
    data2 = _to_mv(data2)

    n_views = len(data1)
    result = np.zeros(n_views)

    for v in range(n_views):
        result[v] = func(data1[v], data2[v])

    return result if n_views>1 else result[0]

def _one_to_all(func, data1, data2):
    data1 = _to_mv(data1)
    data2 = _to_mv(data2)

    n_views1 = len(data1)
    n_views2 = len(data2)
    result = np.zeros((n_views1, n_views2))
    for v1 in range(n_views1):
        for v2 in range(n_views2):
            result[v1, v2] = func(data1[v1], data2[v2])

    if n_views1 == 1 and n_views2 == 1:
        return result[0, 0]
    else:
        return result.squeeze()

def density(w):
    """Calculate density of a given set of graphs.

    Parameters
    ----------
    w : np.array or list of np.array
        Upper triangular part of the adjacency matrices of the graphs. If not 
        a list, it is assumed that a single graph is given.

    Returns
    -------
    densities : float or np.array
        Graph densities. If a single graph is given, a single value is returned.
    """
    def _density(d):
        return np.count_nonzero(d)/len(d)

    return _one(_density, w)

def correlation(w1, w2):
    """Calculate correlation between two sets of graphs.

    Given two set of graphs, :math:`\mathcal{G} = \{G^i\}_{i=1}^N`, and
    :math:`\mathcal{H} = \{H^i\}_{i=1}^M`, this function calculates correlation
    between pair of graphs, :math:`(G^i, H^j)` for all :math:`i` and \math:`j`.

    Parameters
    ----------
    w_1 : np.array or list of np.array
        Upper triangular part of the adjacency matrices of the graphs in first
        set. If np.array, it is assumed that :math:`N=1`.
    w_2 : np.array or list of np.array
        Upper triangular part of the adjacency matrices of the graphs in second
        set. If np.array, it is assumed that :math:`M=1`.

    Returns
    -------
    corrs : float or np.array
        (N, M) dimensional array of correlation. If :math:`N=1`, it is (M,)
        dimensional. If :math:`M=1`, it is (N,) dimensional. If :math:`N=1` and 
        :math:`M=1`, it is a single value.  
    """
    def _correlation(d1, d2):
        return np.corrcoef(np.squeeze(d1>0).astype(int), np.squeeze(d2>0).astype(int))[0,1]

    return _one_to_all(_correlation, w1, w2)

def f1(w_gt, w_hat):
    """Calculate F1-score between ground truth and learned graphs. 

    Given ground truth graphs, :math:`\mathcal{G} = \{G^i\}_{i=1}^N`, 
    and learned graphs :math:`\widehat{\mathcal{G}} = \{\widehat{G}^i\}_{i=1}^N`, 
    this function calculates :math:`F1(G^i, \widehat{G}^i)` for each :math:`i`. 

    Parameters
    ----------
    w_gt : np.array or list of np.array
        Upper triangular part of the adjacency matrices of ground truth graphs.
        If not a list of `np.array`'s, it is assumed that :math:`N=1`.
    w_hat : np.array or list of np.array
        Upper triangular part of the adjacency matrices of learned graphs.
        If not a list of `np.array`'s, it is assumed that :math:`N=1`.
    
    Returns
    -------
    f1s : float or np.array
        Calculated F1 scores. If :math:`N=1`, it is a single F1-score.
    """
    def _f1(w1, w2):
        return f1_score((w1 > 0).astype(int).squeeze(), 
                        (w2 > 0).astype(int).squeeze())

    return _one_to_one(_f1, w_gt, w_hat)

def auprc(w_gt, w_hat):
    """Calculate AUPRC score between ground truth and learned graphs. 

    Given ground truth graphs, :math:`\mathcal{G} = \{G^i\}_{i=1}^N`, 
    and learned graphs :math:`\widehat{\mathcal{G}} = \{\widehat{G}^i\}_{i=1}^N`, 
    this function calculates :math:`AUPRC(G^i, \widehat{G}^i)` for each :math:`i`. 

    Parameters
    ----------
    w_gt : np.array or list of np.array
        Upper triangular part of the adjacency matrices of ground truth graphs.
        If not a list of `np.array`'s, it is assumed that :math:`N=1`.
    w_hat : np.array or list of np.array
        Upper triangular part of the adjacency matrices of learned graphs.
        If not a list of `np.array`'s, it is assumed that :math:`N=1`.
    
    Returns
    -------
    auprcs : float or np.array
        Calculated AUPRC scores. If :math:`N=1`, it is a single AUPRC score.
    """
    def _auprc(w1, w2):
        return average_precision_score(
            (w1 > 0).astype(int).squeeze(), w2.squeeze()
        )

    return _one_to_one(_auprc, w_gt, w_hat)