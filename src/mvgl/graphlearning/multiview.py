import time

import numpy as np

from mvgl.graphlearning import metrics, utils

def _project_to_hyperplane(v, n):
    """ Project v onto the hyperplane defined by np.sum(v) = -n
    """
    return v - (n + np.sum(v))/(len(v))

def _li_step(yi, zi, data_veci, S, alpha, rho, n):
    y = yi + rho*zi - data_veci

    a = 4*alpha + rho
    b = 2*alpha
    c1 = 1/a
    c2 = b/(a*(a+n*b-2*b))
    c3 = (4*b**2)/(a*(a+(n-2)*b)*(a+2*(n-1)*b))

    y = c1*y - c2*(S.T@(S@y)) + c3*np.sum(y)

    return _project_to_hyperplane(y, n)

def _vi_step_mat(zi, z, wi, beta, rho, consensus):
    n_views = len(zi)
    n_pairs = len(z) # number of node pairs

    y = np.zeros((n_pairs, n_views))
    for v in range(n_views):
        y[:, v] = np.squeeze(zi[v] - z - wi[v]/rho)

    scale = beta/rho
    if consensus == "l2sq":
        y = 1/(1 + 2*scale)*y
    elif consensus == "l1":
        y = np.sign(y)*np.maximum(np.abs(y) - scale, 0) 
    elif consensus == "l2":
        row_norms = np.linalg.norm(y, axis=1, keepdims=True)
        y = y*np.maximum(0, 1 - scale / (1e-16 + row_norms))

    out = []
    for v in range(n_views):
        out.append(y[:, v][..., None])

    return out

def _l_step(z, y, gamma, rho):
    y = z + y/rho
    scale = gamma/rho
    return np.sign(y)*np.maximum(np.abs(y) - scale, 0)

def _zi_step(li, yi, vi, z, wi, rho):
    out = (li - yi/rho + vi + z + wi/rho)/2
    out[out>=0] = 0
    return out

def _z_step(zi, vi, wi, rho, l=None, y=None):
    n_views = len(zi)
    n_pairs = len(zi[0])
    
    out = np.zeros((n_pairs, 1)) 
    for v in range(n_views):
        out += (zi[v] - vi[v] - wi[v]/rho)

    if (l is not None) and (y is not None):
        out += l - y/rho
        out /= (n_views+1)
    else:
        out /= n_views

    out[out>=0] = 0
    return out

def _objective(data_vecs, zi, z, S, alpha, beta, gamma, consensus):
    n_views = len(data_vecs)
    n_pairs = len(z) # number of node pairs

    y = np.zeros((n_pairs, n_views))
    for v in range(n_views):
        y[:, v] = np.squeeze(zi[v] - z)

    result = 0
    for view in range(n_views):
        result += (data_vecs[view]).T@zi[view] # smoothness
        result += alpha*np.linalg.norm(S@zi[view])**2 # degree term
        result += alpha*np.linalg.norm(zi[view])**2 # sparsity term

    # consensus term
    if consensus == "l2sq":
        result += beta*np.linalg.norm(y)**2 
    elif consensus == "l2":
        result += beta*np.sum(np.linalg.norm(y, axis=1))
    elif consensus == "l1":
        result += beta*np.linalg.norm(y, 1) 

    if gamma:
        result += gamma*np.linalg.norm(z, 1) # sparsity term for consensus

    return result.item()

def _run(data_vecs, alpha, beta, consensus, S=None, gamma=None, rho=1, max_iter=1000):
    n_views = len(data_vecs)
    n_pairs = len(data_vecs[0]) # number of node pairs
    n_nodes = (1 + np.sqrt(8*n_pairs + 1))//2 # number of nodes

    if S is None:
        S = utils.rowsum_mat(int(n_nodes))

    # Initialization
    yi = []
    zi = []
    wi = []
    for v in range(n_views):
        zi.append(np.zeros((n_pairs, 1)))
        yi.append(np.zeros((n_pairs, 1)))
        wi.append(np.zeros((n_pairs, 1)))
    z = np.zeros((n_pairs, 1))

    if gamma:
        y = np.zeros((n_pairs, 1))
    else:
        y = None

    # ADMM iterations
    objective = []
    for iter in range(max_iter):

        # li-step
        li = [None]*n_views
        for v in range(n_views):
            li[v] = _li_step(yi[v], zi[v], data_vecs[v], S, alpha, rho, n_nodes)

        # vi-step
        vi = _vi_step_mat(zi, z, wi, beta, rho, consensus)

        # l-step
        if gamma:
            l = _l_step(z, y, gamma, rho)
        else:
            l = None

        # zi-step
        for v in range(n_views):
            zi[v] = _zi_step(li[v], yi[v], vi[v], z, wi[v], rho)

        # z-step
        z = _z_step(zi, vi, wi, rho, l, y)

        # update Lagrangian multipliers
        for v in range(n_views):
            yi[v] += rho*(zi[v] - li[v])
            wi[v] += rho*(vi[v] - zi[v] + z)
        
        if gamma:
            y += rho*(z-l)
        else:
            y = None

        objective.append(
            _objective(data_vecs, zi, z, S, alpha, beta, gamma, consensus)
        )

        if (iter > 5) and (np.abs(objective[-1] - objective[-2]) < 1e-4):
            break
    
    # Return adjacency matrices
    for v in range(n_views):
        zi[v][zi[v] > -1e-4] = 0 # Remove very small edges
        zi[v] = np.abs(zi[v])

    z[z>-1e-4] = 0 # Remove very small edges
    z = np.abs(z)

    return zi, z

def learn_multiview_graph(X, alpha, beta, consensus, gamma=None, view_density=None, 
                          consensus_density=None, similarity=None, param_acc=0.01, 
                          n_iters=50):
    r"""Learn a multiview graph from multiple sets of smooth graph signals.

    Assume we are given a set of data matrices :math:`\{\mathbf{X}^i\}_{i=1}^N`,
    where columns of :math:`\mathbf{X}^i \in \mathbb{R}^{n \\times p_i}` are
    *smooth* graph signals on an unkown graph :math:`G^i`. This function learns
    :math:`G^i`'s by minimizing smoothness of :math:`X^i`'s with respect to
    Laplacian matrices. During learning, it is assumed :math:`G^i`'s are related
    to each other through a shared structure represent by a *consensus* graph
    :math:`G`, which is also learned. 

    The function also has an experimental hyperparameter search procedure which
    tries to optimize hyperparameters of the learning algorithm to obtain graphs
    with desired properties. By default this procedure isn't applied if its
    parameters are not set.

    .. warning::
        The hyperparameter search procedure is experimental and hasn't been 
        tested extensively. It might not return expected output, so use with care.

    Parameters
    ----------
    X : list of np.array
        Data matrices.
    alpha : float
        The hyperparameter controling the density of the learned :math:`G^i`'s.
        Its larger values learn denser graphs.
    beta : float
        The hyperparameter controling how similar the learned :math:`G^i`'s are.
        Its larger values impose more similarity between graphs.
    consensus : {'l1', 'l2'}
        The norm to use for consensus term.
    gamma : float, optional
        The hyperparameter controling the density of the learned consensus
        graph. The density of the consensus graph is controled through an
        :math:`\ell_1`-norm term. If `gamma` is None, this term is removed.
        Otherwise, the larger values of `gamma` results in a sparser consensus
        graph. By default None
    view_density : float, optional
        The hyperparameter search procedure optimizes `alpha` to make mean
        density of learned :math:`G^i`'s to be close to this value. If None,
        `alpha` isn't optimized. By default None.
    consensus_density : float, optional
        The hyperparameter search procedure optimizes `gamma` to make density of
        the learned consensus graph to be close to this value. If None, `gamma`
        isn't optimized. By default None.
    similarity : float, optional
        The hyperparameter search procedure optimizes `beta` to make mean
        similarity of learned :math:`G^i`'s to be close to this value.
        Similarity is calculated using correlation between adjacencies. If None,
        `beta` isn't optimized. By default None.
    param_acc : float, optional
        Hyperparameter search procedure assumed to be converged when the
        difference between desired and learned graph properties is smaller than
        this value. By default 0.01.
    n_iters : int, optional
        Maximum number of iterations for hyperparameter search, by default 50.

    Returns
    -------
    w_hat : list of np.array
        Upper triangular part of adjacency matrices of learned view graphs as a
        list of vectors. 
    w : np.array
        Upper triangular part of adjacency matrix of learned consensus graph as
        a vector.
    params : dict
        Hyperparameter values.
    run_time : float
        Time passed to learn the graphs for given values of hyperparameters.

    Raises
    ------
    Exception
        If a single data matrix is provided, *i.e.* len(X) = 1.
    """

    # Input check
    if not isinstance(X, list):
        raise Exception("Mutliple sets of graph signals must be provided " 
                        "when learning multiple graphs.")  

    if gamma is None:
        consensus_density = None 

    # Variable initialization
    n_views = len(X)
    n_nodes = X[0].shape[0]
    S = utils.rowsum_mat(n_nodes)
    data_vecs = utils.calc_data_vecs(X, normalize=True, S=S)

    view_densities = []
    similarities = []
    consensus_densities = []
    for iter_indx in range(n_iters):

        # Learn the graph
        st = time.time()
        w_hat, w = _run(data_vecs, alpha, beta, consensus, S=S, gamma=gamma)
        run_time = time.time() - st

        is_params_updated = False

        # Update alpha
        if view_density is not None:
            view_densities_hat = metrics.density(w_hat)
            mean_density = np.mean(view_densities_hat)
            
            diff = mean_density - view_density
            if np.abs(diff) > param_acc:
                alpha = alpha*view_density/mean_density
                is_params_updated = True

            view_densities.append(mean_density)

        # Update beta
        if similarity is not None:
            similarities_hat = metrics.correlation(w_hat, w_hat)
            
            tot_similarity = np.sum(similarities_hat) - np.trace(similarities_hat)
            mean_similarity = tot_similarity/(n_views*(n_views - 1))

            diff = mean_similarity - similarity
            if np.abs(diff) > param_acc:
                beta = beta*similarity/mean_similarity
                is_params_updated = True

            similarities.append(mean_similarity)

        # Update gamma
        if consensus_density is not None:
            consensus_density_hat = metrics.density(w)
            diff = consensus_density_hat - consensus_density
            if np.abs(diff) > param_acc:
                gamma = gamma*(1-consensus_density)/(1-consensus_density_hat)
                is_params_updated = True

            consensus_densities.append(consensus_density_hat)

        # If no parameters are updated in this step, break the parameter search
        if not is_params_updated:
            break

        # Check convergence of parameter search
        if iter_indx > 6:
            converged = True 
            if view_density is not None:
                converged = (
                    converged and 
                    (np.mean(view_densities[-5:]) - mean_density) < 1e-4
                )
            if similarity is not None:
                converged = (
                    converged and
                    (np.mean(similarities[-5:] - mean_similarity)) < 1e-4
                )
            if consensus_density is not None:
                converged = (
                    converged and
                    (np.mean(consensus_densities[-5:]) - consensus_density_hat) < 1e-4
                )

            if converged: 
                break

    params = {"alpha": alpha, "beta": beta, "gamma": gamma, 
              "consensus": consensus}
    return w_hat, w, params, run_time
    

