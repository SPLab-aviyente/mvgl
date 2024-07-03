import time

import numpy as np

from mvgl.graphlearning import metrics, utils

def _project_to_hyperplane(v, n):
    """ Project v onto the hyperplane defined by np.sum(v) = -n
    """
    return v - (n + np.sum(v))/(len(v))

def _l_step(y, w, data_vec, S, alpha, rho, n):
    y = rho*w - data_vec - y

    a = 4*alpha + rho
    b = 2*alpha
    c1 = 1/a
    c2 = b/(a*(a+n*b-2*b))
    c3 = (4*b**2)/(a*(a+(n-2)*b)*(a+2*(n-1)*b))

    y = c1*y - c2*(S.T@(S@y)) + c3*np.sum(y)

    return _project_to_hyperplane(y, n)

def _objective(data_vec, l, S, alpha):

    result = data_vec.T@l # smoothness
    result += alpha*np.linalg.norm(S@l)**2 # degree term
    result += alpha*np.linalg.norm(l)**2 # sparsity term

    return result.item()

def _run(data_vec, alpha, S=None, rho=1, max_iter=1000):
    n_pairs = len(data_vec) # number of node pairs
    n_nodes = (1 + np.sqrt(8*n_pairs + 1))//2 # number of nodes

    if S is None:
        S = utils.rowsum_mat(int(n_nodes))

    # Initialization
    w = np.zeros((n_pairs, 1)) # slack variable
    y = np.zeros((n_pairs, 1)) # Lagrange multiplier

    # ADMM iterations
    objective = []
    for iter in range(max_iter):
        # Update l
        l = _l_step(y, w, data_vec, S, alpha, rho, n_nodes)

        # Update slack variable
        w = l + y/rho
        w[w>0] = 0
        
        # Update Lagrange multiplier
        y += rho*(l - w)

        objective.append(_objective(data_vec, l, S, alpha))

        if (iter > 5) and (np.abs(objective[-1] - objective[-2]) < 1e-4):
            break

    w[w>-1e-4] = 0 # Remove very small edges

    # return adjacency matrix
    return np.abs(w)

def learn_a_single_graph(X, alpha, density=None, param_acc=0.01, n_iters=50):
    r"""Learn a single graph from a set of smooth graph signals.

    The method learns the graph based on optimization problem proposed in [1].
    The optimization is performed using ADMM.

    The function also has an experimental hyperparameter search procedure which
    tries to optimize hyperparameter of the learning algorithm to obtain a graph
    with desired properties. By default this procedure isn't applied if its
    parameters are not set.

    .. warning::
        The hyperparameter search procedure is experimental and hasn't been 
        tested extensively. It might not return expected output, so use with care.

    Parameters
    ----------
    X : np.array
        Data matrix whose columns are smooth graph signals.
    alpha : float
        The hyperparameter controling the density of the learned graph. Its
        larger values learn denser graphs.
    density : float, optional
        The hyperparameter search procedure optimizes `alpha` to make density of
        learned graph to be close to this value. If None, `alpha` isn't
        optimized. By default None.
    param_acc : float, optional
        Hyperparameter search procedure assumed to be converged when the
        difference between desired and learned graph properties is smaller than
        this value. By default 0.01.
    n_iters : int, optional
        Maximum number of iterations for hyperparameter search, by default 50.

    Returns
    -------
    w_hat : np.array
        Upper triangular part of adjacency matrix of the learned graph.
    params : dict
        Hyperparameter values.
    run_time : float
        Time passed to learn the graphs for given values of hyperparameters.

    References
    ----------
    .. [1] Dong, Xiaowen, et al. "Learning Laplacian matrix in smooth graph
        signal representations." IEEE Transactions on Signal Processing 64.23
        (2016): 6160-6173.
    """
    n_nodes = X.shape[0]

    S = utils.rowsum_mat(n_nodes)
    data_vec = utils.calc_data_vecs(X, normalize=True, S=S)

    densities = []
    for iter_indx in range(n_iters):
        # Learn the graph
        st = time.time()
        w_hat = _run(data_vec, alpha, S=S)
        run_time = time.time() - st

        is_param_updated = False 
        if density is not None:
            density_hat = metrics.density(w_hat)
            diff = density_hat - density

            if np.abs(diff) > param_acc:
                alpha = alpha*density/density_hat
                is_param_updated = True 

            densities.append(density)

        if not is_param_updated:
            break

        # Check convergence of parameter search
        if iter_indx > 6:
            converged = True
            if density is not None:
                converged = (
                    converged and 
                    (np.mean(densities[-5:]) - density_hat) < 1e-4
                )
            if converged:
                break

    params = {"alpha": alpha}

    return w_hat, params, run_time