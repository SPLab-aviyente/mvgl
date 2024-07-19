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

def _vi_step_mat(zi, wij, beta, rho, regularizer):
    n_views = len(zi)
    n_pairs = len(zi[0]) # number of node pairs

    y = np.zeros((n_pairs, int(n_views*(n_views-1)/2)))
    col = -1
    for v in range(n_views):
        for vv in range(v+1, n_views):
            col += 1
            y[:, col] = np.squeeze(zi[v] - zi[vv] - wij[v][vv]/rho)

    scale = beta/rho
    if regularizer == "l2sq":
        y = 1/(1 + 2*scale)*y
    elif regularizer == "l1":
        y = np.sign(y)*np.maximum(np.abs(y) - scale, 0) 
    elif regularizer == "l2":
        row_norms = np.linalg.norm(y, axis=1, keepdims=True)
        y = y*np.maximum(0, 1 - scale / (1e-16 + row_norms))

    out = []
    col = -1
    for v in range(n_views):
        out.append([])
        for vv in range(n_views):
            if vv > v:
                col += 1
                out[v].append(y[:, col][..., None])
            else:
                out[v].append(None)

    return out

def _zi_step(li, yi, vij, wij, zi, i, rho):
    n_views = len(vij)

    out = li - yi/rho
    for j in range(n_views):
        if i < j:
            out += vij[i][j] + zi[j] + wij[i][j]/rho
        elif i > j:
            out += -vij[j][i] + zi[j] - wij[j][i]/rho

    out /= n_views
    
    out[out>=0] = 0
    return out

def _objective(data_vecs, zi, S, alpha, beta, regularizer):
    n_views = len(data_vecs)
    n_pairs = len(zi[0]) # number of node pairs

    y = np.zeros((n_pairs, int(n_views*(n_views-1)/2)))
    col = -1
    for v in range(n_views):
        for vv in range(v+1, n_views):
            col += 1
            y[:, col] = np.squeeze(zi[v] - zi[vv])

    result = 0
    for view in range(n_views):
        result += (data_vecs[view]).T@zi[view] # smoothness
        result += alpha*np.linalg.norm(S@zi[view])**2 # degree term
        result += alpha*np.linalg.norm(zi[view])**2 # sparsity term

    # consensus term
    if regularizer == "l2sq":
        result += beta*np.linalg.norm(y)**2 
    elif regularizer == "l2":
        result += beta*np.sum(np.linalg.norm(y, axis=1))
    elif regularizer == "l1":
        result += beta*np.linalg.norm(y, 1) 

    return result.item()

def _run(data_vecs, alpha, beta, regularizer, S=None, rho=1, max_iter=1000):
    n_views = len(data_vecs)
    n_pairs = len(data_vecs[0]) # number of node pairs
    n_nodes = (1 + np.sqrt(8*n_pairs + 1))//2 # number of nodes

    if S is None:
        S = utils.rowsum_mat(int(n_nodes))

    # Initialization
    yi = []
    zi = []
    wij = []
    for v in range(n_views):
        zi.append(np.zeros((n_pairs, 1)))
        yi.append(np.zeros((n_pairs, 1)))
        wij.append([])
        for vv in range(n_views):
            if vv > v:
                wij[v].append(np.zeros((n_pairs, 1)))
            else:
                wij[v].append(None)

    # ADMM iterations
    objective = []
    for iter in range(max_iter):

        # li-step
        li = [None]*n_views
        for v in range(n_views):
            li[v] = _li_step(yi[v], zi[v], data_vecs[v], S, alpha, rho, n_nodes)

        # vi-step
        vij = _vi_step_mat(zi, wij, beta, rho, regularizer)

        # zi-step
        for v in range(n_views):
            zi[v] = _zi_step(li[v], yi[v], vij, wij, zi, v, rho)

        # update Lagrangian multipliers
        for v in range(n_views):
            yi[v] += rho*(zi[v] - li[v])
            for vv in range(v+1, n_views):
                wij[v][vv] += rho*(vij[v][vv] - zi[v] + zi[vv])
        

        objective.append(
            _objective(data_vecs, zi, S, alpha, beta, regularizer)
        )

        if (iter > 5) and (np.abs(objective[-1] - objective[-2]) < 1e-4):
            break
    
    # Return adjacency matrices
    for v in range(n_views):
        zi[v][zi[v] > -1e-4] = 0 # Remove very small edges
        zi[v] = np.abs(zi[v])

    return zi

def learn_multiview_graph(X, alpha, beta, regularizer, view_density=None, 
                          similarity=None, param_acc=0.01, n_iters=50):
    # Input check
    if not isinstance(X, list):
        raise Exception("Mutliple sets of graph signals must be provided " 
                        "when learning multiple graphs.")  

    # Variable initialization
    n_views = len(X)
    n_nodes = X[0].shape[0]
    S = utils.rowsum_mat(n_nodes)
    data_vecs = utils.calc_data_vecs(X, normalize=True, S=S)

    view_densities = []
    similarities = []
    for iter_indx in range(n_iters):

        # Learn the graph
        st = time.time()
        w_hat = _run(data_vecs, alpha, beta, regularizer, S=S)
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

            if converged: 
                break

    params = {"alpha": alpha, "beta": beta, "regularizer": regularizer}
    return w_hat, params, run_time
    

