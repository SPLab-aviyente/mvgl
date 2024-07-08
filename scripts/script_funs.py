from itertools import product

import numpy as np
from sklearn.preprocessing import scale

import mvgl
from mvgl.thirdparty import jemgl

def run_svgl(X, densities):
    if not isinstance(densities, np.ndarray):
        densities = densities*np.ones(1)

    n_views = len(X)
    for i in range(n_views):
        X[i] = scale(X[i].T).T

    alpha = 0.1

    out = {}
    for density in densities:
        wv_hat = []
        run_times = []
        for v in range(n_views):
            wv, params, run_time = mvgl.learn_a_single_graph(X[v], alpha, density)
            wv_hat.append(wv)
            run_times.append(run_time)

            alpha = params["alpha"]

        out[(density, None)] = {
            "view": wv_hat, "consensus": None, "run time": np.mean(run_times), 
            "params": params
        }

    return out

def run_mvgl(X, consensus, reg_consensus, densities, similarities):
    if not isinstance(densities, np.ndarray):
        densities = densities*np.ones(1)

    if not isinstance(similarities, np.ndarray):
        similarities = similarities*np.ones(1)

    for i in range(len(X)):
        X[i] = scale(X[i].T).T

    alpha = 0.1
    beta = 0.1
    gamma = 0.1 if reg_consensus else None

    out = {}
    for density, similarity in product(densities, similarities):
        wv_hat, wc_hat, params, run_time = mvgl.learn_multiview_graph(
            X, alpha=alpha, beta=beta, consensus=consensus, gamma=gamma, 
            view_density=density, consensus_density=density, similarity=similarity,
        )
        out[(density, similarity)] = {
            "view": wv_hat, "consensus": wc_hat, "run time": run_time, 
            "params": params
        }

        alpha = params["alpha"]
        beta = params["beta"]
        gamma = params["gamma"]

    return out

def run_jemgl(X, model, rho_ns):
    if not isinstance(rho_ns, np.ndarray):
        rho_ns = rho_ns*np.ones(1)

    out = {}
    n_views = len(X)
    for rho_n in rho_ns:
        wv_hat, run_time = jemgl.run(X, model, rho_n, rho_1=1)

        # Thresholding
        all_weights = []
        for i in range(n_views):
            all_weights.extend(wv_hat[i][wv_hat[i] > 0])

        quantiles = np.linspace(0.0, 0.96, 49, endpoint=True)
        ths = np.quantile(all_weights, q=quantiles)
        for i, th in enumerate(ths):
            wv_hat_th = []
            for v in range(n_views):
                wv_hat_th.append(wv_hat[v].copy())
                wv_hat_th[-1][(wv_hat_th[-1] < th)] = 0

            out[(quantiles[i], rho_n)] = {
                "view": wv_hat_th, "consensus": None, "run time": run_time
            }

    return out