from itertools import product

import numpy as np
from sklearn.preprocessing import scale

import mvgl

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