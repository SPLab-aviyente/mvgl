import warnings

import networkx as nx
import numpy as np
import cvxpy as cp

from scipy.linalg import sqrtm
from sklearn.metrics import f1_score as f1
from sklearn.metrics import average_precision_score as auprc

from mvgl.data import gen_consensus_graph

rng = np.random.default_rng(seed = 1)

n_nodes = 20
filter_degree = 5
n_signals = 1000 # non-positive value means we observe true covariance matrix

graph = gen_consensus_graph(n_nodes, "er", p = 0.1, rng=rng)
S0 = nx.adjacency_matrix(graph).toarray()
L0 = nx.laplacian_matrix(graph).toarray()

# Generate signals
filter_coeffs = rng.normal(0, 1, size=filter_degree)
filter_mat = np.zeros(shape=(n_nodes, n_nodes))

for i in range(filter_degree):
    filter_mat += filter_coeffs[i]*np.linalg.matrix_power(S0, i)

cov_mat = filter_mat@filter_mat 
cov_mat = (cov_mat + cov_mat.T)/2

if n_signals > 0:
    white_noise = rng.normal(0, 1, size=(n_nodes, n_signals))
    graph_signals = sqrtm(cov_mat)@white_noise
    cov_mat = (graph_signals@graph_signals.T)/n_signals

cov_mat = cov_mat/np.linalg.norm(cov_mat, "fro")

# Learn the graph
alpha = 1e-8
S = cp.Variable(shape=(n_nodes, n_nodes), symmetric=True)
objective = cp.Minimize(
    cp.sum(cp.abs(S))
)
while True:
    constraints = [
        cp.diag(S) <= 1e-6, 
        cp.sum(S, axis=1) >= 1, 
        cp.norm(cov_mat@S - S@cov_mat, "fro") <= alpha,
        S >= 0,
    ]
    problem = cp.Problem(objective, constraints)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = problem.solve()

    S_hat = S.value
    if S_hat is None:
        alpha = 2*alpha
    else:
        alpha_min = alpha/2
        alpha_max = alpha
        break

print(alpha)

while True:
    alpha = (alpha_min + alpha_max)/2
    constraints = [
        cp.diag(S) <= 1e-6, 
        cp.sum(S, axis=1) >= 1, 
        cp.norm(cov_mat@S - S@cov_mat, "fro") <= alpha,
        S >= 0,
    ]
    problem = cp.Problem(objective, constraints)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = problem.solve()

    S_hat = S.value
    if S_hat is None:
        alpha_min = alpha
    else:
        alpha_max = alpha
        if alpha_max/alpha_min <= 1.1:
            break

print(alpha)
    
# Performance measure
S0_upper = S0[np.triu_indices_from(S0, k=1)]
S_hat_upper = S_hat[np.triu_indices_from(S_hat, k=1)]
th = np.percentile(S_hat_upper, 75)

S_hat_upper_bin = S_hat_upper.copy()
S_hat_upper_bin[S_hat_upper_bin > th] = 1
S_hat_upper_bin[S_hat_upper_bin <= th] = 0

print(f"F1 Score: {f1(S0_upper, S_hat_upper_bin):.4f}")

print(f"AUPRC: {auprc(S0_upper, S_hat_upper):.4f}")

print(f"Density: {np.count_nonzero(S_hat_upper_bin)/len(S_hat_upper_bin):.2f}")