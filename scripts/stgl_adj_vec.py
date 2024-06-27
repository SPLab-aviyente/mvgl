import warnings

import networkx as nx
import numpy as np
import cvxpy as cp

from scipy import sparse
from scipy.linalg import sqrtm
from sklearn.metrics import f1_score as f1
from sklearn.metrics import average_precision_score as auprc

from mvgl.data import gen_consensus_graph
from mvgl.graphlearning.utils import rowsum_mat

def _sigma_mat(covariance_mat):
    n_nodes = covariance_mat.shape[0] # number of nodes
    identity_mat = sparse.eye(n_nodes)

    return (sparse.kron(-covariance_mat, identity_mat) 
            + sparse.kron(identity_mat, covariance_mat))

def _upper_to_all_mat(n):
    i, j = np.triu_indices(n, k=1)
    M = len(i)
    rows = np.concatenate((i*n+j, j*n+i))
    cols = np.concatenate((np.arange(M), np.arange(M)))

    return sparse.csr_matrix((np.ones((2*M, )), (rows, cols)), shape=(n**2, M))

def _diag_to_all_mat(n):
    i, j = np.diag_indices(n)
    rows = i*n + j
    cols = np.arange(n)

    return sparse.csr_matrix((np.ones((n, )), (rows, cols)), shape=(n**2, n))

rng = np.random.default_rng(seed = 1)

n_nodes = 20
n_pairs = int(n_nodes*(n_nodes-1)/2)
filter_degree = 5
n_signals = 10000 # non-positive value means we observe true covariance matrix

graph = gen_consensus_graph(n_nodes, "er", p = 0.1, rng=rng)
S0 = nx.adjacency_matrix(graph).toarray()

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

# Data matrices
m_D = _sigma_mat(cov_mat)
m_Q = _upper_to_all_mat(n_nodes)
m_R = _diag_to_all_mat(n_nodes)
m_S = rowsum_mat(n_nodes)

# Learn the graph
alpha = 1e-8
s = cp.Variable(shape=(n_pairs, 1), nonneg=True)
objective = cp.Minimize(cp.sum(cp.abs(s)))
while True:
    constraints = [ 
        m_S@s >= 1, 
        cp.norm(m_D@m_Q@s, 2) <= alpha
    ]
    problem = cp.Problem(objective, constraints)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = problem.solve(max_iter=1000)

    S_hat_upper = s.value
    if S_hat_upper is None:
        alpha = 2*alpha
    else:
        alpha_min = alpha/2
        alpha_max = alpha
        break

print(alpha)

while True:
    alpha = (alpha_min + alpha_max)/2
    constraints = [ 
        m_S@s >= 1, 
        cp.norm(m_D@m_Q@s, 2) <= alpha
    ]
    problem = cp.Problem(objective, constraints)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = problem.solve(max_iter=1000)

    S_hat_upper = s.value
    if S_hat_upper is None:
        alpha_min = alpha
    else:
        alpha_max = alpha
        if alpha_max/alpha_min <= 1.1:
            break

print(alpha)

S_hat_upper = np.squeeze(S_hat_upper)

# Performance measure
S0_upper = S0[np.triu_indices_from(S0, k=1)]
# S_hat_upper[S_hat_upper <= 1e-4] = 0
# th = np.percentile(S_hat_upper, 75)

S_hat_upper_bin = S_hat_upper.copy()
# S_hat_upper_bin[S_hat_upper_bin > 0] = 1
# S_hat_upper_bin[S_hat_upper_bin > th] = 1
# S_hat_upper_bin[S_hat_upper_bin <= th] = 0


# print(f"F1 Score: {f1(S0_upper, S_hat_upper_bin):.4f}")

print(f"AUPRC: {auprc(S0_upper, S_hat_upper):.4f}")

# print(f"Density: {np.count_nonzero(S_hat_upper_bin)/len(S_hat_upper_bin):.2f}")