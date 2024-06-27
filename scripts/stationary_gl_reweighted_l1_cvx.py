import numpy as np
import networkx as nx
import cvxpy as cp

from scipy import sparse
from sklearn.metrics import f1_score as f1
from sklearn.metrics import average_precision_score as auprc

from mvgl import data
from mvgl.graphlearning.utils import rowsum_mat

np.set_printoptions(precision=4, suppress=True)

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

# Generate stationary graph signals
n_nodes = 20
n_pairs = n_nodes*(n_nodes-1)//2
# graph = data.gen_consensus_graph(n_nodes, "ba", m=1, rng=1)
graph = data.gen_consensus_graph(n_nodes, "er", p=0.1, rng=1)
signals = data.gen_stationary_gs(graph, 1000, rng=2)

laplacian = nx.laplacian_matrix(graph)
sample_cov = np.cov(signals)

# Data matrices 
m_D = _sigma_mat(sample_cov)
m_Q = _upper_to_all_mat(n_nodes)
m_R = _diag_to_all_mat(n_nodes)
m_S = rowsum_mat(n_nodes)
v_s = m_S[[0], :]

# l_upper = laplacian.toarray()[np.triu_indices(n_nodes, k=1)][..., None]

# obj_mat = np.linalg.norm(laplacian@sample_cov - sample_cov@laplacian)**2
# obj_vec = np.linalg.norm(m_D@(m_Q-m_R@m_S)@l_upper)**2

# print(obj_mat)

l_prev = -n_nodes*np.ones((n_pairs, 1))/n_pairs
tau = 1
delta = .01

for iter in range(10):
    weights = tau/(np.abs(l_prev) + delta)

    l = cp.Variable((n_pairs, 1))

    # Solve the problem: 
    objective = cp.Minimize(
        cp.sum_squares(m_D@(m_Q-m_R@m_S)@l) + weights.T@l
    )
    constraints = [l <= 0, cp.sum(l) == -n_nodes]

    prob = cp.Problem(objective, constraints)

    result = prob.solve()

    l_prev = l.value
    # l_prev[l_prev > -1e-4] = 0

    print(f"Density: {np.count_nonzero(l_prev)/n_pairs:.2f}")

##### PERFORMANCE MEASURES #####

A = nx.adjacency_matrix(graph).toarray()
a_gt = A[np.triu_indices_from(A, k=1)][..., None]

a_hat = np.abs(l.value)
a_hat[a_hat<1e-4] = 0

# k = int(n_pairs*0.85)
# a_hat[np.argpartition(a_hat, k, axis=0)[:k]] = 0

a_hat_binary = a_hat.copy()
a_hat_binary[a_hat>0] = 1

print(f"F1 Score: {f1(a_gt, a_hat_binary):.4f}")

print(f"AUPRC: {auprc(a_gt, a_hat):.4f}")

print(f"Density: {np.count_nonzero(a_hat)/n_pairs:.2f}")
