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
graph = data.gen_consensus_graph(n_nodes, "er", p=0.2, rng=1)
signals = data.gen_stationary_gs(graph, 1000, filter_degree=5, rng=2)

laplacian = nx.laplacian_matrix(graph)
sample_cov = np.cov(signals)

# Data matrices 
m_D = _sigma_mat(sample_cov)
m_Q = _upper_to_all_mat(n_nodes)
m_R = _diag_to_all_mat(n_nodes)
m_S = rowsum_mat(n_nodes)
v_s = m_S[[0], :]

# Optimization
l = cp.Variable((n_pairs, 1))

alpha = 1
beta = 1e-6

obj_l2sq = cp.Minimize(
    cp.sum_squares(m_D@(m_Q-m_R@m_S)@l) + 
    2*alpha*cp.sum_squares(l) + 
    alpha*cp.sum_squares(m_S@l)
)

obj_l1 = cp.Minimize(
    cp.sum_squares(m_D@(m_Q-m_R@m_S)@l) + 
    alpha*cp.norm(l, 1) 
)

obj_l2sq_barrier = cp.Minimize(
    cp.sum_squares(m_D@(m_Q-m_R@m_S)@l) + 
    2*alpha*cp.sum_squares(l) -
    beta*cp.sum(cp.log(-m_S@l))
)

obj_l1_barrier = cp.Minimize(
    cp.sum_squares(m_D@(m_Q-m_R@m_S)@l) + 
    2*alpha*cp.norm(l, 1) -
    beta*cp.sum(cp.log(-m_S@l))
)

obj_l1_barrier_nof = cp.Minimize(
    alpha*cp.norm(l, 1) -
    beta*cp.sum(cp.log(-m_S@l))
)

obj_l1_nof = cp.Minimize(
    cp.pnorm(l, 1)
)

st_constraint = [cp.pnorm(m_D@(m_Q-m_R@m_S)@l, 2) <= 4e-4]
neg_constraint = [l <= 0]
trace_constraint = [cp.sum(l) == -n_nodes]
first_node_constraint = [v_s@l == -1]
bounded_degree_constraint = [m_S@l <= -1]

prob = cp.Problem(
    obj_l1_nof, 
    neg_constraint + bounded_degree_constraint + st_constraint
)

result = prob.solve()

A = nx.adjacency_matrix(graph).toarray()
a_gt = A[np.triu_indices_from(A, k=1)]

a_hat = np.squeeze(np.abs(l.value))
# a_hat[np.argpartition(a_hat, -50)[:-50]] = 0
# a_hat[a_hat<1e-4] = 0

# a_hat_binary = a_hat.copy()
# a_hat_binary[a_hat>0] = 1

print(a_hat)

print(f"AUPRC: {auprc(a_gt, a_hat):.4f}")
