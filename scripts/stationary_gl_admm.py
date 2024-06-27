import networkx as nx
import numpy as np

from scipy import sparse
from scipy.linalg import sqrtm

from mvgl import data
from mvgl.graphlearning.utils import rowsum_mat

##### Functions #####

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

def _s_step():
    pass 

def _z_step():
    pass

##### Generate the graph #####

n_nodes = 20
n_pairs = n_nodes*(n_nodes-1)//2

graph = data.gen_consensus_graph(n_nodes, "er", p=0.2, rng=1)

gt_adj_mat = nx.adjacency_matrix(graph).toarray()
gt_adj_upper = gt_adj_mat[np.triu_indices_from(gt_adj_mat, k=1)]

##### Generate the signals #####

filter_degree = 5
n_signals = -1 # Observe infinite amount of signals
rng = np.random.default_rng(seed=2)

filter_coeffs = rng.normal(0, 1, size=filter_degree)
filter_mat = np.zeros(shape=(n_nodes, n_nodes))

for i in range(filter_degree):
    filter_mat += filter_coeffs[i]*np.linalg.matrix_power(gt_adj_mat, i)

cov_mat = filter_mat@filter_mat
cov_mat = (cov_mat + cov_mat.T)/2

if n_signals > 0:
    white_noise = rng.normal(0, 1, size=(n_nodes, n_signals))
    graph_signals = sqrtm(cov_mat)@white_noise
    cov_mat = np.cov(graph_signals) #NOTE

##### Learn the adjacancy vector #####

# Data matrices
m_sigma = _sigma_mat(cov_mat)
m_q = _upper_to_all_mat(n_nodes)
m_r = _diag_to_all_mat(n_nodes)
m_s = rowsum_mat(n_nodes)

m_k = m_sigma@(m_q - m_r@m_s)

# Optimization parameters
param_rho = 1
param_k = 30 

# Initialization
v_z = np.zeros(shape=(n_pairs, 1))
v_y = np.zeros(shape=(n_pairs, 1))

for i in range(500):
    # l-step
    

    

