import numpy as np
import networkx as nx

from numba import njit
from scipy import sparse
from sklearn.metrics import f1_score as f1
from sklearn.metrics import average_precision_score as auprc

from mvgl import data
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

def _prox_l1(x, alpha):
    return np.sign(x)*np.maximum(np.abs(x) - alpha, 0)

def _constrained_sum_prox_l1(v_y, param_gamma, param_b):
    # Source: https://math.stackexchange.com/a/2886715

    n_dims = v_y.shape[0]

    # Edge cases before bisection based solution
    y_min = np.min(v_y)
    y_max = np.max(v_y)
    g_at_beta_max = np.sum(v_y - y_max) - param_b
    g_at_beta_min = np.sum(v_y - y_min) - param_b

    if g_at_beta_max > 0:
        beta = y_max + param_gamma + g_at_beta_max/n_dims
        return _prox_l1(v_y - beta, param_gamma)
    
    if g_at_beta_min < 0:
        beta = y_min - param_gamma - g_at_beta_min/n_dims
        return _prox_l1(v_y - beta, param_gamma)

    beta_candidates = np.squeeze(np.sort(
        np.concatenate([v_y - param_gamma, v_y + param_gamma])
    ))
    lower_idx = 0  # Bisection Lower Bound
    upper_idx = 2 * n_dims - 1  # Bisection Upper Bound
    max_iter = int(np.ceil(np.log2(upper_idx))) + 1

    # Bisection method
    for _ in range(max_iter):
        
        if (upper_idx - lower_idx) <= 1:
            break

        curr_indx = np.round(lower_idx + upper_idx) // 2
        curr_indx = min(max(curr_indx, lower_idx + 1), upper_idx - 1)

        beta = beta_candidates[curr_indx]
        v_x = _prox_l1(v_y - beta, param_gamma)

        if np.sum(v_x) > param_b:
            lower_idx = curr_indx
        else:
            upper_idx = curr_indx

    beta = (beta_candidates[lower_idx] + beta_candidates[upper_idx]) / 2
    v_x = _prox_l1(v_y - beta, param_gamma)
    v_s = v_x != 0
    beta = (np.sum(v_y[v_s] - param_gamma * np.sign(v_x[v_s])) - param_b) / np.sum(v_s)
    return  _prox_l1(v_y - beta, param_gamma)

@njit
def _sparse_proj_on_hyperplane(v_y, param_k, param_b):
    support = np.zeros(v_y.shape[0], dtype="bool")
    support[np.argmin(-v_y)] = 1
    support_size = 1
    while support_size < param_k:
        support_size += 1
        curr_sum = np.sum(v_y[support])
        curr_y = np.abs(v_y - (curr_sum + param_b)/(support_size - 1))
        curr_y[support] = -np.infty
        j = np.argmax(curr_y)
        support[j] = 1

    result = v_y.copy()
    result[~support] = 0
    tau = (np.sum(result) + param_b)/param_k
    result[support] = result[support] - tau
    return result

# def _l_step(m_K, v_z, v_y, alpha, rho, t):
#     n_pairs = v_z.shape[0]
#     n_nodes = int((1 + np.sqrt(8*n_pairs + 1))/2) 

#     delta_f = lambda v_x: 2*m_K.T@(m_K@v_x) + rho*(v_x - v_z) + v_y

#     v_l = -n_nodes*np.ones(shape=(n_pairs, 1))/n_pairs

#     for _ in range(100):
#         v_w = v_l - t*delta_f(v_l)
#         prev_v_l = v_l
#         v_l = _constrained_sum_prox_l1(v_w, t*alpha, -n_nodes)

#     return v_l

def _l_step(m_K, v_z, v_y, alpha, rho, t):
    rng = np.random.default_rng()

    n_pairs = v_z.shape[0]
    n_nodes = int((1 + np.sqrt(8*n_pairs + 1))/2) 

    delta_f = lambda v_x: 2*m_K.T@(m_K@v_x) + 2*rho*(v_x - v_z) + v_y

    v_l = rng.normal(0, 1, size=(n_pairs, 1))
    # v_l[rng.choice(n_nodes, size=alpha, replace=False)]
    # v_z.copy() # 
    # v_l = v_z # -n_nodes*np.ones(shape=(n_pairs, 1))/n_pairs

    for i in range(1000):
        v_w = v_l - t*delta_f(v_l)
        prev_v_l = v_l
        v_l = _sparse_proj_on_hyperplane(v_w, alpha, n_nodes)

        #if (i+1) % 5 == 0:
        #    print(f"{np.linalg.norm(v_l - prev_v_l):.6f}")

    return v_l

def _z_step(v_l, v_y, rho):
    v_z = v_l + v_y/rho
    return np.minimum(v_z, 0)

def _objective(m_K, v_l):
    return np.linalg.norm(m_K@v_l)**2

# Generate stationary graph signals
n_nodes = 20
n_pairs = n_nodes*(n_nodes-1)//2
graph = data.gen_consensus_graph(n_nodes, "er", p=0.1, rng=1)
signals = data.gen_stationary_gs(graph, 100, rng=2, filter_degree=2)
# signals = data.gen_smooth_gs(graph, 10000, rng=2)

laplacian = nx.laplacian_matrix(graph)
sample_cov = np.cov(signals)

# Data matrices 
m_D = _sigma_mat(sample_cov)
m_Q = _upper_to_all_mat(n_nodes)
m_R = _diag_to_all_mat(n_nodes)
m_S = rowsum_mat(n_nodes)
v_s = m_S[[0], :]

m_K = m_D@(m_Q - m_R@m_S)

rho = .1
alpha = 30
v_z = np.zeros(shape=(n_pairs, 1))
v_y = np.zeros(shape=(n_pairs, 1))

for i in range(500):
    v_l = _l_step(m_K, v_z, v_y, alpha, rho, .01)
    v_z = _z_step(v_l, v_y, rho)
    v_y += rho*(v_l - v_z)

    # if i % 5 == 0:
    #     print(f"{np.linalg.norm(v_l - v_z):.4f} | {_objective(m_K, v_z):.4f}")

print("Done!")

##### PERFORMANCE MEASURES #####

A = nx.adjacency_matrix(graph).toarray()
a_gt = A[np.triu_indices_from(A, k=1)][..., None]

a_hat = np.abs(v_l)
a_hat[a_hat<1e-4] = 0

a_hat_binary = a_hat.copy()
a_hat_binary[a_hat>0] = 1

print(a_hat)

print(f"F1 Score: {f1(a_gt, a_hat_binary):.4f}")

print(f"AUPRC: {auprc(a_gt, a_hat):.4f}")

print(f"Density: {np.count_nonzero(a_hat)/n_pairs:.2f}")