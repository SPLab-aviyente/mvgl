import time

import numpy as np
import networkx as nx

from numpy.linalg import eigh, norm, pinv, inv

def _omicron_step(m_Q, m_B, m_E, m_P, rho):
    mat_to_proj = (m_P.T@(m_Q + m_E - rho*m_B)@m_P)/rho
    v_e, m_V = eigh(mat_to_proj)
    v_d = (-rho*v_e + np.sqrt((rho**2)*(v_e**2) + 4*rho))/(2*rho)
    return m_V@np.diag(v_d)@m_V.T

def _a_step(m_B_vec, m_F_vec, m_J, m_J_inv, rho_n, rho_1, rho):
    mat_to_proj = m_B_vec@m_J.T - m_F_vec
    multiplier = (rho_n*rho_1)/(rho*norm(mat_to_proj, ord=2, axis=1, keepdims=True) + 1e-16)
    m_A_vec = np.maximum(1 - multiplier, 0)*mat_to_proj
    return m_A_vec@m_J_inv.T

def _b_step(m_L_vec, m_A_vec, m_E_vec, m_F_vec, m_J, m_J_tilde, m_J_tilde_inv, 
            diags, off_diags, rho):
    
    m_B_vec = np.zeros_like(m_E_vec)
    m_B_vec[diags, :] = np.maximum(m_E_vec[diags, :]/rho + m_L_vec[diags, :], 0)
    m_B_vec[off_diags, :] = np.minimum((
        m_E_vec[off_diags, :]/rho + m_F_vec[off_diags, :]@m_J/rho +
        m_L_vec[off_diags, :] + m_A_vec[off_diags, :]@m_J_tilde
    )@m_J_tilde_inv, 0)

    return m_B_vec

def _run_optimization(m_Qv, m_J, m_P, rho_n, rho_1, rho=1, max_iter=1000):
    n_views = len(m_Qv)
    n_nodes = m_Qv[0].shape[0]

    diags = np.ravel_multi_index(np.diag_indices(n_nodes), (n_nodes, n_nodes))
    off_diags = np.ravel_multi_index(np.where(np.eye(n_nodes) == 0), (n_nodes, n_nodes))

    m_J_inv = pinv(m_J)
    m_J_tilde = m_J.T@m_J
    m_J_tilde_inv = inv(np.eye(n_views) + m_J_tilde)

    # ADMM initialization
    m_tmp = 2*np.eye(n_nodes) - np.ones((n_nodes, n_nodes))
    m_B_vec = np.concatenate([m_tmp.reshape(-1, 1) for v in range(n_views)], axis=1)
    m_E_vec = np.zeros((n_nodes**2, n_views))
    m_F_vec = np.zeros((n_nodes**2, m_J.shape[0]))
    m_L_vec = np.zeros((n_nodes**2, n_views))

    # ADMM steps
    for i in range(max_iter):

        # Omicron step
        for v in range(n_views):
            m_B = np.reshape(np.squeeze(m_B_vec[:, v]), (n_nodes, n_nodes))
            m_E = np.reshape(np.squeeze(m_E_vec[:, v]), (n_nodes, n_nodes))
            m_Omicron = _omicron_step(m_Qv[v], m_B, m_E, m_P, rho)

            m_L_vec[:, v] = (m_P@m_Omicron@m_P.T).reshape(-1, 1).squeeze()

        m_A_vec = _a_step(m_B_vec, m_F_vec, m_J, m_J_inv, rho_n, rho_1, rho)
        
        m_B_vec = _b_step(m_L_vec, m_A_vec, m_E_vec, m_F_vec, m_J, m_J_tilde, 
                          m_J_tilde_inv, diags, off_diags, rho)
        
        # Update Lagrangian multipliers
        m_E_vec = m_E_vec + rho*(m_L_vec - m_B_vec)
        m_F_vec = m_F_vec + rho*(m_A_vec - m_B_vec)@m_J.T   

        # Check convergence
        primal_residual_BL = np.mean(np.abs(m_B_vec - m_L_vec))
        if primal_residual_BL < 1e-4:
            break

    # Return the adjacency matrix
    adj = []
    for v in range(n_views):
        m_L = np.reshape(np.squeeze(m_L_vec[:, v]), (n_nodes, n_nodes))
        adj.append(np.abs(m_L[np.triu_indices_from(m_L, k=1)]))
        adj[-1][adj[-1] < 1e-4] = 0

    return adj

def run(Xv: list[np.array], model: str, rho_n: float, rho_1: float, 
        rho: float = 1, max_iter: int = 1000):
    
    # Input check
    if not isinstance(Xv, list):
        raise Exception("Mutliple sets of graph signals must be provided " 
                        "when learning multiple graphs.")  

    if model not in ["group", "laplacian"]:
        raise Exception("Parameter `model` must be one of ['group', 'laplacian']")
    
    n_views = len(Xv)
    n_nodes = Xv[0].shape[0]
    tot_signals = np.sum([X.shape[1] for X in Xv])

    # construct the data matrices
    m_Qv = []
    for v in range(n_views):
        n_signals_view = Xv[v].shape[1]
        m_Sigma = np.cov(Xv[v] - np.mean(Xv[v], axis=1, keepdims=True))
        m_H = ((tot_signals*rho_n)/n_signals_view)*(
            np.eye(n_nodes) - np.ones((n_nodes, n_nodes))
        )
        m_Qv.append(m_Sigma + m_H)

    # select J matrix
    if model == "group":
        m_J = np.eye(n_views)
    elif model == "laplacian":
        m_J = nx.incidence_matrix(nx.complete_graph(n_views), oriented=True)
        m_J = m_J.toarray().T

    # construct P matrix
    v = np.ones(shape=(n_nodes, 1))
    w = np.zeros(shape=(n_nodes, 1))
    w[0] = np.sqrt(n_nodes)
    w = v + w
    m_P = (np.eye(n_nodes) - 2*(w@w.T)/(w.T@w))[:, 1:]

    st = time.time()
    wv = _run_optimization(m_Qv, m_J, m_P, rho_n, rho_1, rho, max_iter)
    run_time = time.time() - st 

    return wv, run_time
    