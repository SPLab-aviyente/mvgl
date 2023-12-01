import numpy as np
import networkx as nx
from scipy import linalg

# def gen_graph_signals(n_signals, Gv, fltr, noise, seed=None):
#     n_views = len(Gv)

#     if not isinstance(n_signals, list):
#         n_signals = [n_signals]*n_views

#     Xv = []
#     for s in range(n_views):
#         X = signal_gen.gen_smooth_gs(Gv[s], n_signals[s], filter=fltr, 
#                                      noise_amount=noise, 
#                                      seed=rng_seed(seed, 234*s))
#         Xv.append(X)

#     return Xv



def gen_smooth_gs(G, n_signals, filter="gaussian", alpha=10, noise_amount=0.1, 
                  rng=None):
    """Generate a set of smooth graph signals from a given graph.
    
    Smooth graph signal generation is based on [1] and is done by graph
    filtering a white noise signal. Let :math:`L = VDV^T` be the
    eigendecomposition of the graph Laplacian L. A smooth graph signal x is
    generated by:
    
    .. math:: x = Vh(D)y + e,
    where y is the graph Fourier transform of the white noise and e is the
    additive Gaussian noise. h(D) is the graph filter that makes x to be smooth
    over the graph. See [1] for different forms used for h(D).

    Parameters
    ----------
    G : nx.Graph
        An undirected graph.
    n_signals : int
        Number of signals to generate.
    filter : str, optional
        The filter to use to generate smooth signals. It can be 'gaussian',
        'tikhonov' or 'heat'. See [1] for further details, by default
        'gaussian' 
    alpha : float, optional
        A positive number used as the parameter for tikhonov filter and heat
        filter, by default 10. Ignored when the filter is gaussian.
    noise_amount : float, optional
        Amount of the noise to add the graph signals. Amount of the noise
        determined in L2-sense, that is if x is the clean signal |e|_2/|x|_2 =
        noise_amount, where e is the additive noise. By default 0.1.
    rng : np.random.Generator, optional
        Random number generator. If one wants the function to return the same 
        output every time, this needs to be set. By default None

    Returns
    -------
    X : np.array
        Generated smooth graph signals. Its dimension is (n_nodes, n_signals).
     
    References
    -------
    .. [1] Kalofolias, Vassilis. "How to learn a graph from smooth signals."
           Artificial Intelligence and Statistics. PMLR, 2016.
    """
    
    if rng is None: 
        rng = np.random.default_rng()

    n_nodes = G.number_of_nodes()

    # Generate white noise
    X0 = rng.multivariate_normal(np.zeros(n_nodes), np.eye(n_nodes), n_signals).T 

    # Get the graph Laplacian spectrum
    L = nx.laplacian_matrix(G).todense()
    e, V = linalg.eigh(L, overwrite_a=True)

    # Normalize the Laplacian such that |L|_2 = 1
    e[e < 1e-8] = 0
    e /= np.max(e)

    # Filtering to generate smooth graph signals from X0
    if filter == "gaussian":
        h = np.zeros(n_nodes)
        h[e > 0] = 1/np.sqrt(e[e>0])
    elif filter == "tikhonov":
        h = 1/(1+alpha*e)
    elif filter == "heat":
        h = np.exp(-alpha*e)

    X =V@np.diag(h)@V.T@X0

    # Add noise
    X_norm = np.linalg.norm(X)
    E = rng.normal(0, 1, X.shape)
    E_norm = np.linalg.norm(E)
    X += E*(noise_amount*X_norm/E_norm)

    return X