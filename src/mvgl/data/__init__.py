import numpy as np

from mvgl.data import graph

def gen_simulated_data(n_nodes, n_views, n_signals, graph_generator, p, m, 
                       perturbation, noise, seed=None):
    rng = np.random.default_rng(seed)
    Gc = graph.gen_consensus_graph(n_nodes, graph_generator, p, m, rng)
    # Gv = graph.gen_views(n_views, Gc, perturbation)

    # wc_gt, wv_gt = data.vectorize_graphs(Gc, Gv)

    # X = data.gen_data(n_signals, Gv, data_filter, data_noise, r+1)
    return Gc