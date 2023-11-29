from mvgl.data import graph

def gen_simulated_data(n_nodes, n_views, n_signals, graph_generator, p, m, 
                       perturbation, noise):
    Gc = graph.gen_consensus_graph(n_nodes, graph_generator, p, m)
    
    # if "weight" in consensus_gen:
    #     data.assign_edge_weights(Gc, consensus_gen, n_views)

    # Gv = data.gen_views(n_views, Gc, consensus_gen, view_gen, r+1)

    # wc_gt, wv_gt = data.vectorize_graphs(Gc, Gv)

    # X = data.gen_data(n_signals, Gv, data_filter, data_noise, r+1)
    return Gc