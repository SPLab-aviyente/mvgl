from pathlib import Path

import networkx as nx
import numpy as np

from mvgl import data

def test_simulation_save_path_generation():
    """Tests if path generated to save simulation results are correct."""
    expected = Path("data", "simulations", "n_nodes_100", "n_views_5", 
                    "n_signals_500", "er_0.10", "edge_swap_0.10", "noise_0.10")
    actual = data.gen_simulations_save_path(100, 5, 500, "er", 0.1, 0.1, p=0.1)

    assert expected == actual, "Generated simulated save path isn't correct."

    expected = Path("data", "simulations", "n_nodes_100", "n_views_5", 
                    "n_signals_500_400", "er_0.10", "edge_swap_0.10", "noise_0.10")
    actual = data.gen_simulations_save_path(100, 5, [500, 400], "er", 0.1, 0.1, p=0.1)

    msg = "Generated simulated save path is wrong, when n_signals is list."
    assert expected == actual, msg

def test_simulated_data_generation_reproducible():
    G1, Gv1, Xv1 = data.gen_simulated_data(100, 5, 500, "er", 0.1, 0.1, p=0.1, seed=1)
    G2, Gv2, Xv2 = data.gen_simulated_data(100, 5, 500, "er", 0.1, 0.1, p=0.1, seed=1)

    assertion_msg = "Consensus graph is not the same when setting seed."
    assert nx.utils.graphs_equal(G1, G2), assertion_msg

    for i in range(len(Gv1)):
        assertion_msg = "View graphs are not the same when setting seed."
        assert nx.utils.graphs_equal(Gv1[i], Gv2[i]), assertion_msg

        msg = "Signals are not the same when setting seed."
        # Not testing equality, but ensuring small difference in case of machine precision
        assert np.sum(Xv1[i] - Xv2[i]) < 1e-10, msg