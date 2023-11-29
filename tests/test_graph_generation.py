import networkx as nx
import numpy as np
import pytest

from mvgl.data import graph
from mvgl import exceptions

def test_consensus_graph_connectedness():
    """Test if generated consensus graph is connected and if 
    MaxIterReachedException is raised if it cannot."""
    
    # check Erdos-Renyi model
    try:
        G = graph.gen_consensus_graph(n_nodes=10, graph_generator="er", p = 0.1)
        assert nx.is_connected(G), "Generated consensus graph is not connected."
    except exceptions.MaxIterReachedException:
        pass
    except:
        assert False, "Wrong exception raised."

    # check Barabasi-Albert model
    try:
        G = graph.gen_consensus_graph(n_nodes=10, graph_generator="ba", m = 1)
        assert nx.is_connected(G), "Generated consensus graph is not connected."
    except exceptions.MaxIterReachedException:
        pass
    except:
        assert False, "Wrong exception raised."

def test_gen_consensus_graph_reproduciblity():
    """Test if setting seed ensures reproducibility for generated consensus graph"""

    # check for Erdos-Renyi model
    rng = np.random.default_rng(seed=1)
    G1 = graph.gen_consensus_graph(n_nodes=10, graph_generator="er", p = 0.1,
                                   rng=rng)
    
    rng = np.random.default_rng(seed=1)
    G2 = graph.gen_consensus_graph(n_nodes=10, graph_generator="er", p = 0.1,
                                   rng=rng)
    
    assertion_msg = "Consensus graph generation is not reproducible when setting seed."
    assert nx.utils.graphs_equal(G1, G2), assertion_msg

    # check for Barabasi-Albert model
    rng = np.random.default_rng(seed=1)
    G1 = graph.gen_consensus_graph(n_nodes=10, graph_generator="ba", m = 2,
                                   rng=rng)

    rng = np.random.default_rng(seed=1)
    G2 = graph.gen_consensus_graph(n_nodes=10, graph_generator="ba", m = 2,
                                   rng=rng)

    assertion_msg = "Consensus graph generation is not reproducible when setting seed."
    assert nx.utils.graphs_equal(G1, G2), assertion_msg

def test_view_graphs_connected():
    Gc = graph.gen_consensus_graph(n_nodes=10, graph_generator="ba", m = 2)
    try:
        Gv = graph.gen_views(n_views=3, Gc=Gc, perturbation=0.1)
        for G in Gv:
            assert nx.is_connected(G), "Generated view graphs are not connected."
    except exceptions.MaxIterReachedException:
        pass
    except:
        assert False, "Wrong exception raised."

def test_gen_views_reproducibility():
    rng = np.random.default_rng(seed=1)
    Gc = graph.gen_consensus_graph(n_nodes=10, graph_generator="er", p = 0.1,
                                   rng=rng)
    Gv1 = graph.gen_views(n_views=3, Gc=Gc, perturbation=0.1, rng=rng)

    rng = np.random.default_rng(seed=1)
    Gc = graph.gen_consensus_graph(n_nodes=10, graph_generator="er", p = 0.1,
                                   rng=rng)
    Gv2 = graph.gen_views(n_views=3, Gc=Gc, perturbation=0.1, rng=rng)

    for i in range(len(Gv1)):
        assertion_msg = "Consensus graph generation is not reproducible when setting seed."
        assert nx.utils.graphs_equal(Gv1[i], Gv2[i]), assertion_msg