import numpy as np

from mvgl.data import signals, graph

def test_signal_reproducibility():
    """Test if same seed produces the same data."""

    rng = np.random.default_rng(seed=1)
    G = graph.gen_consensus_graph(n_nodes=100, graph_generator="er", p = 0.15,
                                   rng=rng)
    X1 = signals.gen_smooth_gs(G, n_signals=100, filter="gaussian", 
                               noise_amount=0.0, rng=rng)
    
    rng = np.random.default_rng(seed=1)
    G = graph.gen_consensus_graph(n_nodes=100, graph_generator="er", p = 0.15,
                                   rng=rng)
    X2 = signals.gen_smooth_gs(G, n_signals=100, filter="gaussian", 
                               noise_amount=0.0, rng=rng)
    
    msg = "Signal generation is not reproducible when setting seed."
    # Not testing equality, but ensuring small difference in case of machine precision
    assert np.sum(X1 - X2) < 1e-10, msg

def test_noise_amount():
    """Tests if the amount of noise in generate smooth signals."""
    rng = np.random.default_rng(seed=1)
    G = graph.gen_consensus_graph(n_nodes=100, graph_generator="er", p = 0.15,
                                   rng=rng)
    X1 = signals.gen_smooth_gs(G, n_signals=100, filter="gaussian", 
                               noise_amount=0.0, rng=rng)
    
    rng = np.random.default_rng(seed=1)
    noise_expected = 1.0
    G = graph.gen_consensus_graph(n_nodes=100, graph_generator="er", p = 0.15,
                                   rng=rng)
    X2 = signals.gen_smooth_gs(G, n_signals=100, filter="gaussian", 
                               noise_amount=noise_expected, rng=rng) 
    
    noise_actual = np.linalg.norm(X2-X1)/np.linalg.norm(X1)
    msg = "Noise amount in the data (in L2-sense) isn't correct."

    # Not testing equality, but ensuring small difference in case of machine precision
    assert noise_actual - noise_expected < 1e-10, msg