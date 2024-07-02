from pathlib import Path

import numpy as np

from mvgl.data import graph
from mvgl.data import signals
from mvgl.data.graph import gen_consensus_graph, gen_views
from mvgl.data.signals import gen_smooth_gs, gen_stationary_gs

def gen_simulations_save_path(n_nodes, n_views, n_signals, graph_generator, 
                              perturbation, noise, p = None, m = None,
                              base_dir = Path("data", "simulations")):
    """Generate a path for a simulation from its parameters.

    This function generates a path under `base_dir` that can be used to save 
    simulations related outputs such as generated graphs and signals, or outputs 
    of a method when they process simulated data. Note that this function 
    does not create the directory the generated path refers to.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    n_views : int
        Number of views.
    n_signals : int, or list of ints
        Number of signals for each view. 
    graph_generator : str
        The random graph model used when generating simulations. Must be 'er' or 'ba'. 
    perturbation : float
        Perturbation amount of edge swapping used when generating view graphs.
    noise : float
        Amount of the noise to added to the graph signals. 
    p : float, optional
        The edge probability for Erdos-Renyi model. 
    m : int, optional
        Growth parameter for Barabasi-Albert model. 
    base_dir : str or pathlib.Path, optional
        The base directory of the generated path. By default it is
        "data/simulations" under the current working directory.

    Returns
    -------
    save_path : pathlib.Path
        Generated path.

    Examples
    --------
    >>> gen_simulations_save_path(100, 5, 500, "er", 0.1, 0.1, p=0.1)
    Path("data", "simulations", "n_nodes_100", "n_views_5", 
         "n_signals_500", "er_0.10", "edge_swap_0.10", "noise_0.10")

    When `n_signals` is a list:
    
    >>> gen_simulations_save_path(100, 5, [500, 400], "er", 0.1, 0.1, p=0.1)
    Path("data", "simulations", "n_nodes_100", "n_views_5", 
         "n_signals_500_400", "er_0.10", "edge_swap_0.10", "noise_0.10")
    """

    # when number of signals differ across views
    if isinstance(n_signals, list):
        n_signals_str = "_".join(str(x) for x in n_signals)
    else:
        n_signals_str = f"{n_signals}"

    # random graph model name:
    if graph_generator == "er":
        graph_generator_str = f"{graph_generator}_{p:.2f}"
    elif graph_generator == "ba":
        graph_generator_str = f"{graph_generator}_{m:d}"

    return Path(base_dir,  
                f"n_nodes_{n_nodes}", 
                f"n_views_{n_views}", 
                f"n_signals_{n_signals_str}", 
                graph_generator_str, 
                f"edge_swap_{perturbation:.2f}", 
                f"noise_{noise:.2f}")

def gen_simulated_data(n_nodes, n_views, n_signals, graph_generator, perturbation, 
                       signal_type: str, noise, p = None, m = None, seed = None):
    """Generate simulated multiview graph signals.

    Data is generated first by drawing a consensus graph :math:`G` using
    Erdos-Renyi or Barabasi-Albert model. Generated graph then perturbed by
    :func:`mvgl.graph.swap_edges` to generate `n_views` graphs, that are
    considered as views of a multiview graph. Graph signals are sampled from
    each view graph independently using :func:`mvgl.signals.gen_smooth_gs` using
    a Gaussian graph filter.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    n_views : int
        Number of views.
    n_signals : int, or list of ints
        Number of signals to generate for each view. If `list`, its length
        should be `n_views` and for *i*th view, n_signals[i] graph signals will
        be generated.
    graph_generator : str
        The random graph model to use. Must be 'er' or 'ba'.
    perturbation : float
        Fraction of edges to swap when generating view graphs.
    noise : float
        Amount of the noise to add the graph signals. Amount of the noise
        determined in L2-sense, that is if x is the clean signal |e|_2/|x|_2 =
        `noise_amount`, where e is the additive noise. By default 0.1.
    p : float, optional
        The edge probability for Erdos-Renyi model. Must be provided when 
        `graph_generator` is 'er'. By default None.
    m : int, optional
        Growth parameter for Barabasi-Albert model. Must be provided when 
        `graph_generator` is 'ba'. By default None.
    seed : int, optional
        Seed number to use for random number generator. If one wants the
        function to return the same output every time, this needs to be set. By
        default None

    Returns
    -------
    G : nx.Graph
        Generated consensus graph.
    Gv : list of nx.Graph
        List of generated views graphs.
    Xv : list of np.array
        List of generated graph signals. Xv[i] is a (n_nodes, n_signals[i]) 
        dimensional matrix whose columns are graph signal defined on Gv[i]. 

    Raises
    ------
    MaxIterReachedException
        When generating consensus graph and view graphs, this function keep
        drawing them until they are connected. Maximum number of times it draws
        to produce these graphs is set to 500. If this reached, it will raise an
        error. 
    """
    rng = np.random.default_rng(seed)
    Gc = graph.gen_consensus_graph(n_nodes, graph_generator, p, m, rng)
    Gv = graph.gen_views(n_views, Gc, perturbation, rng)

    if not isinstance(n_signals, list):
        n_signals = [n_signals]*n_views

    Xv = []
    for s in range(n_views):
        if signal_type == "smooth":
            X = signals.gen_smooth_gs(Gv[s], n_signals[s], noise_amount=noise, 
                                      rng=rng)
        if signal_type == "stationary":
            X = signals.gen_stationary_gs(Gv[s], n_signals[s], noise_amount=noise, 
                                          rng=rng)
        Xv.append(X)

    return Gc, Gv, Xv