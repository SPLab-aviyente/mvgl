import networkx as nx

def ensure_connectedness(generator):
    """Draw a random graph until finding a connected one."""
    i = 1
    while True:
        G = generator()
        if nx.is_connected(G):
            return G
        else:
            i += 1

        if i>500:
            return None

def gen_consensus_graph(n_nodes, graph_generator, p = None, m = None):
    """Generate the consensus graph using Erdos-Renyi or Barabasi-Albert model.

    Parameters
    ----------
    n_nodes : int
        Number of nodes
    graph_generator : str
        The model to use. Must be 'er' or 'ba'.
    p : float, optional
        The edge probability for Erdos-Renyi model. Ignored if graph_generator 
        is 'ba'. By default None.
    m : int, optional
        Growth parameter for Barabasi-Albert model. Ignored if graph generator 
        is 'er'. By default None

    Returns
    -------
    G : nx.Graph
        Generated consensus graph

    Raises
    ------
    Exception
        This function keeps drawing a random graph until finding a connected 
        graph. Maximum number of draws is set to 500. If this reached it will 
        raise an error. 
    """
    if graph_generator == "er":
        graph_generator = lambda : nx.erdos_renyi_graph(n_nodes, p)
    elif graph_generator == "ba":
        graph_generator = lambda : nx.barabasi_albert_graph(n_nodes, m)

    # Generate the consensus graph, ensure that it is a connected graph
    G = ensure_connectedness(graph_generator)
    if G is None:
        raise Exception("I cannot create a connected consensus graph with "   
                        "given model parameter.")

    return G