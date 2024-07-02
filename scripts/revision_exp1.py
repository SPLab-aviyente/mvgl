from pathlib import Path

import click
import numpy as np
import networkx as nx
import pandas as pd

import mvgl
import mvgl.data
from mvgl.graphlearning import utils, metrics

from script_funs import run_mvgl

@click.command()
@click.option("--method", default="mvgl-l1", show_default=True)
@click.option("--n-edges-to-add", default=50, show_default=True)
@click.option("--n-edges-to-remove", default=50, show_default=True)
@click.option("--n-views-to-change", default=4, show_default=True)
@click.option("--seed", default=None, type=int, show_default=True)
def main(
    method, n_edges_to_add, n_edges_to_remove, n_views_to_change, seed
):
    n_views = 6
    n_nodes = 100
    rng = np.random.default_rng(seed)
    
    ###############################
    ## SIMULATED DATA GENERATION ##
    ###############################

    # Generate the consensus graph
    Gc = mvgl.data.gen_consensus_graph(n_nodes, graph_generator="ba", m=5, rng=rng)
    wc_gt = utils.vectorize_a_graph(Gc)

    # Generate the views
    connected_pairs = np.where(wc_gt == 1)[0]
    unconnected_pairs = np.where(wc_gt == 0)[0]
    edges_to_remove = rng.choice(connected_pairs, n_edges_to_remove, replace=False)
    edges_to_add = rng.choice(unconnected_pairs, n_edges_to_add, replace=False)

    wv_gt = np.repeat(wc_gt[..., None], 6, axis=1)
    for e in edges_to_remove:
        updates = rng.choice(n_views, n_views_to_change, replace=False)
        wv_gt[e, updates] = 0

    for e in edges_to_add:
        updates = rng.choice(n_views, n_views_to_change, replace=False)
        wv_gt[e, updates] = 1

    wv_gt = [wv_gt[:, v].squeeze() for v in range(n_views)]

    # Generate the signals
    Gv = []
    Xv = []
    for v in range(n_views):
        Wv = np.zeros(shape=(100, 100))
        Wv[np.triu_indices_from(Wv, k=1)] = wv_gt[v]
        Wv += Wv.T

        Gv.append(nx.from_numpy_array(Wv))
        Xv.append(mvgl.data.gen_smooth_gs(Gv[-1], 500))

    ####################
    ## GRAPH LEARNING ##
    ####################

    if method == "mvgl-l1":
        densities = np.linspace(0.06, 0.2, 8, endpoint=True)
        similarities = np.linspace(0.7, 0.9, 5, endpoint=True)
        out = run_mvgl(Xv, "l1", False, densities, 0.8)
    elif method == "mvgl-l2": 
        densities = np.linspace(0.06, 0.2, 8, endpoint=True)
        out = run_mvgl(Xv, "l2", True, densities, 0.8)

    ############################# 
    ## PERFORMANCE CALCULATION ##
    #############################

    # Construct the output dataframe
    performances = {
        "Run": [], "Method": [], "F1": [], "Graph": [], "Density": [], 
        "Similarity": [], "LearnedDensity": [], "LearnedSimilarity": [], 
        "RunTime": []
    }
    for case, curr_out in out.items():
        density, similarity = case

        for graph in ["view", "consensus"]:
            w_gt = wc_gt if graph == "consensus" else wv_gt
            case_similarity = None if graph == "consensus" else similarity
            learned_similarity = None if graph == "consensus" else \
                metrics.correlation(curr_out[graph], curr_out[graph])

            performances["Run"].append(seed)
            performances["Method"].append(method)
            performances["F1"].append(np.mean(metrics.f1(w_gt, curr_out[graph])))
            performances["Graph"].append(graph)
            performances["Density"].append(density)
            performances["Similarity"].append(case_similarity)
            performances["LearnedDensity"].append(np.mean(metrics.density(curr_out[graph])))
            performances["LearnedSimilarity"].append(learned_similarity)
    
    if len(performances["Run"]) > 0:
        performances = pd.DataFrame.from_dict(performances)

        save_path = Path(
            mvgl.ROOT_DIR, "data", "simulations", "outputs", 
            f"revision-exp1-{n_edges_to_add}-{n_edges_to_remove}-{n_views_to_change}"
        )
        save_path.mkdir(parents=True, exist_ok=True)

        save_file = Path(save_path, f"{method}-run-{seed}")

        performances.to_csv(save_file)

if __name__ == "__main__":
    main()