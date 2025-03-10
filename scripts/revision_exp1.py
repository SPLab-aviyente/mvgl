r"""Script for edge addition-removal based experiments

In this script methods are applied to a datasets where view graphs are generated
from the consensus graph by adding and/or removing edges. Namely, the script
first generates a consensus graph :math:`G` with 100 nodes using either ER graph
model (edge probability set 0.1) or BA graph model (growth parameter is set 5).
6 view graphs are then generated from :math:`G` as follows: 

- Initiate each view graph the same as :math:`G`.
- Randomly select `n-edges-to-add` number of unconnected pairs from :math:`G`.
Each selected pair is connected in randomly chosen `n-views-to-change` number
of views. 
- Randomly select `n-edges-to-remove` number of edges from :math:`G`. Each 
selected edge is removed from randomly chosen `n-views-to-change` number of
views.

"""

from pathlib import Path

import click
import numpy as np
import networkx as nx
import pandas as pd

import mvgl
import mvgl.data
from mvgl.graphlearning import utils, metrics

from script_funs import run_mvgl, run_svgl

@click.command()
@click.option("--method", default="svgl", show_default=True)
@click.option("--graph-model", default="er", show_default=True)
@click.option("--n-edges-to-add", default=50, show_default=True)
@click.option("--n-edges-to-remove", default=50, show_default=True)
@click.option("--n-views-to-change", default=4, show_default=True)
@click.option("--seed", default=None, type=int, show_default=True)
def main(
    method, graph_model, n_edges_to_add, n_edges_to_remove, n_views_to_change, seed
):
    n_views = 6
    n_nodes = 100
    n_signals = 500
    rng = np.random.default_rng(seed)
    
    ###############################
    ## SIMULATED DATA GENERATION ##
    ###############################

    # Generate the consensus graph
    Gc = mvgl.data.gen_consensus_graph(n_nodes, graph_generator=graph_model, p=0.1, 
                                       m=5, rng=rng)
    wc_gt = utils.vectorize_a_graph(Gc)

    # Generate the views
    connected_pairs = np.where(wc_gt == 1)[0]
    unconnected_pairs = np.where(wc_gt == 0)[0]
    edges_to_remove = rng.choice(connected_pairs, n_edges_to_remove, replace=False)
    edges_to_add = rng.choice(unconnected_pairs, n_edges_to_add, replace=False)

    wv_gt = np.repeat(wc_gt[..., None], n_views, axis=1)
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
        Wv = np.zeros(shape=(n_nodes, n_nodes))
        Wv[np.triu_indices_from(Wv, k=1)] = wv_gt[v]
        Wv += Wv.T

        Gv.append(nx.from_numpy_array(Wv))
        Xv.append(mvgl.data.gen_smooth_gs(Gv[-1], n_signals))

    ####################
    ## GRAPH LEARNING ##
    ####################

    if method == "mvgl-l1":
        densities = np.linspace(0.06, 0.20, 8, endpoint=True)
        similarities = np.linspace(0.7, 0.95, 6, endpoint=True)
        out = run_mvgl(Xv, "l1", False, densities, similarities)
    elif method == "mvgl-l2": 
        densities = np.linspace(0.06, 0.20, 8, endpoint=True)
        similarities = np.linspace(0.7, 0.95, 6, endpoint=True)
        out = run_mvgl(Xv, "l2", True, densities, similarities)
    elif method == "svgl":
        densities = np.linspace(0.06, 0.20, 8, endpoint=True)
        out = run_svgl(Xv, densities)

    ############################# 
    ## PERFORMANCE CALCULATION ##
    #############################

    # Construct the output dataframe
    performances = {
        "Run": [], "Method": [], "F1": [], "AUPRC": [], "Graph": [], "Density": [], 
        "Similarity": [], "LearnedDensity": [], "LearnedSimilarity": [], 
        "RunTime": []
    }
    for case, curr_out in out.items():
        density, similarity = case

        for graph in ["view", "consensus"]:
            
            if curr_out[graph] is None:
                continue

            w_gt = wc_gt if graph == "consensus" else wv_gt
            case_similarity = None if graph == "consensus" else similarity

            if graph == "view":
                learned_similarity = metrics.correlation(curr_out[graph], curr_out[graph])
                learned_similarity = (
                    np.mean((np.sum(learned_similarity, axis=1) - 1)/(n_views - 1))
                )
            else:
                learned_similarity = None
            
            performances["Run"].append(seed)
            performances["Method"].append(method)
            performances["F1"].append(np.mean(metrics.f1(w_gt, curr_out[graph])))
            performances["AUPRC"].append(np.mean(metrics.auprc(w_gt, curr_out[graph])))
            performances["Graph"].append(graph)
            performances["Density"].append(density)
            performances["Similarity"].append(case_similarity)
            performances["LearnedDensity"].append(np.mean(metrics.density(curr_out[graph])))
            performances["LearnedSimilarity"].append(learned_similarity)
            performances["RunTime"].append(curr_out["run time"])
    
    ############################# 
    ## PERFORMANCE CALCULATION ##
    #############################

    if len(performances["Run"]) > 0:
        performances = pd.DataFrame.from_dict(performances)

        exp_folder = (f"revision-exp1-{graph_model}-{n_edges_to_add}-"
                      f"{n_edges_to_remove}-{n_views_to_change}")
        save_path = Path(
            mvgl.ROOT_DIR, "data", "simulations", "outputs", exp_folder, method
        )
        save_path.mkdir(parents=True, exist_ok=True)

        save_file = Path(save_path, f"run-{seed}.csv")

        performances.to_csv(save_file)

if __name__ == "__main__":
    main()