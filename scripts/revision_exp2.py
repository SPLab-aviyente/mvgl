r"""Script for edge swapping based experiments

In this script methods are applied to a datasets where view graphs are generated
from the consensus graph using edge swapping. Namely, the script first generates
a consensus graph :math:`G` with 100 nodes using either ER graph model (edge
probability set 0.1) or BA graph model (growth parameter is set 5). 6 view 
graphs are then independently generated from :math:`G` by randomly swapping 
`perturbation` fraction of its edges.
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd

import mvgl
import mvgl.data
from mvgl.graphlearning import utils, metrics

from script_funs import run_mvgl, run_svgl

@click.command()
@click.option("--method", default="svgl", show_default=True)
@click.option("--graph-model", default="er", show_default=True)
@click.option("--perturbation", default=0.1, show_default=True)
@click.option("--seed", default=None, type=int, show_default=True)
def main(
    method, graph_model, perturbation, seed
):
    n_views = 6
    n_nodes = 100
    n_signals = 500
    
    ###############################
    ## SIMULATED DATA GENERATION ##
    ###############################

    Gc, Gv, Xv = mvgl.gen_simulated_data(
        n_nodes, n_views, n_signals, graph_generator=graph_model, p=0.1, m=5, 
        perturbation=perturbation, signal_type="smooth", noise=0.1, 
    )

    wc_gt= utils.vectorize_a_graph(Gc)
    wv_gt = []
    for i in range(n_views):
        wv_gt.append(utils.vectorize_a_graph(Gv[i]))

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

        save_path = Path(
            mvgl.ROOT_DIR, "data", "simulations", "outputs", 
            f"revision-exp2-{graph_model}-{perturbation:.2f}", 
            method
        )
        save_path.mkdir(parents=True, exist_ok=True)

        save_file = Path(save_path, f"run-{seed}.csv")

        performances.to_csv(save_file)

if __name__ == "__main__":
    main()