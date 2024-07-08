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

from script_funs import run_mvgl, run_svgl, run_jemgl

@click.command()
@click.option("--method", default="jemgl-laplacian", show_default=True)
@click.option("--graph-model", default="er", show_default=True)
@click.option("--n-views", default=6, show_default=True)
@click.option("--seed", default=None, type=int, show_default=True)
def main(
    method, graph_model, n_views, seed
):
    n_nodes = 100
    n_signals = 500
    
    ###############################
    ## SIMULATED DATA GENERATION ##
    ###############################

    Gc, Gv, Xv = mvgl.gen_simulated_data(
        n_nodes, n_views, n_signals, graph_generator=graph_model, p=0.1, m=5, 
        perturbation=0.1, signal_type="smooth", noise=0.1, 
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
    elif method == "jemgl-group":
        rho_ns = 10**(-2 + 2*np.arange(0, 21)/15)
        out = run_jemgl(Xv, "group", rho_ns)
    elif method == "jemgl-laplacian":
        rho_ns = 10**(-2 + 2*np.arange(0, 21)/15)
        out = run_jemgl(Xv, "laplacian", rho_ns)

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

        graphs = {"view": curr_out["view"]}
        
        if curr_out["consensus"] is None:
            graphs["consensus-mean"] = np.mean(curr_out["view"], axis=0)
            graphs["consensus-median"] = np.median(curr_out["view"], axis=0)
        else:
            graphs["consensus"] = curr_out["consensus"]

        for graph, w_hat in graphs.items():
            w_gt = wc_gt if graph.startswith("consensus") else wv_gt
            case_similarity = None if graph.startswith("consensus") else similarity

            if graph == "view":
                similarities_hat = metrics.correlation(w_hat, w_hat)
                similarities_hat = similarities_hat[np.triu_indices(len(w_hat), k=1)]
                similarity_hat = np.mean(similarities_hat)
            else:
                similarity_hat = None
            
            performances["Run"].append(seed)
            performances["Method"].append(method)
            performances["F1"].append(np.mean(metrics.f1(w_gt, w_hat)))
            performances["AUPRC"].append(np.mean(metrics.auprc(w_gt, w_hat)))
            performances["Graph"].append(graph)
            performances["Density"].append(density)
            performances["Similarity"].append(case_similarity)
            performances["LearnedDensity"].append(np.mean(metrics.density(w_hat)))
            performances["LearnedSimilarity"].append(similarity_hat)
            performances["RunTime"].append(curr_out["run time"])
    
    ############################# 
    ## PERFORMANCE CALCULATION ##
    #############################

    if len(performances["Run"]) > 0:
        performances = pd.DataFrame.from_dict(performances)

        save_path = Path(
            mvgl.ROOT_DIR, "data", "simulations", "outputs", 
            f"exp1-{graph_model}-{n_views:d}", method
        )
        save_path.mkdir(parents=True, exist_ok=True)

        save_file = Path(save_path, f"run-{seed}.csv")

        performances.to_csv(save_file)

if __name__ == "__main__":
    main()