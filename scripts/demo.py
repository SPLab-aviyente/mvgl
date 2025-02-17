from sklearn.preprocessing import StandardScaler

import mvgl
from mvgl.graphlearning import utils, metrics

n_nodes = 100
n_signals = 500
n_views = 3
seed = 1
consensus_kws = {"graph_generator": "er", "p": 0.1}
view_kws = {"perturbation": 0.1}

signal_kws = {"filter": "gaussian"}

# Generate simulated data
Gc, Gv, Xv = mvgl.gen_simulated_data(
    n_nodes,
    n_views,
    n_signals,
    "smooth",
    noise=0.1,
    consensus_kws=consensus_kws,
    view_kws=view_kws,
    signal_kws=signal_kws,
    seed=seed,
)

# Data pre-processing
scaler = StandardScaler()
for i in range(len(Xv)):
    Xv[i] = scaler.fit_transform(Xv[i].T).T

## Learn the graph: check docstring for learn_multiview_graph for further info
alpha = 0.1
beta = 0.1

wv_hat, wc_hat, params, run_time = mvgl.learn_multiview_graph(
    Xv,
    alpha=alpha,
    beta=beta,
    consensus="l1",
    view_density=0.1, # when this set, a bisection search is performed to find 
                      # alpha that will provide a graph with this density
    similarity=0.8, # when this set, a bisection search is done to find beta 
                    # that will ensure correlation across views is this value
)

# Performance measurement
wc_gt = utils.vectorize_a_graph(Gc)
wv_gt = []
for i in range(len(Xv)):
    wv_gt.append(utils.vectorize_a_graph(Gv[i]))

print(metrics.f1(wc_gt, wc_hat))
print(metrics.f1(wv_gt, wv_hat))
