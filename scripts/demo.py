from sklearn.preprocessing import StandardScaler

import mvgl
from mvgl.graphlearning import utils, metrics

# Generate simulated data
Gc, Gv, Xv = mvgl.gen_simulated_data(100, 12, 500, "er", 0.1, "smooth", 0.1, 0.1, 1)

# Data pre-processing
scaler = StandardScaler()
for i in range(len(Xv)):
    Xv[i] = scaler.fit_transform(Xv[i].T).T

# Learn the graph
wv_hat, wc_hat, params, rt = mvgl.learn_multiview_graph(
    Xv, 0.1, 0.1, "l1", view_density=0.1, similarity=0.8
)

# Performance measurement
wc_gt= utils.vectorize_a_graph(Gc)
wv_gt = []
for i in range(len(Xv)):
    wv_gt.append(utils.vectorize_a_graph(Gv[i]))

print(metrics.f1(wc_gt, wc_hat))
print(metrics.f1(wv_gt, wv_hat))

