import mvgl
from mvgl.graphlearning import utils, metrics

# Generate simulated data
Gc, Gv, Xv = mvgl.gen_simulated_data(100, 12, 500, "er", 0.1, 0.1, 0.1, 1)