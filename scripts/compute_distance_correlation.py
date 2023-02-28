import sys
import pickle
import numpy
from scipy.stats import pearsonr

distances1 = numpy.asarray(pickle.load(open(sys.argv[1], "rb")))
distances2 = numpy.asarray(pickle.load(open(sys.argv[2], "rb")))

assert distances1.shape == distances2.shape
num_instances = distances1.shape[0]

# Considering only upper triangle, ignoring the diagonals (which should be 0s)
distances1_list = distances1[numpy.triu_indices(n=num_instances, k=1)]
distances2_list = distances2[numpy.triu_indices(n=num_instances, k=1)]

correl = pearsonr(distances1_list, distances2_list)
print(correl)