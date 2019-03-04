from graph_builder import NoGraph
import numpy as np
from numpy import power as np_power, sum as np_sum


# power_normalization function is used for graph-based fastText experiments with Ben.
def power_normalization(power, matrix_path):
    original_matrix = np.load(matrix_path)
    powered_matrix = np_power(original_matrix, power)
    train_words_pow = np_sum(powered_matrix, axis=1, keepdims=True)  # sum of each row and preserve the dimension
    normalized_power_matrix = powered_matrix / train_words_pow
    return normalized_power_matrix


m = power_normalization(0.25, '/Users/zzcoolj/Desktop/ns/random walk/encoded_edges_count_window_size_5_undirected_2_step_rw_matrix.npy')
np.save('/Users/zzcoolj/Desktop/ns/difference_window5_t2_noSelfLoops_p0.25_matrix.npy', m, fix_imports=False)
print(m.shape)
print(len(m[0]))
print(sum(m[0]))
print(m[0][0])
print(m[0][1])
print(m[0][2])
print(m[0][57])
print(m[0][9989])
