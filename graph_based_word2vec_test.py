import unittest
import graph_based_word2vec as gbw


class TestGraphBasedWord2vec(unittest.TestCase):
    def test_1(self):
        # Fixed parameters for word2vec
        sg = 1  # Only care about skip-gram
        keep_folder = 'output/intermediate data for unittest/graph/keep/'

        gs = gbw.GridSearch_old(training_data_folder='data/training data/unittest_data',
                                index2word_path=keep_folder + 'dict_merged_undirected_for_unittest.txt',
                                merged_word_count_path=keep_folder + 'word_count_all_undirected.txt',
                                valid_vocabulary_path=keep_folder + 'valid_vocabulary_min_count_5_undirected.txt',
                                workers=1, sg=sg, negative=1, potential_ns_len=6)

        # gs.one_search(ns_path=None)
        gs.one_search_distribution(matrix_path='output/intermediate data for unittest/negative_samples/encoded_edges_count_window_size_6_vocab_size_none_undirected_for_unittest_1_step_rw_matrix.npy',
                                   row_column_indices_value_path='output/intermediate data for unittest/negative_samples/encoded_edges_count_window_size_6_vocab_size_none_undirected_for_unittest_1_step_rw_nodes.pickle')


if __name__ == '__main__':
    unittest.main()
