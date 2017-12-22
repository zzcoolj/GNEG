import unittest
import graph_based_word2vec as gbw


class TestGraphBasedWord2vec(unittest.TestCase):
    def test_1(self):
        # Fixed parameters for word2vec
        sg = 1  # Only care about skip-gram
        dicts_folder = 'output/intermediate data for unittest/dicts_and_encoded_texts/'

        gs = gbw.GridSearch(training_data_folder='data/training data/unittest_data',
                            index2word_path=dicts_folder + 'dict_merged.txt',
                            merged_word_count_path=dicts_folder + 'word_count_all.txt',
                            valid_vocabulary_path=dicts_folder + 'valid_vocabulary_min_count_5.txt',
                            workers=1, sg=sg, negative=1, potential_ns_len=6)

        # TODO NOW fill out below
        gs.one_search(ns_path='output/intermediate data for unittest/graph/encoded_edges_count_window_size_6_vocab_size_none_undirected_for_unittest_matrix_dict.pickle')
        # gs.one_search(ns_path=None)



if __name__ == '__main__':
    unittest.main()
