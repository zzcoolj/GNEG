import unittest
import graph_builder as gb


class TestGraphBuilder(unittest.TestCase):
    encoded_edges_count_undirected_path = 'output/intermediate data for unittest/graph/keep/encoded_edges_count_window_size_6_vocab_size_none_undirected_for_unittest.txt'
    valid_vocabulary_undirected_path = 'output/intermediate data for unittest/graph/keep/valid_vocabulary_min_count_5_undirected.txt'

    def test_1_get_ns_dict_by_shortest_path(self):
        # stochastic matrix calculated by NoGraph class
        print('NoGraph')
        no_graph = gb.NoGraph(self.encoded_edges_count_undirected_path,
                              valid_vocabulary_path=self.valid_vocabulary_undirected_path)
        print(no_graph.graph_index2wordId)
        print(no_graph.cooccurrence_matrix)
