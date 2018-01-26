import unittest
import graph_builder as gb


class TestGraphBuilder(unittest.TestCase):
    encoded_edges_count_undirected_path = 'output/intermediate data for unittest/graph/keep/encoded_edges_count_window_size_6_vocab_size_none_undirected_for_unittest.txt'
    valid_vocabulary_undirected_path = 'output/intermediate data for unittest/graph/keep/valid_vocabulary_min_count_5_undirected.txt'
    word_count_undirected_path = 'output/intermediate data for unittest/graph/keep/word_count_all_undirected.txt'

    def test_1_NoGraph_reorder_matrix(self):
        # stochastic matrix calculated by NoGraph class
        no_graph = gb.NoGraph(self.encoded_edges_count_undirected_path,
                              valid_vocabulary_path=self.valid_vocabulary_undirected_path)
        print('old order')
        print(no_graph.graph_index2wordId)
        print(no_graph.cooccurrence_matrix)
        new_wordId_order, reordered_matrix = no_graph.reorder_matrix(no_graph.cooccurrence_matrix,
                                                                     word_count_path=self.word_count_undirected_path)
        print('new order (based on count in descending order)')
        print(new_wordId_order)
        print(reordered_matrix)
        self.assertTrue(new_wordId_order[:3] == [12, 6, 14])
        for i in range(no_graph.cooccurrence_matrix.shape[0]):
            for j in range(no_graph.cooccurrence_matrix.shape[1]):
                new_i = new_wordId_order.index(no_graph.graph_index2wordId[i])
                new_j = new_wordId_order.index(no_graph.graph_index2wordId[j])
                self.assertTrue(no_graph.cooccurrence_matrix[i][j] == reordered_matrix[new_i][new_j])
