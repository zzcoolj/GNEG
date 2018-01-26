import unittest
import negative_samples_generator as gbn
import graph_builder as gb
import numpy as np


class TestGraphDataProvider(unittest.TestCase):
    graph_folder = 'output/intermediate data for unittest/graph/'
    ns_folder = 'output/intermediate data for unittest/negative_samples/'
    merged_dict_path = 'output/intermediate data for unittest/graph/keep/dict_merged_for_unittest.txt'
    encoded_edges_count_path = 'output/intermediate data for unittest/graph/keep/encoded_edges_count_window_size_6_vocab_size_none_for_unittest.txt'
    merged_dict_undirected_path = 'output/intermediate data for unittest/graph/keep/dict_merged_undirected_for_unittest.txt'
    encoded_edges_count_undirected_path = 'output/intermediate data for unittest/graph/keep/encoded_edges_count_window_size_6_vocab_size_none_undirected_for_unittest.txt'
    valid_vocabulary_undirected_path = 'output/intermediate data for unittest/graph/keep/valid_vocabulary_min_count_5_undirected.txt'
    word_count_undirected_path = 'output/intermediate data for unittest/graph/keep/word_count_all_undirected.txt'

    def test_1_get_ns_dict_by_shortest_path(self):
        # Directed graph
        graph = gbn.NXGraph.from_encoded_edges_count_file(self.encoded_edges_count_path, directed=True,
                                                          output_folder=self.graph_folder)
        graph.print_graph_information()
        nodes, matrix = graph.get_shortest_path_lengths_between_all_nodes(output_folder=self.graph_folder)
        ns = gbn.NegativeSamples(matrix=matrix, graph_index2wordId=nodes, merged_dict_path=self.merged_dict_path,
                                 name_prefix=graph.name_prefix)
        ns.get_and_print_matrix_and_token_order()
        translate_shortest_path_nodes_dict = ns.write_translated_negative_samples_dict(n=3, selected_mode='min',
                                                                                       output_folder=self.graph_folder)

        self.assertTrue('.' in translate_shortest_path_nodes_dict['in'])
        self.assertTrue(',' in translate_shortest_path_nodes_dict['in'])
        self.assertTrue('and' in translate_shortest_path_nodes_dict['in'])

        self.assertTrue('the' in translate_shortest_path_nodes_dict['of'])
        self.assertTrue(',' in translate_shortest_path_nodes_dict['of'])
        self.assertTrue('in' in translate_shortest_path_nodes_dict['of'])

        self.assertFalse('the' in translate_shortest_path_nodes_dict['the'])
        self.assertFalse(',' in translate_shortest_path_nodes_dict[','])
        self.assertFalse('.' in translate_shortest_path_nodes_dict['.'])
        self.assertFalse('and' in translate_shortest_path_nodes_dict['and'])
        self.assertFalse('in' in translate_shortest_path_nodes_dict['in'])
        self.assertFalse('of' in translate_shortest_path_nodes_dict['of'])

        translate_shortest_path_nodes_dict = ns.write_translated_negative_samples_dict(n=3, selected_mode='max',
                                                                                       output_folder=self.graph_folder)

        self.assertTrue('.' in translate_shortest_path_nodes_dict[','])
        self.assertTrue('of' in translate_shortest_path_nodes_dict[','])
        self.assertTrue('and' in translate_shortest_path_nodes_dict[','])

        self.assertTrue('the' in translate_shortest_path_nodes_dict['in'])
        self.assertTrue('of' in translate_shortest_path_nodes_dict['in'])

        self.assertTrue('of' in translate_shortest_path_nodes_dict['the'])
        self.assertTrue('of' in translate_shortest_path_nodes_dict['and'])

        self.assertTrue('.' in translate_shortest_path_nodes_dict['of'])
        self.assertTrue('and' in translate_shortest_path_nodes_dict['of'])

        self.assertFalse('the' in translate_shortest_path_nodes_dict['the'])
        self.assertFalse(',' in translate_shortest_path_nodes_dict[','])
        self.assertFalse('.' in translate_shortest_path_nodes_dict['.'])
        self.assertFalse('and' in translate_shortest_path_nodes_dict['and'])
        self.assertFalse('in' in translate_shortest_path_nodes_dict['in'])
        self.assertFalse('of' in translate_shortest_path_nodes_dict['of'])

        # Undirected
        graph = gbn.NXGraph.from_encoded_edges_count_file(self.encoded_edges_count_undirected_path, directed=False,
                                                          output_folder=self.graph_folder)
        graph.print_graph_information()
        nodes, matrix = graph.get_shortest_path_lengths_between_all_nodes(output_folder=self.graph_folder)

        ns = gbn.NegativeSamples(matrix=matrix, graph_index2wordId=nodes,
                                 merged_dict_path=self.merged_dict_undirected_path,
                                 name_prefix=graph.name_prefix)
        ns.get_and_print_matrix_and_token_order()
        translate_shortest_path_nodes_dict = ns.write_translated_negative_samples_dict(n=3, selected_mode='max',
                                                                                       output_folder=self.graph_folder)
        self.assertTrue('the' in translate_shortest_path_nodes_dict['and'])
        self.assertTrue(',' in translate_shortest_path_nodes_dict['and'])

        self.assertTrue('of' in translate_shortest_path_nodes_dict['.'])
        self.assertTrue(',' in translate_shortest_path_nodes_dict['.'])

        self.assertTrue('the' in translate_shortest_path_nodes_dict['in'])

        self.assertTrue('and' in translate_shortest_path_nodes_dict['the'])
        self.assertTrue('of' in translate_shortest_path_nodes_dict['the'])

        self.assertTrue('the' in translate_shortest_path_nodes_dict['of'])
        self.assertTrue('and' in translate_shortest_path_nodes_dict['of'])
        self.assertTrue('.' in translate_shortest_path_nodes_dict['of'])

        self.assertTrue('and' in translate_shortest_path_nodes_dict[','])
        self.assertTrue('the' in translate_shortest_path_nodes_dict[','])
        self.assertTrue('.' in translate_shortest_path_nodes_dict[','])

    def test_2_print_tokens_negative_samples_and_their_value_in_matrix(self):
        graph = gbn.NXGraph.from_encoded_edges_count_file(self.encoded_edges_count_undirected_path, directed=False,
                                                          output_folder=self.graph_folder)
        nodes, matrix = graph.get_shortest_path_lengths_between_all_nodes(output_folder=self.graph_folder)
        ns = gbn.NegativeSamples(matrix=matrix, graph_index2wordId=nodes,
                                 merged_dict_path=self.merged_dict_undirected_path,
                                 name_prefix=graph.name_prefix)
        ns.write_translated_negative_samples_dict(n=3, selected_mode='max', output_folder=self.graph_folder)
        ns.print_tokens_negative_samples_and_their_value_in_matrix(['the', 'of'])

    def test_3_get_ns_dict_by_t_step_random_walk(self):
        # Undirected
        graph = gbn.NXGraph.from_encoded_edges_count_file(self.encoded_edges_count_undirected_path, directed=False,
                                                          output_folder=self.graph_folder)
        graph.print_graph_information()

        # t=1 step random walk
        nodes, matrix1 = graph.get_t_step_random_walk_stochastic_matrix(t=1)
        ns = gbn.NegativeSamples(matrix=matrix1, graph_index2wordId=nodes,
                                 merged_dict_path=self.merged_dict_undirected_path,
                                 name_prefix=graph.name_prefix)
        ns.get_and_print_matrix_and_token_order()
        # check weight based transition probability
        self.assertTrue(ns.get_matrix_value_by_token_xy('.', 'the') == 2/(2+2+3+1))
        self.assertTrue(ns.get_matrix_value_by_token_xy('and', 'the') == 4/(2+1+4+3+4))
        self.assertTrue(ns.get_matrix_value_by_token_xy('the', ',') == 3/(3+4+2+6+8))
        self.assertTrue(ns.get_matrix_value_by_token_xy(',', '.') == 0)
        self.assertTrue(ns.get_matrix_value_by_token_xy('in', ',') == 2/(2+2+6+1+1))

        # stochastic matrix calculated by NoGraph class
        print('NoGraph')
        no_graph = gbn.NoGraph(self.encoded_edges_count_undirected_path,
                               valid_vocabulary_path=self.valid_vocabulary_undirected_path)
        matrix11 = no_graph.get_stochastic_matrix()
        ns_no_graph = gbn.NegativeSamples(matrix=matrix11,
                                          graph_index2wordId=no_graph.graph_index2wordId,
                                          merged_dict_path=self.merged_dict_undirected_path,
                                          name_prefix=no_graph.name_prefix)
        ns_no_graph.get_and_print_matrix_and_token_order()
        self.assertTrue(ns_no_graph.get_matrix_value_by_token_xy('.', 'the') == 2 / (2 + 2 + 3 + 1))
        self.assertTrue(ns_no_graph.get_matrix_value_by_token_xy('and', 'the') == 4 / (2 + 1 + 4 + 3 + 4))
        self.assertTrue(ns_no_graph.get_matrix_value_by_token_xy('the', ',') == 3 / (3 + 4 + 2 + 6 + 8))
        self.assertTrue(ns_no_graph.get_matrix_value_by_token_xy(',', '.') == 0)
        self.assertTrue(ns_no_graph.get_matrix_value_by_token_xy('in', ',') == 2 / (2 + 2 + 6 + 1 + 1))

        # t=2 steps random walk
        nodes, matrix2 = graph.get_t_step_random_walk_stochastic_matrix(t=2)
        ns = gbn.NegativeSamples(matrix=matrix2, graph_index2wordId=nodes,
                                 merged_dict_path=self.merged_dict_undirected_path,
                                 name_prefix=graph.name_prefix)
        ns.get_and_print_matrix_and_token_order()
        # check the calculation of cell value.
        value_sum = 0
        for i in range(6):
            value_sum += matrix1[3, i] * matrix1[i, 5]
        self.assertTrue(value_sum == matrix2[3, 5])

        # t=2 steps random walk NoGraph
        nodes, matrix22 = no_graph.get_t_step_random_walk_stochastic_matrix(t=2)
        ns_no_graph = gbn.NegativeSamples(matrix=matrix22, graph_index2wordId=nodes,
                                          merged_dict_path=self.merged_dict_undirected_path,
                                          name_prefix=no_graph.name_prefix)
        ns_no_graph.get_and_print_matrix_and_token_order()
        # check the calculation of cell value.
        value_sum = 0
        for i in range(6):
            value_sum += matrix11[3, i] * matrix11[i, 5]
        np.testing.assert_array_almost_equal(value_sum, matrix22[3, 5])

        # t=3 steps random walk
        nodes, matrix3 = graph.get_t_step_random_walk_stochastic_matrix(t=3)
        ns = gbn.NegativeSamples(matrix=matrix3, graph_index2wordId=nodes,
                                 merged_dict_path=self.merged_dict_undirected_path,
                                 name_prefix=graph.name_prefix)
        ns.get_and_print_matrix_and_token_order()
        # check the sum of each line in matrix equals to 1
        for i in range(0, matrix3.shape[0]):
            self.assertTrue(np.sum(matrix3[i]) == 1.0)
        # check the calculation of cell value.
        value_sum = 0
        for i in range(6):
            value_sum += matrix2[3, i] * matrix1[i, 5]  # matrix1 is the transition matrix
        self.assertTrue(value_sum == matrix3[3, 5])

        # t=3 steps random walk NoGraph
        nodes, matrix33 = no_graph.get_t_step_random_walk_stochastic_matrix(t=3)
        ns_no_graph = gbn.NegativeSamples(matrix=matrix33, graph_index2wordId=nodes,
                                          merged_dict_path=self.merged_dict_undirected_path,
                                          name_prefix=no_graph.name_prefix)
        ns_no_graph.get_and_print_matrix_and_token_order()
        # check the sum of each line in matrix equals to 1
        for i in range(0, matrix33.shape[0]):
            np.testing.assert_array_almost_equal(np.sum(matrix33[i]), 1)
        # check the calculation of cell value.
        value_sum = 0
        for i in range(6):
            value_sum += matrix22[3, i] * matrix11[i, 5]  # matrix1 is the transition matrix
        np.testing.assert_array_almost_equal(value_sum, matrix33[3, 5])

    def test_4_reorder_matrix(self):
        # Undirected
        graph = gbn.NXGraph.from_encoded_edges_count_file(self.encoded_edges_count_undirected_path, directed=False)
        # t=1 step random walk
        nodes, matrix1 = graph.get_t_step_random_walk_stochastic_matrix(t=1, output_folder=self.ns_folder)
        ns = gbn.NegativeSamples(matrix=matrix1, graph_index2wordId=nodes,
                                 merged_dict_path=self.merged_dict_undirected_path,
                                 name_prefix=graph.name_prefix)
        matrix, graph_token_order = ns.get_and_print_matrix_and_token_order()
        word2vec_index2word = {0: '.', 1: 'the', 2: ',', 3: 'and', 4: 'in', 5: 'of'}
        word2vec_word2index = {}
        for index, word in word2vec_index2word.items():
            word2vec_word2index[word] = index
        reordered_matrix = ns.reorder_matrix(word2vec_index2word=word2vec_index2word)
        print(reordered_matrix)
        matrix_length = len(graph_token_order)
        for x in range(matrix_length):
            x_token = graph_token_order[x]
            for y in range(matrix_length):
                y_token = graph_token_order[y]
                graph_matrix_value = matrix[x][y]
                word2vec_matrix_value = reordered_matrix[word2vec_word2index[x_token]][word2vec_word2index[y_token]]
                self.assertTrue(graph_matrix_value == word2vec_matrix_value)

    def test_5_reorder_matrix_by_word_count(self):
        # stochastic matrix calculated by NoGraph class
        no_graph = gb.NoGraph(self.encoded_edges_count_undirected_path,
                              valid_vocabulary_path=self.valid_vocabulary_undirected_path)
        print('old order')
        print(no_graph.graph_index2wordId)
        print(no_graph.cooccurrence_matrix)
        ns = gbn.NegativeSamples(matrix=no_graph.cooccurrence_matrix, graph_index2wordId=no_graph.graph_index2wordId,
                                 merged_dict_path=None, name_prefix=None)
        new_wordId_order, reordered_matrix = ns.reorder_matrix_by_word_count(self.word_count_undirected_path)
        print('new order (based on count in descending order)')
        print(new_wordId_order)
        print(reordered_matrix)
        self.assertTrue(new_wordId_order[:3] == [12, 6, 14])
        for i in range(no_graph.cooccurrence_matrix.shape[0]):
            for j in range(no_graph.cooccurrence_matrix.shape[1]):
                new_i = new_wordId_order.index(no_graph.graph_index2wordId[i])
                new_j = new_wordId_order.index(no_graph.graph_index2wordId[j])
                self.assertTrue(no_graph.cooccurrence_matrix[i][j] == reordered_matrix[new_i][new_j])


if __name__ == '__main__':
    unittest.main()
