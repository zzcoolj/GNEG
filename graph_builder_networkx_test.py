import unittest
import graph_builder_networkx as gbn


class TestGraphDataProvider(unittest.TestCase):
    graph_folder = 'output/intermediate data for unittest/graph/'
    merged_dict_path = 'output/intermediate data for unittest/graph/keep/dict_merged_for_unittest.txt'
    encoded_edges_count_path = 'output/intermediate data for unittest/graph/keep/encoded_edges_count_window_size_6_vocab_size_none_for_unittest.txt'
    merged_dict_undirected_path = 'output/intermediate data for unittest/graph/keep/dict_merged_undirected_for_unittest.txt'
    encoded_edges_count_undirected_path = 'output/intermediate data for unittest/graph/keep/encoded_edges_count_window_size_6_vocab_size_none_undirected_for_unittest.txt'

    def test_1_get_ns_dict_by_shortest_path(self):
        # Directed graph
        graph = gbn.NXGraph.from_encoded_edges_count_file(self.encoded_edges_count_path, directed=True,
                                                          output_folder=self.graph_folder)
        graph.print_graph_information()
        nodes, matrix = graph.get_shortest_path_lengths_between_all_nodes(output_folder=self.graph_folder)

        index2word = gbn.get_index2word(file=self.merged_dict_path)
        print([index2word[node] for node in nodes])
        print(matrix)
        print()

        ns = gbn.NegativeSamples(matrix=matrix, row_column_indices_value=nodes, merged_dict_path=self.merged_dict_path)
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

        index2word = gbn.get_index2word(file=self.merged_dict_undirected_path)
        print([index2word[node] for node in nodes])
        print(matrix)
        print()

        ns = gbn.NegativeSamples(matrix=matrix, row_column_indices_value=nodes,
                                 merged_dict_path=self.merged_dict_undirected_path)
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
        ns = gbn.NegativeSamples(matrix=matrix, row_column_indices_value=nodes,
                                 merged_dict_path=self.merged_dict_undirected_path)
        ns.write_translated_negative_samples_dict(n=3, selected_mode='max', output_folder=self.graph_folder)
        ns.print_tokens_negative_samples_and_their_value_in_matrix(['the', 'of'])

    def test_3_get_ns_dict_by_t_step_random_walk(self):
        # Undirected
        graph = gbn.NXGraph.from_encoded_edges_count_file(self.encoded_edges_count_undirected_path, directed=False,
                                                          output_folder=self.graph_folder)
        graph.print_graph_information()
        nodes, matrix = graph.get_t_step_random_walk_stochastic_matrix(t=1)

        index2word = gbn.get_index2word(file=self.merged_dict_undirected_path)
        print([index2word[node] for node in nodes])
        print(matrix)
        print()

        ns = gbn.NegativeSamples(matrix=matrix, row_column_indices_value=nodes, merged_dict_path=self.merged_dict_undirected_path)
        translate_shortest_path_nodes_dict = ns.write_translated_negative_samples_dict(n=3, selected_mode='min',
                                                                                       output_folder=self.graph_folder)
        print(translate_shortest_path_nodes_dict)



if __name__ == '__main__':
    unittest.main()
