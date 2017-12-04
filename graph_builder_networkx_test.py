import unittest
import graph_builder_networkx as gbn
import networkx as nx
import numpy as np

class TestGraphDataProvider(unittest.TestCase):
    graph_folder = 'output/intermediate data for unittest/graph/'
    merged_dict_path = 'output/intermediate data for unittest/graph/keep/dict_merged_for_unittest.txt'
    encoded_edges_count_path = 'output/intermediate data for unittest/graph/keep/encoded_edges_count_window_size_6_vocab_size_none_for_unittest.txt'
    merged_dict_undirected_path = 'output/intermediate data for unittest/graph/keep/dict_merged_undirected_for_unittest.txt'
    encoded_edges_count_undirected_path = 'output/intermediate data for unittest/graph/keep/encoded_edges_count_window_size_6_vocab_size_none_undirected_for_unittest.txt'

    def test_1_translate_shortest_path_nodes_dict(self):
        # Directed graph
        graph = gbn.NXGraph(self.encoded_edges_count_path, gpickle_name='graph.gpickle', directed=True)
        graph.show_graph_information()
        nodes, matrix = graph.get_shortest_path_lengths_between_all_nodes(output_folder=self.graph_folder)
        index2word = gbn.get_index2word(file=self.merged_dict_path)
        print([index2word[node] for node in nodes])
        print(matrix)
        print()

        translate_shortest_path_nodes_dict = gbn.NXGraph.translate_shortest_path_nodes_dict(
            gbn.NXGraph.get_selected_shortest_path_nodes(3, selected_mode='min', data_folder=self.graph_folder),
            self.merged_dict_path, output_folder=self.graph_folder)

        # for key, value in translate_shortest_path_nodes_dict.items():
        #     print(key, '\t', value)

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

        translate_shortest_path_nodes_dict = gbn.NXGraph.translate_shortest_path_nodes_dict(
            gbn.NXGraph.get_selected_shortest_path_nodes(3, selected_mode='max', data_folder=self.graph_folder),
            self.merged_dict_path, output_folder=self.graph_folder)

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
        graph = gbn.NXGraph(self.encoded_edges_count_undirected_path, gpickle_name='graph.gpickle')
        graph.show_graph_information()
        nodes, matrix = graph.get_shortest_path_lengths_between_all_nodes(output_folder=self.graph_folder)
        index2word = gbn.get_index2word(file=self.merged_dict_undirected_path)
        print([index2word[node] for node in nodes])
        print(matrix)
        print()
        translate_shortest_path_nodes_dict = gbn.NXGraph.translate_shortest_path_nodes_dict(
            gbn.NXGraph.get_selected_shortest_path_nodes(3, selected_mode='max', data_folder=self.graph_folder),
            self.merged_dict_undirected_path, output_folder=self.graph_folder)
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

    def test_2_negative_samples_detail(self):
        gbn.NXGraph.negative_samples_detail(
            translated_shortest_path_nodes_dict_path=self.graph_folder+'translated_shortest_path_nodes_dict.pickle',
            merged_dict_path=self.graph_folder+'keep/dict_merged_undirected_for_unittest.txt',
            matrix_path=self.graph_folder+'matrix.npy',
            nodes_path=self.graph_folder+'nodes.pickle',
            words_list=['the', 'of'])

        graph = gbn.NXGraph(self.encoded_edges_count_undirected_path, gpickle_name='graph.gpickle')
        print(nx.to_numpy_matrix(graph.graph))
        stochastic_graph = graph.stochastic_graph_for_undirected_graph()
        stochastic_graph_matrix = nx.to_numpy_matrix(stochastic_graph)
        print(stochastic_graph_matrix)
        print(np.matmul(stochastic_graph_matrix, stochastic_graph_matrix))

if __name__ == '__main__':
    unittest.main()
