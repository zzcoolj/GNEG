import unittest
import graph_builder_networkx as gbn


class TestGraphDataProvider(unittest.TestCase):
    graph_folder = 'output/intermediate data for unittest/graph/'
    merged_dict_path = 'output/intermediate data for unittest/graph/keep/dict_merged_for_unittest.txt'

    def test_translate_shortest_path_nodes_dict(self):
        # Directed graph
        graph = gbn.NXGraph(self.graph_folder + 'keep/encoded_edges_count_window_size_6_vocab_size_none_for_unittest.txt',
                            gpickle_name='graph.gpickle',
                            directed=True)
        nodes, matrix = graph.get_shortest_path_lengths_between_all_nodes(output_folder=self.graph_folder)
        index2word = gbn.read_two_columns_file_to_build_dictionary_type_specified(file=self.merged_dict_path)
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
        graph = gbn.NXGraph(self.graph_folder + 'encoded_edges_count_window_size_6_undirected.txt',
                            gpickle_name='graph.gpickle')
        nodes, matrix = graph.get_shortest_path_lengths_between_all_nodes(output_folder=self.graph_folder)
        index2word = gbn.read_two_columns_file_to_build_dictionary_type_specified(file=self.graph_folder+ 'dict_merged.txt')
        print([index2word[node] for node in nodes])
        print(matrix)
        print()


if __name__ == '__main__':
    unittest.main()
