import unittest
import graph_builder_networkx as gbn


class TestGraphDataProvider(unittest.TestCase):
    graph_folder = 'output/intermediate data for unittest/graph/'
    merged_dict_path = 'output/intermediate data for unittest/dicts_and_encoded_texts/dict_merged.txt'

    def test_get_shortest_shortest_path_nodes(self):
        graph = gbn.NXGraph(self.graph_folder + 'encoded_edges_count_window_size_6_vocab_size_none.txt',
                            gpickle_name='graph.gpickle')
        graph.get_shortest_path_lengths_between_all_nodes(output_folder=self.graph_folder)
        translate_shortest_path_nodes_dict = gbn.NXGraph.translate_shortest_path_nodes_dict(
            gbn.NXGraph.get_selected_shortest_path_nodes(3, selected_mode='min', data_folder=self.graph_folder),
            self.merged_dict_path, output_folder=self.graph_folder)
        print(translate_shortest_path_nodes_dict)


if __name__ == '__main__':
    unittest.main()
