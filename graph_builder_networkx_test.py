import unittest
import graph_builder_networkx as gbn


class TestGraphDataProvider(unittest.TestCase):
    graph_folder = 'output/intermediate data for unittest/graph/'

    def test_get_longest_shortest_path_nodes(self):
        graph = gbn.NXGraph(self.graph_folder + 'encoded_edges_count_window_size_6_vocab_size_none.txt',
                            gpickle_name='graph')
        graph.get_shortest_path_lengths_between_all_nodes(output_folder=self.graph_folder)
        graph.get_longest_shortest_path_nodes(3, data_folder=self.graph_folder)


if __name__ == '__main__':
    unittest.main()
