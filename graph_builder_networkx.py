import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Process
import time
import numpy as np
import configparser
import sys
sys.path.insert(0, '../common/')
import common
import multi_processing


config = configparser.ConfigParser()
config.read('config.ini')


class NXGraph:
    def __init__(self, path, gpickle_name=None):
        if path.endswith('.gpickle'):
            self.graph = nx.read_gpickle(path)
        elif path.endswith('.txt'):
            self.graph = self.create_graph_with_weighted_edges(path, directed=True)
            nx.write_gpickle(self.graph, multi_processing.get_file_folder(path) + '/' + gpickle_name)



    @staticmethod
    def create_graph_with_weighted_edges(edges_file, directed):
        if directed:
            graph = nx.read_weighted_edgelist(edges_file, create_using=nx.DiGraph(), nodetype=int)
        else:
            graph = nx.read_weighted_edgelist(edges_file, create_using=nx.Graph(), nodetype=int)
        return graph

    @staticmethod
    def create_graph_with_token_list(window):
        """ Usage
        For test, 8 times "is"
        create_graph_with_token_list(["This", "is", "is", "is", "is", "is", "is", "is", "is", "a"])
        """
        g = nx.DiGraph()
        g.add_nodes_from(set(window))
        window_size = len(window)
        for start_index in range(window_size):
            for end_index in range(start_index + 1, window_size):
                # !!! We don't care self-loop edges
                if not window[start_index] == window[end_index]:
                    if g.has_edge(window[start_index], window[end_index]):
                        g[window[start_index]][window[end_index]]['weight'] += 1
                    else:
                        g.add_edge(window[start_index], window[end_index], weight=1)
        return g

    def draw_graph(self):
        """
        Takes too much time with big data.
        """
        nx.draw(self.graph, with_labels=True)

        plt.show()

    def show_detailed_information(self):
        print("#nodes:", self.graph.number_of_nodes(), "#edges:", self.graph.number_of_edges())

    def get_shortest_path_lengths_between_all_nodes(self):
        """
        From test, these three algorithms below take more than 20 hours (processes have been killed after 20 hours) to
        calculate.
        'floyd_warshall_numpy' takes around 100 minutes to get the result.
        """
        # length1 = dict(nx.all_pairs_dijkstra_path_length(g))
        # length2 = dict(nx.all_pairs_bellman_ford_path_length(g))
        # length3 = nx.johnson(g, weight='weight')
        # for node in [0, 1, 2, 3, 4]:
        #     print('1 - {}: {}'.format(node, length2[1][node]))

        matrix = nx.floyd_warshall_numpy(self.graph)
        np.save(config['graph']['graph_folder'] + 'matrix.npy', matrix, fix_imports=False)
        common.write_to_pickle(g.nodes(), config['graph']['graph_folder'] + 'nodes.pickle')

    def get_longest_shortest_path_nodes(self, g, source, n):
        # TODO change
        pass

        # sorted_nodes = list(sorted(shortest_path_dict, key=shortest_path_dict.get, reverse=True))
        # return sorted_nodes[:n]


if __name__ == '__main__':
    # graph = NXGraph(config['graph']['graph_folder']+'graph.gpickle')
    # graph = NXGraph('output/intermediate data for unittest/graph/encoded_edges_count_window_size_6.txt', gpickle_name='test')

    matrix = np.load(config['graph']['graph_folder'] + 'matrix.npy')
    start = time.time()
    print('start')
    shortest_20_indices = np.argpartition(matrix, 5)[:,5]
    print(shortest_20_indices[0])
    print(matrix[0][shortest_20_indices][0])
    largest_20_indices = np.argpartition(matrix, -5)[:,-5]
    print(largest_20_indices[0])
    print(matrix[0][largest_20_indices][0])
    print(max(matrix[0]))
    print(common.count_time(start))
