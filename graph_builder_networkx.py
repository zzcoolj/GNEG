import networkx as nx
import matplotlib.pyplot as plt
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

    def get_shortest_path_lengths_between_all_nodes(self, output_folder=config['graph']['graph_folder']):
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
        np.save(output_folder + 'matrix.npy', matrix, fix_imports=False)
        common.write_to_pickle(self.graph.nodes(), output_folder + 'nodes.pickle')

    @staticmethod
    def get_longest_shortest_path_nodes(n, data_folder=config['graph']['graph_folder']):
        # TODO NOW Deal with inf
        matrix = np.load(data_folder + 'matrix.npy')
        nodes = common.read_pickle(data_folder + 'nodes.pickle')
        largest_indices = np.argpartition(matrix, -n)[:, -n:]
        cleaned_largest_indices = np.empty([largest_indices.shape[0], n-1], dtype=int)
        result_nodes = np.empty([largest_indices.shape[0], n-1], dtype=int)
        for i in range(matrix.shape[1]):
            index_to_remove = np.where(largest_indices[i]==i)
            if index_to_remove[0].size == 0:
                cleaned_largest_indices[i] = largest_indices[i][1:]
            else:
                cleaned_largest_indices[i] = np.delete(largest_indices[i], index_to_remove)
            result_nodes[i] = np.array(nodes)[cleaned_largest_indices[i]]
        return result_nodes

    @staticmethod
    def get_shortest_shortest_path_nodes(n, data_folder=config['graph']['graph_folder']):
        # TODO NOW Deal with inf
        matrix = np.load(data_folder + 'matrix.npy')
        nodes = common.read_pickle(data_folder + 'nodes.pickle')
        shortest_indices = np.argpartition(matrix, n)[:, :n]
        cleaned_shortest_indices = np.empty([shortest_indices.shape[0], n - 1], dtype=int)
        result_nodes = np.empty([shortest_indices.shape[0], n - 1], dtype=int)
        for i in range(matrix.shape[1]):
            index_to_remove = np.where(shortest_indices[i] == i)
            if index_to_remove[0].size == 0:
                cleaned_shortest_indices[i] = shortest_indices[i][:n-1]
            else:
                cleaned_shortest_indices[i] = np.delete(shortest_indices[i], index_to_remove)
            result_nodes[i] = np.array(nodes)[cleaned_shortest_indices[i]]
        return result_nodes


if __name__ == '__main__':
    # graph = NXGraph(config['graph']['graph_folder']+'graph.gpickle')
    # graph = NXGraph('output/intermediate data for unittest/graph/encoded_edges_count_window_size_6.txt', gpickle_name='test')
    NXGraph.get_longest_shortest_path_nodes(n=21)

