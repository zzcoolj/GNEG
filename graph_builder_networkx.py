import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import configparser
import sys
sys.path.insert(0, '../common/')
import common
import multi_processing
import graph_data_provider as gdp

config = configparser.ConfigParser()
config.read('config.ini')


class NXGraph:
    def __init__(self, path, gpickle_name=None, directed=config.getboolean("graph", "directed")):
        if path.endswith('.gpickle'):
            self.graph = nx.read_gpickle(path)
        elif path.endswith('.txt'):
            self.graph = self.create_graph_with_weighted_edges(path, directed=directed)
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
        number_of_edges = self.graph.number_of_edges()
        number_of_selfloops = self.graph.number_of_selfloops()
        number_of_nodes = self.graph.number_of_nodes()
        if nx.is_directed(self.graph):
            print('The graph is directed.')
            connected_edges_proportion = round(
                (number_of_edges - number_of_selfloops) / (number_of_nodes * (number_of_nodes - 1)) * 100, 2)
        else:
            print('The graph is undirected.')
            connected_edges_proportion = round(
                (number_of_edges - number_of_selfloops) / ((number_of_nodes * (number_of_nodes - 1)) / 2) * 100, 2)
        print("#nodes:", number_of_nodes, "#edges:",  number_of_edges, "#selfloops:", number_of_selfloops)
        print(str(connected_edges_proportion) + '% of the node pairs are connected via edges.')
        # TODO Code below takes long time to calculate for big graphs.
        print('Average shortest path length (weight=None):', str(round(nx.average_shortest_path_length(self.graph), 2)))
        # TODO LATER: average_clustering has not implemented for undirected graph yet.
        if not nx.is_directed(self.graph):
            # For unweighted graphs, the clustering of a node
            # is the fraction of possible triangles through that node that exist
            print('The clustering coefficient for the graph is ' + str(
                round(nx.average_clustering(self.graph, weight=None), 2)))

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

        """ ATTENTION
        'floyd_warshall_numpy' has already considered situations below:
        1. If there's no path between source and target node, matrix will put 'inf'
        2. No matter how much the weight is between node and node itself(self loop), the shortest path will always be 0.
        """

        matrix = nx.floyd_warshall_numpy(self.graph)
        np.save(output_folder + 'matrix.npy', matrix, fix_imports=False)
        common.write_to_pickle(self.graph.nodes(), output_folder + 'nodes.pickle')
        return self.graph.nodes, matrix

    @staticmethod
    def get_selected_shortest_path_nodes(n, selected_mode, data_folder=config['graph']['graph_folder']):
        # TODO NOW Deal with inf
        """e.g.
        nodes -> a list of word indices (here word index is there index in merged dict.)
        [index2word[node] for node in nodes] -> ['the', '.', ',', 'and', 'in', 'of']
        matrix -> matrix's lines and rows follow the order of nodes, the value in each cell is the shortest path length.
            [[  0.   2.   2.   2.   1.   6.]
             [ inf   0.  inf  inf  inf  inf]
             [  1.   2.   0.   2.   1.   7.]
             [  1.   2.   2.   0.   2.   7.]
             [  2.   1.   1.   1.   0.   8.]
             [  2.   3.   1.   3.   2.   0.]]
        """
        n += 1  # add one more potential results, in case results have self loop node.
        matrix = np.load(data_folder + 'matrix.npy')
        nodes = common.read_pickle(data_folder + 'nodes.pickle')
        nodes_list = list(nodes)
        if selected_mode == 'min':
            selected_indices = np.argpartition(matrix, n)[:, :n]
        elif selected_mode == 'max':
            selected_indices = np.argpartition(matrix, -n)[:, -n:]
        # indices here means the indices of nodes list, not the value(word index) inside nodes list.
        cleaned_selected_indices = np.empty([selected_indices.shape[0], n - 1], dtype=int)
        shortest_path_nodes_dict = {}
        for i in range(matrix.shape[1]):  # shape[0] = shape[1]
            # e.g. for the first row (i=0), find the index in selected_indices where the value equals 0 (self loop)
            self_loop_index = np.where(selected_indices[i] == i)
            if self_loop_index[0].size == 0:  # no self loop
                shortest_path = matrix[i][selected_indices[i]]
                selected_index_shortest_path_length_dict = dict(zip(selected_indices[i], shortest_path))
                sorted_indices = sorted(selected_index_shortest_path_length_dict,
                                        key=selected_index_shortest_path_length_dict.get)
                if selected_mode == 'min':
                    cleaned_selected_indices[i] = sorted_indices[:n - 1]
                elif selected_mode == 'max':
                    cleaned_selected_indices[i] = sorted_indices[1:]
            else:
                cleaned_selected_indices[i] = np.delete(selected_indices[i], self_loop_index)
            # translate nodes list indices to words indices (nodes list values),
            # and the row's order follows the order of nodes
            shortest_path_nodes_dict[nodes_list[i]] = np.array(nodes)[cleaned_selected_indices[i]].tolist()
        # common.write_to_pickle(shortest_path_nodes_dict, data_folder+'shortest_path_nodes_dict.pickle')
        return shortest_path_nodes_dict

    @staticmethod
    def translate_shortest_path_nodes_dict(shortest_path_nodes_dict, index2word_path, output_folder):
        index2word = get_index2word(file=index2word_path)
        translated_shortest_path_nodes_dict = {}
        for key, value in shortest_path_nodes_dict.items():
            translated_shortest_path_nodes_dict[index2word[key]] = [index2word[node_id] for node_id in value]
        common.write_to_pickle(translated_shortest_path_nodes_dict,
                               output_folder + 'translated_shortest_path_nodes_dict.pickle')
        return translated_shortest_path_nodes_dict

    @staticmethod
    def negative_samples_detail(translated_shortest_path_nodes_dict_path, merged_dict_path, matrix_path, nodes_path,
                                words_list):
        translated_shortest_path_nodes_dict = common.read_pickle(translated_shortest_path_nodes_dict_path)
        word2index = gdp.read_two_columns_file_to_build_dictionary_type_specified(
            file=merged_dict_path, key_type=str, value_type=int)
        matrix = np.load(matrix_path)
        nodes = list(common.read_pickle(nodes_path))
        for word in words_list:
            print('For word:', word)
            word_index = word2index[word]
            matrix_x = nodes.index(word_index)
            ns_words = translated_shortest_path_nodes_dict[word]
            for ns_word in ns_words:
                ns_word_index = word2index[ns_word]
                matrix_y = nodes.index(ns_word_index)
                matrix_cell_value = matrix[matrix_x][matrix_y]
                print(' ', ns_word, matrix_cell_value)


def get_index2word(file, key_type=int, value_type=str):
    """ATTENTION
    This function is different from what in graph_data_provider.
    Here, key is id and token is value, while in graph_data_provider, token is key and id is value.
    """
    d = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            (key, val) = line.rstrip('\n').split("\t")
            d[key_type(val)] = value_type(key)
        return d


if __name__ == '__main__':
    graph = NXGraph(config['graph']['graph_folder'] + 'encoded_edges_count_window_size_5.txt',
                    gpickle_name='graph.gpickle')
    graph.show_detailed_information()

    # graph.get_shortest_path_lengths_between_all_nodes(output_folder=config['graph']['graph_folder'])
    # translated_shortest_path_nodes_dict = NXGraph.translate_shortest_path_nodes_dict(
    #     NXGraph.get_selected_shortest_path_nodes(20, selected_mode='max', data_folder=config['graph']['graph_folder']),
    #     config['graph']['dicts_and_encoded_texts_folder']+'dict_merged.txt',
    #     output_folder=config['graph']['graph_folder'])
