__author__ = 'Zheng ZHANG'

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
    def __init__(self, graph, name_prefix, directed):
        # name_prefix = encoded_edges_count file's name - '.txt' => encoded_edges_count file names must be unique.
        self.name_prefix = name_prefix
        self.graph = graph
        self.directed = directed

    @classmethod
    def from_gpickle(cls, path):
        name_prefix = multi_processing.get_file_name(path).split('.')[0]
        graph = nx.read_gpickle(path)
        return cls(graph, name_prefix, nx.is_directed(graph))

    @classmethod
    def from_encoded_edges_count_file(cls, path, directed=config.getboolean("graph", "directed"),
                                      output_folder=config['graph']['graph_folder']):
        name_prefix = multi_processing.get_file_name(path).split('.')[0]
        if directed:
            graph = nx.read_weighted_edgelist(path, create_using=nx.DiGraph(), nodetype=int)
        else:
            graph = nx.read_weighted_edgelist(path, create_using=nx.Graph(), nodetype=int)
        nx.write_gpickle(graph, output_folder + name_prefix + '.gpickle')
        return cls(graph, name_prefix, directed)

    def draw_graph(self):
        """
        Takes too much time with big data.
        """
        nx.draw(self.graph, with_labels=True)
        plt.show()

    def print_graph_information(self):
        print('\n###################### Graph Information ######################')
        number_of_edges = self.graph.number_of_edges()
        number_of_selfloops = nx.number_of_selfloops(self.graph)
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
        # TODO LATER: Code below takes long time to calculate for big graphs.
        # print('Average shortest path length (weight=None):', str(round(nx.average_shortest_path_length(self.graph), 2)))
        # TODO LATER: average_clustering has not implemented for undirected graph yet.
        if not nx.is_directed(self.graph):
            # For unweighted graphs, the clustering of a node
            # is the fraction of possible triangles through that node that exist
            print('The clustering coefficient for the graph is ' + str(
                round(nx.average_clustering(self.graph, weight=None), 2)))
        print('###############################################################\n')

    def get_shortest_path_lengths_between_all_nodes(self, output_folder=config['graph']['graph_folder']):
        """
        From test, these three algorithms below take more than 20 hours (processes have been killed after 20 hours) to
        calculate.
        'floyd_warshall_numpy' takes around 100 minutes to get the result.

        # length1 = dict(nx.all_pairs_dijkstra_path_length(g))
        # length2 = dict(nx.all_pairs_bellman_ford_path_length(g))
        # length3 = nx.johnson(g, weight='weight')
        # for node in [0, 1, 2, 3, 4]:
        #     print('1 - {}: {}'.format(node, length2[1][node]))
        """
        """ ATTENTION
        'floyd_warshall_numpy' has already considered situations below:
        1. If there's no path between source and target node, matrix will put 'inf'
        2. No matter how much the weight is between node and node itself(self loop), the shortest path will always be 0.
        """
        matrix = nx.floyd_warshall_numpy(self.graph)  # ATTENTION: return type is NumPy matrix not NumPy ndarray.
        # ATTENTION: after saving, NumPy matrix has been changed to 2darray.
        np.save(output_folder + self.name_prefix + '_matrix.npy', matrix, fix_imports=False)
        common.write_to_pickle(self.graph.nodes(), output_folder + self.name_prefix + '_nodes.pickle')
        return self.graph.nodes(), matrix

    def get_t_step_random_walk_stochastic_matrix(self, t):
        def get_stochastic_matrix():
            self.graph.remove_edges_from(list(nx.selfloop_edges(self.graph)))  # remove self loop
            if self.directed:
                directed_graph = self.graph
            else:
                directed_graph = self.graph.to_directed()
            # this function only works with directed graph
            stochastic_graph = nx.stochastic_graph(directed_graph, weight='weight')
            return nx.to_numpy_matrix(stochastic_graph)

        transition_matrix = get_stochastic_matrix()
        result = transition_matrix
        while t > 1:
            result = np.matmul(result, transition_matrix)
            t -= 1
        return self.graph.nodes(), result


# matrix = np.load(data_folder + self.name_prefix + '_matrix.npy')
# nodes = common.read_pickle(data_folder + self.name_prefix + '_nodes.pickle')


class NegativeSamples:
    def __init__(self, matrix, row_column_indices_value, merged_dict_path):
        self.matrix = np.asarray(matrix)  # ATTENTION: change NumPy matrix type to NumPy ndarray.
        self.row_column_indices_value = row_column_indices_value
        self.merged_dict_path = merged_dict_path
        self.translated_negative_samples_dict = None

    def print_matrix_and_token_order(self):
        index2word = gdp.get_index2word(file=self.merged_dict_path)
        print('\n******************* Matrix & tokens order *******************')
        print([index2word[index] for index in self.row_column_indices_value])
        print(self.matrix)
        print('*************************************************************\n')

    def get_matrix_value_by_token_xy(self, token_x, token_y):
        # Does not need translated ns dict to be calculated.
        word2index = gdp.read_two_columns_file_to_build_dictionary_type_specified(
            file=self.merged_dict_path, key_type=str, value_type=int)
        nodes = list(self.row_column_indices_value)
        matrix_x = nodes.index(word2index[token_x])
        matrix_y = nodes.index(word2index[token_y])
        return self.matrix[matrix_x][matrix_y]

    def __get_negative_samples_dict_from_matrix(self, n, selected_mode):
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
        nodes_list = list(self.row_column_indices_value)
        if selected_mode == 'min':
            selected_indices = np.argpartition(self.matrix, n)[:, :n]
        elif selected_mode == 'max':
            selected_indices = np.argpartition(self.matrix, -n)[:, -n:]
        # indices here means the indices of nodes list, not the value(word index) inside nodes list.
        cleaned_selected_indices = np.empty([selected_indices.shape[0], n - 1], dtype=int)
        negative_samples_dict = {}
        for i in range(self.matrix.shape[1]):  # shape[0] = shape[1]
            # e.g. for the first row (i=0), find the index in selected_indices where the value equals 0 (self loop)
            self_loop_index = np.where(selected_indices[i] == i)
            if self_loop_index[0].size == 0:  # no self loop
                shortest_path = self.matrix[i][selected_indices[i]]
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
            negative_samples_dict[nodes_list[i]] = np.array(self.row_column_indices_value)[
                cleaned_selected_indices[i]].tolist()
        # common.write_to_pickle(shortest_path_nodes_dict, data_folder+'shortest_path_nodes_dict.pickle')
        return negative_samples_dict

    def write_translated_negative_samples_dict(self, n, selected_mode, output_folder):
        index2word = gdp.get_index2word(file=self.merged_dict_path)
        translated_negative_samples_dict = {}
        for key, value in self.__get_negative_samples_dict_from_matrix(n, selected_mode).items():
            translated_negative_samples_dict[index2word[key]] = [index2word[node_id] for node_id in value]
        # TODO NOW file name should be unique, not always the same.
        common.write_to_pickle(translated_negative_samples_dict,
                               output_folder + 'translated_negative_samples_dict.pickle')
        self.translated_negative_samples_dict = translated_negative_samples_dict
        return translated_negative_samples_dict

    def load_translated_negative_samples_dict(self, path):
        self.translated_negative_samples_dict = common.read_pickle(path)

    def print_tokens_negative_samples_and_their_value_in_matrix(self, tokens_list):
        # TODO use get_matrix_value_by_token_xy function.
        if not self.translated_negative_samples_dict:
            sys.exit('translated_negative_samples_dict not found.')
        word2index = gdp.read_two_columns_file_to_build_dictionary_type_specified(
            file=self.merged_dict_path, key_type=str, value_type=int)
        nodes = list(self.row_column_indices_value)
        for word in tokens_list:
            print('For word:', word)
            word_index = word2index[word]
            matrix_x = nodes.index(word_index)
            ns_words = self.translated_negative_samples_dict[word]
            for ns_word in ns_words:
                ns_word_index = word2index[ns_word]
                matrix_y = nodes.index(ns_word_index)
                matrix_cell_value = self.matrix[matrix_x][matrix_y]
                print(' ', ns_word, matrix_cell_value)


if __name__ == '__main__':
    g = NXGraph(config['graph']['graph_folder'] + 'encoded_edges_count_window_size_5.txt')
    g.print_graph_information()

    # graph.get_shortest_path_lengths_between_all_nodes(output_folder=config['graph']['graph_folder'])
    # translated_shortest_path_nodes_dict = NXGraph.write_translated_negative_samples_dict(
    #     NXGraph.get_negative_samples_dict_from_matrix(20, selected_mode='max', data_folder=config['graph']['graph_folder']),
    #     config['graph']['dicts_and_encoded_texts_folder']+'dict_merged.txt',
    #     output_folder=config['graph']['graph_folder'])
