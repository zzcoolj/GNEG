__author__ = 'Zheng ZHANG'

"""
graph_builder is used by negative_samples_generator.py to get what is needed to build the negative samples.
"""

import numpy as np
import networkx as nx

import matplotlib
matplotlib.use('agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import configparser
import graph_data_provider as gdp
import sys
sys.path.insert(0, '../common/')
import common
import multi_processing

config = configparser.ConfigParser()
config.read('config.ini')


class NoGraph:
    def __init__(self, encoded_edges_count_file_path, valid_vocabulary_path):
        """
        Theoretically valid_vocabulary file is not necessary. We could build a graph_index2wordId dict by going through
        encoded_edges_count_file_path and getting all wordIds. But it's not efficient.

        valid_wordId's order is really important and should be static, because:
            1. graph_index2wordId is built on this
            2. graph_wordId2index (temp var), on which cooccurrence_matrix element's order is based, is built on this.
            3. graph_index2wordId represents cooccurrence_matrix element's order
        """
        self.name_prefix = multi_processing.get_file_name(encoded_edges_count_file_path).split('.')[0]
        valid_wordId = list(set(gdp.read_valid_vocabulary(valid_vocabulary_path)))  # make sure no duplication
        # ATTENTION: graph_index2wordId should be a list of which the index order is from 0 to vocab_size-1
        # TODO LATER No need to make graph_index2wordId an int list. Find where graph_index2wordId is needed and changed them.
        self.graph_index2wordId = list(map(int, valid_wordId))
        vocab_size = len(valid_wordId)
        # ATTENTION: the index is of the type int, while the wordId is of the type str
        graph_wordId2index = dict(zip(valid_wordId, range(vocab_size)))
        # initialize numpy 2d array
        cooccurrence_matrix = np.zeros((vocab_size, vocab_size))
        # read encoded_edges_count_file
        for line in common.read_file_line_yielder(encoded_edges_count_file_path):
            # ATTENTION: line e.g. '17'  '57'  '10' or '57'   '17'  '10' (only one of them will appear in the file.)
            (source, target, weight) = line.split("\t")
            cooccurrence_matrix[graph_wordId2index[source]][graph_wordId2index[target]] = weight
            # undirected graph
            cooccurrence_matrix[graph_wordId2index[target]][graph_wordId2index[source]] = weight
        self.cooccurrence_matrix = cooccurrence_matrix

    def get_stochastic_matrix(self, power=None):
        """
        A replacement of get_stochastic_matrix function NXGraph class.
        """
        vocab_size = self.cooccurrence_matrix.shape[0]
        stochastic_matrix = self.cooccurrence_matrix.copy()
        # power co-occurrence if needed.
        if power:
            stochastic_matrix = np.power(stochastic_matrix, power)
        # remove self loop
        for i in range(vocab_size):
            stochastic_matrix[i][i] = 0
        # calculate percentage
        matrix_sum_row = np.sum(stochastic_matrix, axis=1, keepdims=True)  # sum of each row and preserve the dimension
        stochastic_matrix /= matrix_sum_row
        return stochastic_matrix

    def one_to_t_step_random_walk_stochastic_matrix_yielder(self, t):
        """
        Instead of getting a specific t step random walk result, this method gets a dict of result from 1 step random
        walk to t step random walk. This method should be used for grid search.
        """
        transition_matrix = self.get_stochastic_matrix()
        result = transition_matrix
        for t in range(1, t+1):
            if t != 1:
                result = np.matmul(result, transition_matrix)
            yield result, t

    def get_t_step_random_walk_stochastic_matrix(self, t, output_folder=None):
        # TODO NOW not the same result from 1 step random walk
        transition_matrix = self.get_stochastic_matrix()
        result = transition_matrix
        while t > 1:
            result = np.matmul(result, transition_matrix)
            t -= 1
        if output_folder:
            file_prefix = output_folder + self.name_prefix + '_' + str(t)
            np.save(file_prefix + '_step_rw_matrix.npy', result, fix_imports=False)
            common.write_to_pickle(self.graph_index2wordId, file_prefix + '_step_rw_nodes.pickle')
        return self.graph_index2wordId, result

    def reorder_matrix(self, matrix, word_count_path):
        """
        Works for cooccurrence_matrix, stochastic_matrix and random walk matrix. They share the same wordId order.
        :param word_count_path: wordId -> count
        :return: reordered matrix (following word count in a descending order), new graph_index2wordId
        """
        word_count = gdp.read_two_columns_file_to_build_dictionary_type_specified(word_count_path, key_type=int,
                                                                                  value_type=int)
        wordId2count = {}
        for valid_wordId in self.graph_index2wordId:  # a list of valid vocabulary wordIds => old wordId order
            wordId2count[valid_wordId] = word_count[valid_wordId]
        new_wordId_order = list(sorted(wordId2count, key=wordId2count.get, reverse=True))
        new_index_order = [self.graph_index2wordId.index(wordId) for wordId in new_wordId_order]
        # reorder rows
        reordered_matrix = matrix[new_index_order, :]
        # reorder columns
        reordered_matrix = reordered_matrix[:, new_index_order]
        return new_wordId_order, reordered_matrix


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
        # nx.write_gpickle(graph, output_folder + name_prefix + '.gpickle')
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

    def log_edges_count(self):
        # TODO NOW maybe could be repalced in make_cum_matrix part
        pass

    def get_shortest_path_lengths_between_all_nodes(self, output_folder):
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

    def __get_stochastic_matrix(self):
        # ATTENTION: for a big graph, this method consumes too much memory and calculation time.
        self.graph.remove_edges_from(list(nx.selfloop_edges(self.graph)))  # remove self loop
        if self.directed:
            directed_graph = self.graph
        else:
            directed_graph = self.graph.to_directed()
        # this function only works with directed graph
        stochastic_graph = nx.stochastic_graph(directed_graph, weight='weight')
        return nx.to_numpy_matrix(stochastic_graph)

    def get_t_step_random_walk_stochastic_matrix(self, t, output_folder=None):
        # TODO LATER not the same result from 1 step random walk
        transition_matrix = self.__get_stochastic_matrix()
        result = transition_matrix
        while t > 1:
            result = np.matmul(result, transition_matrix)
            t -= 1
        if output_folder:
            file_prefix = output_folder + self.name_prefix + '_' + str(t)
            np.save(file_prefix + '_step_rw_matrix.npy', result, fix_imports=False)
            common.write_to_pickle(self.graph.nodes(), file_prefix + '_step_rw_nodes.pickle')
        return self.graph.nodes(), result

    def one_to_t_step_random_walk_stochastic_matrix_yielder(self, t):
        """
        Instead of getting a specific t step random walk result, this method gets a dict of result from 1 step random
        walk to t step random walk. This method should be used for grid search.
        """
        transition_matrix = self.__get_stochastic_matrix()
        result = transition_matrix
        for t in range(1, t+1):
            if t != 1:
                result = np.matmul(result, transition_matrix)
            yield result, t


if __name__ == '__main__':
    ng = NoGraph(encoded_edges_count_file_path=config['graph']['graph_folder'] + 'encoded_edges_count_window_size_3_undirected.txt',
                 valid_vocabulary_path=config['graph']['dicts_and_encoded_texts_folder'] + 'valid_vocabulary_min_count_5_vocab_size_10000.txt')
    import negative_samples_generator as nsg
    output_folder = config['graph']['graph_folder'] + 'png/'
    word_count_path = config['graph']['dicts_and_encoded_texts_folder'] + 'word_count_all.txt'
    cooc = ng.cooccurrence_matrix
    _, reorder_cooc = ng.reorder_matrix(cooc, word_count_path)
    nsg.NegativeSamples.heatmap_bis(reorder_cooc, output_folder=output_folder, name='reorder_cooc.png')
    print('saved1')
    stoc = ng.get_stochastic_matrix()
    _, reorder_stoc = ng.reorder_matrix(stoc, word_count_path)
    nsg.NegativeSamples.heatmap_bis(reorder_stoc, output_folder=output_folder, name='reorder_stochastic.png')
    print('saved2')
