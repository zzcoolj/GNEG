__author__ = 'Zheng ZHANG'

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import configparser
from multiprocessing import Pool
from itertools import repeat
import sys
sys.path.insert(0, '../common/')
import common
import multi_processing
import graph_data_provider as gdp

config = configparser.ConfigParser()
config.read('config.ini')


class NoGraph:
    def __init__(self, encoded_edges_count_file_path, vocab_size):
        self.name_prefix = multi_processing.get_file_name(encoded_edges_count_file_path).split('.')[0]
        # initialize numpy 2d array
        cooccurrence_matrix = np.zeros((vocab_size, vocab_size))
        # initialize graph_index2wordId
        graph_index2wordId = []
        # read encoded_edges_count_file
        for line in common.read_file_line_yielder(encoded_edges_count_file_path):
            # ATTENTION: line e.g. 17  57  10 or 57   17  10 (only one of them will appear in the file.)
            (source, target, weight) = line.split("\t")
            source = int(source)
            target = int(target)
            if source in graph_index2wordId:
                source_index = graph_index2wordId.index(source)
            else:
                source_index = len(graph_index2wordId)
                graph_index2wordId.append(source)

            if target in graph_index2wordId:
                target_index = graph_index2wordId.index(target)
            else:
                target_index = len(graph_index2wordId)
                graph_index2wordId.append(target)
            cooccurrence_matrix[source_index][target_index] = weight
            # ATTENTION: undirected graph
            cooccurrence_matrix[target_index][source_index] = weight
        self.graph_index2wordId = graph_index2wordId
        self.cooccurrence_matrix = cooccurrence_matrix

    def get_stochastic_matrix(self):
        """
        A replacement of get_stochastic_matrix function NXGraph class.
        """
        vocab_size = self.cooccurrence_matrix.shape[0]
        stochastic_matrix = self.cooccurrence_matrix.copy()
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
        self.graph.remove_edges_from(list(nx.selfloop_edges(self.graph)))  # remove self loop
        if self.directed:
            directed_graph = self.graph
        else:
            directed_graph = self.graph.to_directed()
        # this function only works with directed graph
        stochastic_graph = nx.stochastic_graph(directed_graph, weight='weight')
        return nx.to_numpy_matrix(stochastic_graph)

    def get_t_step_random_walk_stochastic_matrix(self, t, output_folder=None):
        # TODO NOW not the same result from 1 step random walk
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


class NegativeSamples:
    def __init__(self, matrix, graph_index2wordId, merged_dict_path, name_prefix):
        self.name_prefix = name_prefix
        self.matrix = np.asarray(matrix)  # ATTENTION: change NumPy matrix type to NumPy ndarray.
        self.graph_index2wordId = graph_index2wordId
        self.merged_dict_path = merged_dict_path
        self.translated_negative_samples_dict = None

    @classmethod
    def load(cls, matrix_path, graph_index2wordId_path, merged_dict_path):
        matrix = np.load(matrix_path)
        graph_index2wordId = common.read_pickle(graph_index2wordId_path)
        return cls(matrix=matrix, graph_index2wordId=graph_index2wordId, merged_dict_path=merged_dict_path,
                   name_prefix=None)

    def get_and_print_matrix_and_token_order(self):
        wordId2word = gdp.get_index2word(file=self.merged_dict_path)
        print('\n******************* Matrix & tokens order *******************')
        token_order = [wordId2word[wordId] for wordId in self.graph_index2wordId]
        print(token_order)
        print(self.matrix)
        print('*************************************************************\n')
        return self.matrix, token_order

    def get_matrix_value_by_token_xy(self, token_x, token_y):
        # Does not need translated ns dict to be calculated.
        word2index = gdp.read_two_columns_file_to_build_dictionary_type_specified(
            file=self.merged_dict_path, key_type=str, value_type=int)
        nodes = list(self.graph_index2wordId)
        matrix_x = nodes.index(word2index[token_x])
        matrix_y = nodes.index(word2index[token_y])
        return self.matrix[matrix_x][matrix_y]

    def reorder_matrix(self, word2vec_index2word):
        """
        Called in word2vec_gensim_modified.py make_cum_matrix function.
        :param word2vec_index2word: self.wv.index2word in word2vec_gensim_modified.py,
                                    different from graph_index2wordId order.
        :return: a reordered matrix following the order of word2vec_index2word
        """
        graph_wordId2word = gdp.get_index2word(file=self.merged_dict_path)
        ''' 
            e.g. graph_index2wordId = [7, 91, 20, ...] means:
                the 1st row/column represents token of id (graph_wordId2word) 7
                the 2nd row/column represents token of id (graph_wordId2word) 91
        '''
        translated_matrix_order = [graph_wordId2word[wordId] for wordId in self.graph_index2wordId]
        '''
        new matrix row/column index order is always [0, 1, ..., 9999] (if vocab_size is 10000), but here index is not
        based on the graph_wordId2word. It's based on the word2vec_index2word (self.wv.index2word)
        '''
        reordered_matrix_length = self.matrix.shape[0]
        translated_reordered_matrix_order = [word2vec_index2word[index] for index in range(reordered_matrix_length)]
        '''e.g.
        translated_matrix_order: [windows, apple, ibm, tesla]
        translated_reordered_matrix_order: [apple, tesla, ibm, windows] (what I want)
        new_index_order = [1, 3, 2, 0]
        1 means translated_matrix_order index 1 element apple is the first element in translated_reordered_matrix_order
        3 means translated_matrix_order index 3 element tesla is the second element in translated_reordered_matrix_order
        '''
        new_index_order = [translated_matrix_order.index(token) for token in translated_reordered_matrix_order]
        # reorder rows
        reordered_matrix = self.matrix[new_index_order, :]
        # reorder columns
        reordered_matrix = reordered_matrix[:, new_index_order]
        return reordered_matrix

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
        nodes_list = list(self.graph_index2wordId)
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
            negative_samples_dict[nodes_list[i]] = np.array(self.graph_index2wordId)[
                cleaned_selected_indices[i]].tolist()
        # common.write_to_pickle(shortest_path_nodes_dict, data_folder+'shortest_path_nodes_dict.pickle')
        return negative_samples_dict

    def write_translated_negative_samples_dict(self, n, selected_mode, output_folder, name_suffix=None):
        """ATTENTION [DEPRECATED]
        This function has only been used by deprecated FromEncodedEdgesCountToTranslatedNSDict class
        """
        index2word = gdp.get_index2word(file=self.merged_dict_path)
        translated_negative_samples_dict = {}
        for key, value in self.__get_negative_samples_dict_from_matrix(n, selected_mode).items():
            translated_negative_samples_dict[index2word[key]] = [index2word[node_id] for node_id in value]
        if not name_suffix:
            name_suffix=''
        common.write_to_pickle(translated_negative_samples_dict,
                               output_folder + self.name_prefix + '_ns' + name_suffix + '.pickle')
        self.translated_negative_samples_dict = translated_negative_samples_dict
        return translated_negative_samples_dict

    def load_translated_negative_samples_dict(self, path):
        self.translated_negative_samples_dict = common.read_pickle(path)

    def print_tokens_negative_samples_and_their_value_in_matrix(self, tokens_list):
        if not self.translated_negative_samples_dict:
            sys.exit('translated_negative_samples_dict not found.')
        for word in tokens_list:
            print('==>', word)
            ns_words = self.translated_negative_samples_dict[word]
            for ns_word in ns_words:
                print('\t', ns_word, '\t', self.get_matrix_value_by_token_xy(token_x=word, token_y=ns_word))


class FromEncodedEdgesCountToTranslatedNSDict:
    """ATTENTION [DEPRECATED]
    This class only serves the old uniform ns selection idea, which should be deprecated.
    """
    def __init__(self, encoded_edges_count_file_folder, ns_folder, merged_dict_path):
        self.encoded_edges_count_file_folder = encoded_edges_count_file_folder  # input folder
        self.ns_folder = ns_folder  # output folder
        self.merged_dict_path = merged_dict_path

    def one_to_one_rw(self, encoded_edges_count_file_path, directed, t, potential_ns_len, selected_mode):
        graph = NXGraph.from_encoded_edges_count_file(encoded_edges_count_file_path, directed=directed)
        nodes, matrix = graph.get_t_step_random_walk_stochastic_matrix(t=t)
        ns = NegativeSamples(matrix=matrix, graph_index2wordId=nodes,
                             merged_dict_path=self.merged_dict_path,
                             name_prefix=graph.name_prefix)
        ns.write_translated_negative_samples_dict(n=potential_ns_len, selected_mode=selected_mode,
                                                  output_folder=self.ns_folder,
                                                  name_suffix='_'+str(t)+'_'+selected_mode)

    def one_to_many_rw(self, encoded_edges_count_file_path, directed, potential_ns_len, t_max):
        """
        For one encoded_edges_count_file, get ns dict by different combinations of parameters:
            t & selected_mode
        """
        print(multi_processing.get_pid(), encoded_edges_count_file_path)
        graph = NXGraph.from_encoded_edges_count_file(encoded_edges_count_file_path, directed=directed)
        nodes = graph.graph.nodes()
        for matrix, t in graph.one_to_t_step_random_walk_stochastic_matrix_yielder(t=t_max):
            ns = NegativeSamples(matrix=matrix, graph_index2wordId=nodes,
                                 merged_dict_path=self.merged_dict_path,
                                 name_prefix=graph.name_prefix)
            for selected_mode in ['max', 'min']:
                ns.write_translated_negative_samples_dict(n=potential_ns_len, selected_mode=selected_mode,
                                                          output_folder=self.ns_folder,
                                                          name_suffix='_' + str(t) + '_' + selected_mode)

    def many_to_many_rw(self, directed, potential_ns_len, t_max, process_num):
        """
        For all encoded_edges_count_file (of different window size)
        """
        kw = {'directed': directed, 'potential_ns_len': potential_ns_len, 't_max': t_max}
        multi_processing.master(files_getter=multi_processing.get_files_endswith,
                                data_folder=self.encoded_edges_count_file_folder,
                                file_extension='undirected.txt',
                                worker=self.one_to_many_rw,
                                process_num=process_num,
                                **kw)


class GraphGridSearcher:
    """
    This class is just a helper for NXGraph. When there are several encoded_edges_count files. This class goes through
    all of them and produce result of different t steps.
    """
    def __init__(self, ns_folder):
        self.ns_folder = ns_folder  # output folder

    def one_to_one(self, encoded_edges_count_file_path, directed, t):
        # # NXGraph version: too slow
        # graph = NXGraph.from_encoded_edges_count_file(encoded_edges_count_file_path, directed=directed)
        # graph.get_t_step_random_walk_stochastic_matrix(t=t, output_folder=self.ns_folder)

        no_graph = NoGraph(encoded_edges_count_file_path)
        no_graph.get_t_step_random_walk_stochastic_matrix(t=t, output_folder=self.ns_folder)

    def one_to_many(self, encoded_edges_count_file_path, directed, t_max):
        print(multi_processing.get_pid(), encoded_edges_count_file_path)

        """ATTENTION
        For encoded_edges_count_file generated by whole wiki data, this function takes around
        The bigger window_size is, the bigger the encoded_edges_file is and the more memory taken.
        """
        # graph = NXGraph.from_encoded_edges_count_file(encoded_edges_count_file_path, directed=directed)
        # # They share the same nodes file
        # nodes = graph.graph.nodes()
        # common.write_to_pickle(nodes, self.ns_folder + graph.name_prefix + '_nodes.pickle')
        # for matrix, t in graph.one_to_t_step_random_walk_stochastic_matrix_yielder(t=t_max):

        no_graph = NoGraph(encoded_edges_count_file_path)
        common.write_to_pickle(no_graph.graph_index2wordId, self.ns_folder + no_graph.name_prefix + '_nodes.pickle')
        for matrix, t in no_graph.one_to_t_step_random_walk_stochastic_matrix_yielder(t=t_max):
            file_prefix = self.ns_folder + no_graph.name_prefix + '_' + str(t)
            print('write matrix', file_prefix)
            np.save(file_prefix + '_step_rw_matrix.npy', matrix, fix_imports=False)
        print('need memory clean')
        return None

    def many_to_many(self, encoded_edges_count_file_folder, directed, t_max, process_num):
        """
        For all encoded_edges_count_file (of different window size)

        t_max does not influence memory usage

        ATTENTION: for real server test, set process_num to 3.
        """
        # kw = {'directed': directed, 't_max': t_max}
        if directed:
            # TODO LATER: So far, all directed encoded_edges_count files don't have such file extension below.
            file_extension = '_directed.txt'
        else:
            file_extension = '_undirected.txt'

        files_list = multi_processing.get_files_endswith(encoded_edges_count_file_folder, file_extension)
        p = Pool(process_num, maxtasksperchild=1)
        p.starmap_async(self.one_to_many, zip(files_list, repeat(directed), repeat(t_max)))
        p.close()
        p.join()


if __name__ == '__main__':
    # # DEPRECATED
    # bridge = FromEncodedEdgesCountToTranslatedNSDict(encoded_edges_count_file_folder=config['graph']['graph_folder'],
    #                                                  ns_folder=config['word2vec']['negative_samples_folder'],
    #                                                  merged_dict_path=config['graph']['dicts_and_encoded_texts_folder'] + 'dict_merged.txt')
    # bridge.one_to_one_rw(encoded_edges_count_file_path=bridge.encoded_edges_count_file_folder+'encoded_edges_count_window_size_5_undirected.txt',
    #                      directed=False, t=1, negative=20, selected_mode='min')
    #
    # bridge.one_to_many_rw(encoded_edges_count_file_path=bridge.encoded_edges_count_file_folder+'encoded_edges_count_window_size_5_undirected.txt',
    #                       directed=False, t_max=1, negative=20)
    # bridge.many_to_many_rw(directed=False, t_max=2, potential_ns_len=1000, process_num=2)

    grid_searcher = GraphGridSearcher(ns_folder=config['word2vec']['negative_samples_folder'])
    grid_searcher.many_to_many(encoded_edges_count_file_folder=config['graph']['graph_folder'], directed=False, t_max=5,
                               process_num=3)
