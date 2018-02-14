__author__ = 'Zheng ZHANG'

import time
import re
import numpy as np
import configparser
from multiprocessing import Pool
from itertools import repeat

import matplotlib
matplotlib.use('agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

from graph_builder import NoGraph, NXGraph
import sys
sys.path.insert(0, '../common/')
import common
import multi_processing
import graph_data_provider as gdp

config = configparser.ConfigParser()
config.read('config.ini')


class NegativeSamples:
    """
    NegativeSamples class is for word2vec_gensim_modified.py
    """
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
        word2wordId = gdp.read_two_columns_file_to_build_dictionary_type_specified(
            file=self.merged_dict_path, key_type=str, value_type=int)
        nodes = list(self.graph_index2wordId)
        matrix_x = nodes.index(word2wordId[token_x])
        matrix_y = nodes.index(word2wordId[token_y])
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
        This function has only been used by deprecated NegativeSamplesGenerator_old class
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

    def reorder_matrix_by_word_count(self, word_count_path):
        """ For visualization
        Works for cooccurrence_matrix, stochastic_matrix and random walk matrix. They share the same wordId order.
        :param word_count_path: wordId -> count
        :return: reordered matrix (following word count in a descending order), new graph_index2wordId
        """
        word_count = gdp.read_two_columns_file_to_build_dictionary_type_specified(word_count_path, key_type=int,
                                                                                  value_type=int)
        self.graph_index2wordId = list(self.graph_index2wordId)
        wordId2count = {}
        for valid_wordId in self.graph_index2wordId:  # a list of valid vocabulary wordIds => old wordId order
            wordId2count[valid_wordId] = word_count[valid_wordId]
        # TODO NOW get new_wordId_order list directly.
        new_wordId_order = list(sorted(wordId2count, key=wordId2count.get, reverse=True))
        new_index_order = [self.graph_index2wordId.index(wordId) for wordId in new_wordId_order]
        # reorder rows
        reordered_matrix = self.matrix[new_index_order, :]
        # reorder columns
        reordered_matrix = reordered_matrix[:, new_index_order]
        return new_wordId_order, reordered_matrix

    @staticmethod
    def get_valid_vocab_count_list(word_count_path, valid_vocabulary_path):
        count_list = []
        merged_word_count = gdp.read_two_columns_file_to_build_dictionary_type_specified(word_count_path,
                                                                                         key_type=str, value_type=int)
        valid_vocabulary = dict.fromkeys(gdp.read_valid_vocabulary(valid_vocabulary_path))
        for index in valid_vocabulary:
            count_list.append(merged_word_count[str(index)])
        return count_list


class NegativeSamplesGenerator:
    """
    This class is just a helper for NXGraph. When there are several encoded_edges_count files. This class goes through
    all of them and produce result of different t steps.
    """
    def __init__(self, ns_folder, valid_vocabulary_path):
        self.ns_folder = ns_folder  # output folder
        self.valid_vocabulary_path = valid_vocabulary_path

    # for t-step random walk
    def one_to_one(self, encoded_edges_count_file_path, t):
        # # [DEPRECATED] NXGraph version: too slow, it takes around 50 mins for window size 10 for whole wiki data.
        # graph = NXGraph.from_encoded_edges_count_file(encoded_edges_count_file_path, directed=directed)
        # graph.get_t_step_random_walk_stochastic_matrix(t=t, output_folder=self.ns_folder)

        # fast, takes less than 3 minutes for whole wiki data when window_size=10
        no_graph = NoGraph(encoded_edges_count_file_path, valid_vocabulary_path=self.valid_vocabulary_path)
        no_graph.get_t_step_random_walk_stochastic_matrix(t=t, output_folder=self.ns_folder)

    def one_to_many(self, encoded_edges_count_file_path, t_max):
        print(multi_processing.get_pid(), encoded_edges_count_file_path)

        # # [DEPRECATED] NXGraph version: too slow.
        # # The bigger window_size is, the bigger the encoded_edges_file is and the more memory taken.
        # graph = NXGraph.from_encoded_edges_count_file(encoded_edges_count_file_path, directed=directed)
        # # They share the same nodes file
        # nodes = graph.graph.nodes()
        # common.write_to_pickle(nodes, self.ns_folder + graph.name_prefix + '_nodes.pickle')
        # for matrix, t in graph.one_to_t_step_random_walk_stochastic_matrix_yielder(t=t_max):

        print('1')
        no_graph = NoGraph(encoded_edges_count_file_path, valid_vocabulary_path=self.valid_vocabulary_path)
        print('1.5')
        common.write_to_pickle(no_graph.graph_index2wordId, self.ns_folder + no_graph.name_prefix + '_nodes.pickle')
        print('2')
        for matrix, t in no_graph.one_to_t_step_random_walk_stochastic_matrix_yielder(t=t_max):
            print('3')
            file_prefix = self.ns_folder + no_graph.name_prefix + '_' + str(t)
            print('write matrix', file_prefix)
            np.save(file_prefix + '_step_rw_matrix.npy', matrix, fix_imports=False)
            print('4')
        print('need memory clean')
        return None

    def many_to_many(self, encoded_edges_count_file_folder, directed, t_max, process_num, partial=False):
        """
        For all encoded_edges_count_file (of different window size)
        There are four types of files/extension:
            encoded_edges_count_window_size_3.txt
            encoded_edges_count_window_size_3_undirected.txt
            encoded_edges_count_window_size_3_partial.txt
            encoded_edges_count_window_size_3_undirected_partial.txt

        t_max does not influence memory usage

        ATTENTION: for real server test, set process_num to 3.
        """
        if directed:
            # TODO LATER: So far, all directed encoded_edges_count files don't have such file extension below.
            file_extension = '_directed.txt'
        else:
            if partial:
                file_extension = '_undirected_partial.txt'
            else:
                file_extension = '_undirected.txt'

        files_list = multi_processing.get_files_endswith(encoded_edges_count_file_folder, file_extension)
        p = Pool(process_num, maxtasksperchild=1)
        p.starmap_async(self.one_to_many, zip(files_list, repeat(t_max)))
        p.close()
        p.join()

    # for stochastic matrix
    def get_stochastic_matrix(self, encoded_edges_count_file_path):
        print(multi_processing.get_pid(), encoded_edges_count_file_path)
        no_graph = NoGraph(encoded_edges_count_file_path, valid_vocabulary_path=self.valid_vocabulary_path)
        common.write_to_pickle(no_graph.graph_index2wordId, self.ns_folder + no_graph.name_prefix + '_nodes.pickle')
        file_prefix = self.ns_folder + no_graph.name_prefix

        stochastic_matrix = no_graph.get_stochastic_matrix(change_zeros_to_minimum_positive_value=False)
        print('write matrix zeros ', file_prefix)
        np.save(file_prefix + '_zeros_matrix.npy', stochastic_matrix, fix_imports=False)

        stochastic_matrix = no_graph.get_stochastic_matrix(change_zeros_to_minimum_positive_value=True)
        print('write matrix no zeros ', file_prefix)
        np.save(file_prefix + '_noZeros_matrix.npy', stochastic_matrix, fix_imports=False)

        print('need memory clean')
        return None

    @staticmethod
    def multi_functions(f, encoded_edges_count_file_folder, directed, process_num, partial=False):
        if directed:
            # TODO LATER: So far, all directed encoded_edges_count files don't have such file extension below.
            file_extension = '_directed.txt'
        else:
            if partial:
                file_extension = '_undirected_partial.txt'
            else:
                file_extension = '_undirected.txt'

        files_list = multi_processing.get_files_endswith(encoded_edges_count_file_folder, file_extension)
        p = Pool(process_num, maxtasksperchild=1)
        p.starmap_async(f, zip(files_list))
        p.close()
        p.join()

    # difference between unigram and stochastic matrix
    def get_difference_matrix(self, encoded_edges_count_file_path, merged_word_count_path):
        print(multi_processing.get_pid(), encoded_edges_count_file_path)
        no_graph = NoGraph(encoded_edges_count_file_path, valid_vocabulary_path=self.valid_vocabulary_path)
        common.write_to_pickle(no_graph.graph_index2wordId, self.ns_folder + no_graph.name_prefix + '_nodes.pickle')
        file_prefix = self.ns_folder + no_graph.name_prefix
        difference_matrix = no_graph.get_difference_matrix(merged_word_count_path=merged_word_count_path)
        print('write matrix zeros ', file_prefix)
        np.save(file_prefix + '_matrix.npy', difference_matrix, fix_imports=False)
        print('need memory clean')
        return None

    def multi_difference_matrix(self, encoded_edges_count_file_folder, merged_word_count_path, directed, process_num,
                                partial=False):
        if directed:
            # TODO LATER: So far, all directed encoded_edges_count files don't have such file extension below.
            file_extension = '_directed.txt'
        else:
            if partial:
                file_extension = '_undirected_partial.txt'
            else:
                file_extension = '_undirected.txt'

        files_list = multi_processing.get_files_endswith(encoded_edges_count_file_folder, file_extension)
        p = Pool(process_num, maxtasksperchild=1)
        p.starmap_async(self.get_difference_matrix, zip(files_list, repeat(merged_word_count_path)))
        p.close()
        p.join()


class Visualization:
    def __init__(self):
        pass

    @staticmethod
    def matrix_vis(matrix, output_path):
        # NoGraph matrix is initialized with all zeros. So here we don't consider empty cell case, only zero cell case.
        # find zero position in the matrix
        zero_indices_x, zero_indices_y = np.where(matrix == 0)
        # find the second minimum value in matrix, temp_matrix is used for that
        max_value = np.amax(matrix)
        temp_matrix = np.copy(matrix)
        for i in range(len(zero_indices_x)):
            temp_matrix[zero_indices_x[i]][zero_indices_y[i]] = max_value
        second_minimum = np.amin(temp_matrix)  # first minimum is always 0
        # set all zeros to second minimum value
        for i in range(len(zero_indices_x)):
            matrix[zero_indices_x[i]][zero_indices_y[i]] = second_minimum

        matrix = np.log10(matrix)  # Necessary for negative samples matrix, nearly all black if not.
        plt.imshow(matrix, cmap="nipy_spectral")  # plt.cm.BuPu_r, hot -> bad choices (no big difference)
        plt.colorbar()
        # print(np.amax(matrix))
        # print(np.amin(matrix))
        # plt.show()
        plt.savefig(output_path)
        plt.clf()

    @staticmethod
    def list_vis(l, sort=False, output_path=None):
        if sort:
            l.sort(reverse=True)
        # plt.xlim(xmin=0, xmax=len(l))  # for visualization, it's better don't use it, cause many points are near to x=0
        # ax.plot(prob, color='r', ls='-.', lw=1)  # line, used when y trend is stable
        plt.plot(range(1, len(l)+1), l, 'o', color='r', ms=1)  # dot
        plt.yscale('log')
        # plt.show()
        plt.savefig(output_path)
        plt.clf()

    @staticmethod
    def double_list_vis(prob, count):
        def color_y_axis(ax, color):
            """Color your axes."""
            for t in ax.get_yticklabels():
                t.set_color(color)
            return None

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        # line, used when y trend is stable
        # ax.plot(prob, color='r', ls='-.', lw=1)
        # ax2.plot(count, color='b', ls=':', lw=1)

        # dot
        ax.plot(prob, 'o', color='r', ms=1)
        ax2.plot(count, 'o', color='b', ms=0.5)

        ax.set_xlabel('word index')
        ax.set_ylabel('prob')
        ax2.set_ylabel('count')

        plt.xlim(xmin=0, xmax=len(prob))

        # ax.set_yscale('log')
        # ax2.set_yscale('log')
        # ax.set_ylim(0, np.max(y1[0]))
        # ax2.set_ylim(0, np.max(y2[0]))

        color_y_axis(ax, 'r')
        color_y_axis(ax2, 'b')

        plt.show()

    @staticmethod
    def cooccurrence_vis(encoded_edges_count_file_path, valid_vocabulary_path, word_count_path, output_folder):
        print(encoded_edges_count_file_path)
        ng = NoGraph(encoded_edges_count_file_path=encoded_edges_count_file_path,
                     valid_vocabulary_path=valid_vocabulary_path)
        ns = NegativeSamples(matrix=ng.cooccurrence_matrix, graph_index2wordId=ng.graph_index2wordId,
                             merged_dict_path=None, name_prefix=None)
        _, reorder_cooc = ns.reorder_matrix_by_word_count(word_count_path)
        png_name = multi_processing.get_file_name(encoded_edges_count_file_path).split('.txt')[0] + '_cooc.png'
        Visualization.matrix_vis(reorder_cooc, output_path=output_folder+png_name)

        ''' replaced by negative_samples_matrix_vis endswith='_1_step_rw_matrix.npy'
        ns_stoc = NegativeSamples(matrix=ng.get_stochastic_matrix(), graph_index2wordId=ng.graph_index2wordId,
                                  merged_dict_path=None, name_prefix=None)
        _, reorder_stoc = ns_stoc.reorder_matrix_by_word_count(word_count_path)
        png_name = multi_processing.get_file_name(encoded_edges_count_file_path).split('.txt')[0] + '_stoc.png'
        Visualization.matrix_vis(reorder_stoc, output_path=output_folder + png_name)
        '''

    @staticmethod
    def multi_cooccurrence_vis(encoded_edges_count_files_folder, word_count_path, valid_vocabulary_path, output_folder,
                           process_num):
        files_list = multi_processing.get_files_endswith(encoded_edges_count_files_folder, '_undirected.txt')
        p = Pool(process_num, maxtasksperchild=1)
        p.starmap_async(Visualization.cooccurrence_vis,
                        zip(files_list, repeat(valid_vocabulary_path), repeat(word_count_path), repeat(output_folder)))
        p.close()
        p.join()

    @staticmethod
    def negative_samples_matrix_vis(matrix_path, word_count_path, output_folder):
        print(matrix_path)
        nodes_path = re.search('(.*)_(.*)_step_rw_matrix.npy', matrix_path).group(1) + '_nodes.pickle'
        ns = NegativeSamples.load(matrix_path=matrix_path, graph_index2wordId_path=nodes_path,
                                  merged_dict_path=None)
        _, reordered_matrix = ns.reorder_matrix_by_word_count(word_count_path)
        png_name = multi_processing.get_file_name(matrix_path).split('.npy')[0] + '.png'
        Visualization.matrix_vis(reordered_matrix, output_path=output_folder+'png/'+png_name)

    @staticmethod
    def multi_negative_samples_matrix_vis(ns_folder, word_count_path, process_num, endswith):
        files_list = multi_processing.get_files_endswith(ns_folder, endswith)
        p = Pool(process_num, maxtasksperchild=1)
        p.starmap_async(Visualization.negative_samples_matrix_vis,
                        zip(files_list, repeat(word_count_path), repeat(ns_folder)))
        p.close()
        p.join()


'''[DEPRECATED]
class NegativeSamplesGenerator_old:
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
'''

if __name__ == '__main__':
    '''[DEPRECATED]
    # bridge = NegativeSamplesGenerator_old(encoded_edges_count_file_folder=config['graph']['graph_folder'],
    #                                                  ns_folder=config['word2vec']['negative_samples_folder'],
    #                                                  merged_dict_path=config['graph']['dicts_and_encoded_texts_folder'] + 'dict_merged.txt')
    # bridge.one_to_one_rw(encoded_edges_count_file_path=bridge.encoded_edges_count_file_folder+'encoded_edges_count_window_size_5_undirected.txt',
    #                      directed=False, t=1, negative=20, selected_mode='min')
    #
    # bridge.one_to_many_rw(encoded_edges_count_file_path=bridge.encoded_edges_count_file_folder+'encoded_edges_count_window_size_5_undirected.txt',
    #                       directed=False, t_max=1, negative=20)
    # bridge.many_to_many_rw(directed=False, t_max=2, potential_ns_len=1000, process_num=2)
    '''

    # # Generate ns matrix
    # start_time = time.time()
    # grid_searcher = NegativeSamplesGenerator(ns_folder=config['word2vec']['negative_samples_folder'],
    #                                          valid_vocabulary_path=config['graph']['dicts_and_encoded_texts_folder'] + 'valid_vocabulary_min_count_5_vocab_size_10000.txt')
    # grid_searcher.one_to_one(encoded_edges_count_file_path='output/intermediate data/graph/encoded_edges_count_window_size_10_undirected.txt',
    #                          t=1)
    # grid_searcher.many_to_many(encoded_edges_count_file_folder=config['graph']['graph_folder'], directed=False, t_max=5,
    #                            process_num=3)
    # print(common.count_time(start_time))
