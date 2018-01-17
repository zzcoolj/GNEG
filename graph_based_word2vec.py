import os
from word2vec_gensim_modified import Word2Vec
import pandas as pd
import re
import random
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

import sys
sys.path.insert(0, '../common/')
import multi_processing


# WikiSentences class modified based on the code from https://rare-technologies.com/word2vec-tutorial/
class WikiSentences(object):
    def __init__(self, dirname, units=None):
        """
        :param dirname:
        :param units: 0: use all data in dir
        :param random_selection:
        """
        self.dirname = dirname
        # wiki data has folders like 'AA', 'AB', ..., 'EJ', one unit stands for one of these folders.
        self.sub_folder_names = [sub_folder_name for sub_folder_name in os.listdir(self.dirname)
                                 if not sub_folder_name.startswith('.')]
        if units:
            # if random_selection:
            #     self.sub_folder_names = random.sample(self.sub_folder_names, units)
            # else:
            #     # get last units number of elements; the original list is like ['AB', 'AA']
            #     self.sub_folder_names = self.sub_folder_names[-units:]
            self.sub_folder_names = units
            print('ATTENTION: Only part of the corpus is used.', self.sub_folder_names)

    def __iter__(self):
        for sub_folder_name in self.sub_folder_names:
            sub_folder_path = os.path.join(self.dirname, sub_folder_name)
            for fname in os.listdir(sub_folder_path):
                if not fname.startswith('.'):
                    for line in open(os.path.join(sub_folder_path, fname), 'r', encoding='utf-8'):
                        yield line.strip().split()


class GridSearch_old(object):
    """ATTENTION [DEPRECATED]
    This class only serves the old uniform ns selection idea, which should be deprecated.
    (work together with NegativeSamplesGenerator_old class in graph_builder_network.py)

    Based on the assumption that we already have all 'write_translated_negative_samples_dict's based on different
    parameters combination.
    """
    def __init__(self, training_data_folder, index2word_path, merged_word_count_path, valid_vocabulary_path,
                 workers, sg, negative, potential_ns_len):
        # common parameters
        self.training_data_folder = training_data_folder
        self.index2word_path = index2word_path  # same as merged_dict_path
        self.merged_word_count_path = merged_word_count_path
        self.valid_vocabulary_path = valid_vocabulary_path
        self.workers = workers  # number of threads use for one word2vec calculation.
        self.sg = sg  # (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.
        self.negative = negative
        self.potential_ns_len = potential_ns_len

    def one_search(self, ns_path):
        sentences = WikiSentences(self.training_data_folder)  # a memory-friendly iterator

        # ns_mode_pyx:  0: original, using cum_table; 1: using graph-based ns_table
        if ns_path:
            ns_mode_pyx = 1
        else:
            ns_mode_pyx = 0

        """ATTENTION
        The only reason the Word2Vec class needs index2word_path, merged_word_count_path, valid_vocabulary_path is to 
        get valid words' count.
        """

        model = Word2Vec(sentences=sentences,
                         index2word_path=self.index2word_path,
                         merged_word_count_path=self.merged_word_count_path,
                         valid_vocabulary_path=self.valid_vocabulary_path,
                         translated_shortest_path_nodes_dict_path=ns_path,
                         ns_mode_pyx=ns_mode_pyx,
                         potential_ns_len=self.potential_ns_len,
                         size=100, window=5, min_count=5, max_vocab_size=10000, workers=self.workers, sg=self.sg,
                         negative=self.negative)
        word_vectors = model.wv
        del model

        ''' Result of evaluate_word_pairs contains 3 parts:
        ((0.43915524919358867, 2.3681259690228147e-13),                                     Pearson
        SpearmanrResult(correlation=0.44614214937080449, pvalue=8.8819867392097872e-14),    Spearman 
        28.328611898016998)                                                                 ratio of pairs with unknown 
                                                                                            words (float)
        '''
        evaluation = word_vectors.evaluate_word_pairs('data/evaluation data/wordsim353/combined.tab')
        if ns_path:
            ns_name = multi_processing.get_file_name(ns_path)
            # e.g. encoded_edges_count_window_size_3_undirected_ns_2_max
            ns_name_information = re.search('encoded_edges_count_window_size_(.*)_(.*)_ns_(.*)_(.*)', ns_name)
            result = [ns_name, int(ns_name_information.group(1)), ns_name_information.group(2),
                      int(ns_name_information.group(3)), ns_name_information.group(4),
                      evaluation[0][0], evaluation[0][1], evaluation[1][0], evaluation[1][1], evaluation[2]]
        else:
            result = [ns_path, None, None, None, None,
                      evaluation[0][0], evaluation[0][1], evaluation[1][0], evaluation[1][1], evaluation[2]]
        print(result)
        return result

    def grid_search(self, ns_folder=config['word2vec']['negative_samples_folder']):
        evaluation_result = self.one_search(ns_path=None)  # baseline: original word2vec
        df = pd.DataFrame(columns=['NS file', 'Graph window size', 'Directed/Undirected', 't-random-walk', 'Max/Min',
                                   'Pearson correlation', 'Pearson pvalue', 'Spearman correlation',
                                   'Spearman pvalue', 'Ration of pairs with OOV'])
        df.loc[0] = evaluation_result

        i = 1
        files = multi_processing.get_files_endswith(data_folder=ns_folder, file_extension='.pickle')
        for file in files:
            evaluation_result = self.one_search(ns_path=file)
            df.loc[i] = evaluation_result
            i += 1

        writer = pd.ExcelWriter('output.xlsx')
        df.to_excel(writer, 'Sheet1')
        writer.save()


class GridSearch_new(object):
    def __init__(self, training_data_folder, index2word_path, merged_word_count_path, valid_vocabulary_path,
                 workers, sg, negative, units=None):
        # common parameters
        self.training_data_folder = training_data_folder
        self.index2word_path = index2word_path  # same as merged_dict_path
        self.merged_word_count_path = merged_word_count_path
        self.valid_vocabulary_path = valid_vocabulary_path
        self.workers = workers  # number of threads use for one word2vec calculation.
        self.sg = sg  # (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.
        self.negative = negative
        self.units = units

    def one_search(self, matrix_path, graph_index2wordId_path, power, ns_mode_pyx=0):
        sentences = WikiSentences(self.training_data_folder, units=self.units)

        # ns_mode_pyx:  0: original, using cum_table; 1: using graph-based ns_table
        if matrix_path:
            ns_mode_pyx = 1
        elif ns_mode_pyx == -1:
            ns_mode_pyx = -1
        else:
            ns_mode_pyx = 0

        """ATTENTION
        The only reason the Word2Vec class needs index2word_path, merged_word_count_path, valid_vocabulary_path is to 
        get valid words' count.
        """
        # TODO LATER a valid word count function in gdp, so as to transfer only one parameter to Word2Vec class.
        # TODO LATER rethink about min_count & max_vocab_size, maybe they are useless?

        model = Word2Vec(sentences=sentences,
                         index2word_path=self.index2word_path,
                         merged_word_count_path=self.merged_word_count_path,
                         valid_vocabulary_path=self.valid_vocabulary_path,
                         matrix_path=matrix_path,
                         graph_index2wordId_path=graph_index2wordId_path,
                         ns_mode_pyx=ns_mode_pyx,
                         power=power,
                         size=100, window=5, min_count=5, max_vocab_size=10000, workers=self.workers, sg=self.sg,
                         negative=self.negative)
        word_vectors = model.wv
        # TODO LATER save wv
        del model

        ''' Result of evaluate_word_pairs contains 3 parts:
        ((0.43915524919358867, 2.3681259690228147e-13),                                     Pearson
        SpearmanrResult(correlation=0.44614214937080449, pvalue=8.8819867392097872e-14),    Spearman 
        28.328611898016998)                                                                 ratio of pairs with unknown 
                                                                                            words (float)
        '''
        evaluation = word_vectors.evaluate_word_pairs('data/evaluation data/wordsim353/combined.tab')
        if ns_mode_pyx == 1:
            ns_name = multi_processing.get_file_name(matrix_path)
            # e.g. encoded_edges_count_window_size_5_undirected_1_step_rw_matrix
            ns_name_information = re.search('encoded_edges_count_window_size_(.*)_(.*)_(.*)_step_rw_matrix', ns_name)
            result = [ns_name, int(ns_name_information.group(1)), ns_name_information.group(2),
                      int(ns_name_information.group(3)), power,
                      evaluation[0][0], evaluation[0][1], evaluation[1][0], evaluation[1][1], evaluation[2]]
        elif ns_mode_pyx == 0:
            # in original word2vec (frontline), power is set to 0.75 as default.
            result = [matrix_path, None, None, None, 0.75,
                      evaluation[0][0], evaluation[0][1], evaluation[1][0], evaluation[1][1], evaluation[2]]
        elif ns_mode_pyx == -1:
            # ibottomline, power is set to 0 as default.
            result = [matrix_path, None, None, None, 0,
                      evaluation[0][0], evaluation[0][1], evaluation[1][0], evaluation[1][1], evaluation[2]]
        print(result)
        return result

    def grid_search(self, ns_folder=config['word2vec']['negative_samples_folder']):
        # frontline: original word2vec
        evaluation_result = self.one_search(matrix_path=None, graph_index2wordId_path=None, power=None)
        df = pd.DataFrame(columns=['NS file', 'Graph window size', 'Directed/Undirected', 't-random-walk', 'power',
                                   'Pearson correlation', 'Pearson pvalue', 'Spearman correlation',
                                   'Spearman pvalue', 'Ration of pairs with OOV'])
        df.loc[0] = evaluation_result

        # bottomline: uniformly distribution
        evaluation_result = self.one_search(matrix_path=None, graph_index2wordId_path=None, power=None, ns_mode_pyx=-1)
        df.loc[1] = evaluation_result

        i = 2
        files = multi_processing.get_files_endswith(data_folder=ns_folder, file_extension='.npy')
        for file in files:
            nodes_path = re.search('(.*)_(.*)_step_rw_matrix.npy', file).group(1) + '_nodes.pickle'
            # for power in [-0.001, -0.1, -1, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2]:
            for power in [-1, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1]:
                try:
                    evaluation_result = self.one_search(matrix_path=file, graph_index2wordId_path=nodes_path,
                                                        power=power)
                except:
                    print('ERROR:', file, nodes_path)
                    continue
                else:
                    df.loc[i] = evaluation_result
                    i += 1

        writer = pd.ExcelWriter(ns_folder+'output.xlsx')
        df.to_excel(writer, 'Sheet1')
        writer.save()


if __name__ == '__main__':
    # Fixed parameters for word2vec
    sg = 1  # Only care about skip-gram

    # # DEPRECATED
    # gs = GridSearch_old(training_data_folder='data/training data/Wikipedia-Dumps_en_20170420_prep',
    #                     index2word_path=config['graph']['dicts_and_encoded_texts_folder'] + 'dict_merged.txt',
    #                     merged_word_count_path=config['graph']['dicts_and_encoded_texts_folder'] + 'word_count_all.txt',
    #                     valid_vocabulary_path=config['graph']['dicts_and_encoded_texts_folder'] + 'valid_vocabulary_min_count_5_vocab_size_10000.txt',
    #                     workers=5, sg=sg, negative=20, potential_ns_len=200)
    # gs.grid_search(ns_folder='output/intermediate data/negative_samples_potential_ns_len_200/')

    gs2 = GridSearch_new(training_data_folder='data/training data/Wikipedia-Dumps_en_20170420_prep',
                         index2word_path=config['graph']['dicts_and_encoded_texts_folder'] + 'dict_merged.txt',
                         merged_word_count_path=config['graph']['dicts_and_encoded_texts_folder'] + 'word_count_all.txt',
                         valid_vocabulary_path=config['graph']['dicts_and_encoded_texts_folder'] + 'valid_vocabulary_min_count_5_vocab_size_10000.txt',
                         workers=5, sg=sg, negative=20, units=None)
    gs2.one_search(matrix_path=None, graph_index2wordId_path=None, power=None)
    # gs2.one_search(matrix_path=config['word2vec']['negative_samples_folder']+'encoded_edges_count_window_size_9_undirected_2_step_rw_matrix.npy',
    #                graph_index2wordId_path=config['word2vec']['negative_samples_folder']+'encoded_edges_count_window_size_9_undirected_nodes.pickle',
    #                power=1)
    # gs2.grid_search()
    # gs2.one_search(matrix_path=config['word2vec']['negative_samples_folder']+'shelter/encoded_edges_count_window_size_10_undirected_1_step_rw_matrix_new.npy',
    #                graph_index2wordId_path=config['word2vec']['negative_samples_folder']+'shelter/encoded_edges_count_window_size_10_undirected_nodes.pickle',
    #                power=0.75)

