import graph_based_word2vec as gbw
import configparser
import time
import sys
sys.path.insert(0, '../common/')
import common

config = configparser.ConfigParser()
config.read('config.ini')

# # Whole wiki data
# print('timer starts')
# start_time = time.time()
#
# sg = 1  # Only care about skip-gram
# gs = gbw.GridSearch_new(training_data_folder='/dev/shm/zzheng-tmp/prep/',
#                         index2word_path=config['graph']['dicts_and_encoded_texts_folder'] + 'dict_merged.txt',
#                         merged_word_count_path=config['graph']['dicts_and_encoded_texts_folder'] + 'word_count_all.txt',
#                         valid_vocabulary_path=config['graph']['dicts_and_encoded_texts_folder'] + 'valid_vocabulary_min_count_5_vocab_size_10000.txt',
#                         workers=60, sg=sg, negative=20)
#
# # gs.one_search(matrix_path=None, graph_index2wordId_path=None, power=None)
# gs.one_search(matrix_path=config['word2vec']['negative_samples_folder']+'encoded_edges_count_window_size_10_undirected_3_step_rw_matrix.npy',
#               graph_index2wordId_path=config['word2vec']['negative_samples_folder'] + 'encoded_edges_count_window_size_10_undirected_nodes.pickle',
#               power=0.1)
#
# print('time in seconds:', common.count_time(start_time))

# small data
# 20 cores 7862s for 99 tests
print('timer starts')
start_time = time.time()

sg = 1  # Only care about skip-gram
gs = gbw.GridSearch_new(training_data_folder='data/training data/Wikipedia-Dumps_en_20170420_prep',
                        index2word_path='output/intermediate_data_for_small_corpus/dicts_and_encoded_texts/dict_merged.txt',
                        merged_word_count_path='output/intermediate_data_for_small_corpus/dicts_and_encoded_texts/word_count_all.txt',
                        valid_vocabulary_path='output/intermediate_data_for_small_corpus/dicts_and_encoded_texts/valid_vocabulary_min_count_5_vocab_size_10000.txt',
                        workers=4, sg=sg, negative=20)

gs.one_search(matrix_path='output/intermediate_data_for_small_corpus/negative_samples/encoded_edges_count_window_size_9_undirected_2_step_rw_matrix.npy',
              graph_index2wordId_path='output/intermediate_data_for_small_corpus/negative_samples/encoded_edges_count_window_size_9_undirected_nodes.pickle',
              power=1)

# gs.grid_search(ns_folder='output/intermediate_data_for_small_corpus/negative_samples/')
print('time in seconds:', common.count_time(start_time))
