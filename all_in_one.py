import graph_data_provider as gdp
import negative_samples_generator as nsg
import graph_based_word2vec as gbw
import configparser
import time
import pandas as pd
import sys
sys.path.insert(0, '../common/')
import common

config = configparser.ConfigParser()
config.read('config.ini')

window_size = 10
small_units = ['AA']
medium_units = ['AA', 'BB', 'CC', 'DD', 'EE']
medium_folder = 'output/medium/'
sg = 1  # Only care about skip-gram

# print('build graph')
# start_time = time.time()
#
# # 100 files in one unit, so set process_num to be 10 is okay
# # TODO TEMP medium_folder
# gdp.part_of_data(units=medium_units, window_size=window_size, process_num=30, output_folder='output/intermediate data/')
#
# print('time in seconds:', common.count_time(start_time))

# print('build ns')
# start_time = time.time()
#
# # output/intermediate data/negative_samples_partial/
# # output/intermediate data/dicts_and_encoded_texts/valid_vocabulary_partial_min_count_5_vocab_size_10000.txt
# grid_searcher = nsg.NegativeSamplesGenerator(ns_folder='output/intermediate data/negative_samples_partial_difference/',
#                                              valid_vocabulary_path='output/intermediate data/dicts_and_encoded_texts/valid_vocabulary_partial_min_count_5_vocab_size_10000.txt')
# # # t-step random walks
# # grid_searcher.many_to_many(encoded_edges_count_file_folder='output/intermediate data/graph/', directed=False, t_max=4, process_num=window_size-1, partial=True)
# # # stochastic matrix
# # grid_searcher.multi_functions(f=grid_searcher.get_stochastic_matrix, encoded_edges_count_file_folder='output/intermediate data/graph/', directed=False, process_num=window_size-1, partial=True)
# # difference matrix
# grid_searcher.multi_difference_matrix(encoded_edges_count_file_folder='output/intermediate data/graph/',
#                                       merged_word_count_path=config['graph']['dicts_and_encoded_texts_folder'] + 'word_count_partial.txt',
#                                       directed=False, process_num=window_size-1, partial=True)
#
# print('time in seconds:', common.count_time(start_time))

print('graph-based word2vec')
start_time = time.time()

# partial wiki data
# data/training data/Wikipedia-Dumps_en_20170420_prep
# gs = gbw.GridSearch_new(training_data_folder='/dev/shm/zzheng-tmp/prep/',
#                         index2word_path=config['graph']['dicts_and_encoded_texts_folder'] + 'dict_merged.txt',
#                         merged_word_count_path=config['graph']['dicts_and_encoded_texts_folder'] + 'word_count_partial.txt',
#                         valid_vocabulary_path=config['graph']['dicts_and_encoded_texts_folder'] + 'valid_vocabulary_partial_min_count_5_vocab_size_10000.txt',
#                         workers=62, sg=sg, size=200, negative=5, units=small_units, iterations=3)
gs = gbw.GridSearch_new(training_data_folder='/dev/shm/zzheng-tmp/prep/',
                        index2word_path=config['graph']['dicts_and_encoded_texts_folder'] + 'dict_merged.txt',
                        merged_word_count_path=medium_folder + 'dicts_and_encoded_texts/word_count_partial.txt',
                        valid_vocabulary_path=medium_folder + 'dicts_and_encoded_texts/valid_vocabulary_partial_min_count_5_vocab_size_10000.txt',
                        workers=62, sg=sg, size=200, negative=5, units=medium_units, iterations=3)
gs.one_search(matrix_path=None, graph_index2wordId_path=None, power=None, ns_mode_pyx=0)
gs = gbw.GridSearch_new(training_data_folder='/dev/shm/zzheng-tmp/prep/',
                        index2word_path=config['graph']['dicts_and_encoded_texts_folder'] + 'dict_merged.txt',
                        merged_word_count_path=medium_folder + 'dicts_and_encoded_texts/word_count_partial.txt',
                        valid_vocabulary_path=medium_folder + 'dicts_and_encoded_texts/valid_vocabulary_partial_min_count_5_vocab_size_10000.txt',
                        workers=62, sg=sg, size=200, negative=10, units=medium_units, iterations=3)
gs.one_search(matrix_path=None, graph_index2wordId_path=None, power=None, ns_mode_pyx=0)
gs = gbw.GridSearch_new(training_data_folder='/dev/shm/zzheng-tmp/prep/',
                        index2word_path=config['graph']['dicts_and_encoded_texts_folder'] + 'dict_merged.txt',
                        merged_word_count_path=medium_folder + 'dicts_and_encoded_texts/word_count_partial.txt',
                        valid_vocabulary_path=medium_folder + 'dicts_and_encoded_texts/valid_vocabulary_partial_min_count_5_vocab_size_10000.txt',
                        workers=62, sg=sg, size=200, negative=15, units=medium_units, iterations=3)
gs.one_search(matrix_path=None, graph_index2wordId_path=None, power=None, ns_mode_pyx=0)
# # t-step random walks
# gs.grid_search(ns_folder='output/intermediate data/negative_samples_partial_noSelfLoops/')  # 116876.32733845711s
# # stochastic matrix
# gs.grid_search_bis(ns_folder='output/intermediate data/negative_samples_partial_stochastic/')  # 68200s
# # difference matrix
# gs.grid_search_tri(ns_folder='output/intermediate data/negative_samples_partial_difference/')  # 31848s

# # whole wiki data
# gs = gbw.GridSearch_new(training_data_folder='/dev/shm/zzheng-tmp/prep/',
#                         index2word_path=config['graph']['dicts_and_encoded_texts_folder'] + 'dict_merged.txt',
#                         merged_word_count_path=config['graph']['dicts_and_encoded_texts_folder'] + 'word_count_all.txt',
#                         valid_vocabulary_path=config['graph']['dicts_and_encoded_texts_folder'] + 'valid_vocabulary_min_count_5_vocab_size_10000.txt',
#                         workers=62, sg=sg, size=200, negative=5, iterations=3)
# gs.one_search(matrix_path=None, graph_index2wordId_path=None, power=None, ns_mode_pyx=0)
# gs.one_search(matrix_path='output/intermediate data/negative_samples_partial/encoded_edges_count_window_size_5_undirected_partial_3_step_rw_matrix.npy',
#               graph_index2wordId_path='output/intermediate data/negative_samples_partial/encoded_edges_count_window_size_5_undirected_partial_nodes.pickle',
#               power=0.75, ns_mode_pyx=1)

print('time in seconds:', common.count_time(start_time))












# corpus size count
# merged_word_count = gdp.read_two_columns_file_to_build_dictionary_type_specified(
#     config['graph']['dicts_and_encoded_texts_folder'] + 'word_count_all.txt', key_type=str, value_type=int)
# count = 0
# for temp in merged_word_count.values():
#     count += temp
# print('all', count)
#
# merged_word_count = gdp.read_two_columns_file_to_build_dictionary_type_specified(
#     config['graph']['dicts_and_encoded_texts_folder'] + 'word_count_partial.txt', key_type=str, value_type=int)
# count = 0
# for temp in merged_word_count.values():
#     count += temp
# print('small partial', count)
#
# merged_word_count = gdp.read_two_columns_file_to_build_dictionary_type_specified(
#     'output/medium/dicts_and_encoded_texts/' + 'word_count_partial.txt', key_type=str, value_type=int)
# count = 0
# for temp in merged_word_count.values():
#     count += temp
# print('medium partial', count)
