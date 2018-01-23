import graph_data_provider as gdp
import negative_samples_generator as nsg
import graph_based_word2vec as gbw
import configparser
import time
import sys
sys.path.insert(0, '../common/')
import common

config = configparser.ConfigParser()
config.read('config.ini')

window_size = 10
units = ['AA']

# print('build graph')
# start_time = time.time()
#
# # 100 files in one unit, so set process_num to be 10 is okay
# gdp.part_of_data(units=units, window_size=window_size, process_num=10)
#
# print('time in seconds:', common.count_time(start_time))
#
# print('build ns')
# start_time = time.time()
# grid_searcher = nsg.NegativeSamplesGenerator(ns_folder='output/intermediate data/negative_samples_partial/',
#                                              valid_vocabulary_path='output/intermediate data/dicts_and_encoded_texts/valid_vocabulary_partial_min_count_5_vocab_size_10000.txt')
# grid_searcher.many_to_many(encoded_edges_count_file_folder='output/intermediate data/graph/', directed=False, t_max=6, process_num=window_size-1, partial=True)
# print('time in seconds:', common.count_time(start_time))

print('graph-based word2vec')
start_time = time.time()
sg = 1  # Only care about skip-gram

# data/training data/Wikipedia-Dumps_en_20170420_prep

gs = gbw.GridSearch_new(training_data_folder='/dev/shm/zzheng-tmp/prep/',
                        index2word_path=config['graph']['dicts_and_encoded_texts_folder'] + 'dict_merged.txt',
                        merged_word_count_path=config['graph']['dicts_and_encoded_texts_folder'] + 'word_count_partial.txt',
                        valid_vocabulary_path=config['graph']['dicts_and_encoded_texts_folder'] + 'valid_vocabulary_partial_min_count_5_vocab_size_10000.txt',
                        workers=60, sg=sg, negative=20, units=units)
# gs.grid_search(ns_folder='output/intermediate data/negative_samples_partial/')  # 116876.32733845711s
gs.one_search(matrix_path=None, graph_index2wordId_path=None, power=None, ns_mode_pyx=0)
gs.one_search(matrix_path='output/intermediate data/negative_samples_partial/encoded_edges_count_window_size_5_undirected_partial_3_step_rw_matrix.npy',
              graph_index2wordId_path='output/intermediate data/negative_samples_partial/encoded_edges_count_window_size_5_undirected_partial_nodes.pickle',
              power=0.75, ns_mode_pyx=1)
print('time in seconds:', common.count_time(start_time))
