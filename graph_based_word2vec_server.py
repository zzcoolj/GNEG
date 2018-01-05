import graph_based_word2vec as gbw
import configparser
import time
import sys
sys.path.insert(0, '../common/')
import common

config = configparser.ConfigParser()
config.read('config.ini')

print('timer starts')
start_time = time.time()

sg = 1  # Only care about skip-gram
gs = gbw.GridSearch_new(training_data_folder='/dev/shm/zzheng-tmp/prep/',
                        index2word_path=config['graph']['dicts_and_encoded_texts_folder'] + 'dict_merged.txt',
                        merged_word_count_path=config['graph']['dicts_and_encoded_texts_folder'] + 'word_count_all.txt',
                        valid_vocabulary_path=config['graph']['dicts_and_encoded_texts_folder'] + 'valid_vocabulary_min_count_5_vocab_size_10000.txt',
                        workers=50, sg=sg, negative=20)

# gs.grid_search()
gs.one_search(matrix_path=config['word2vec']['negative_samples_folder']+'encoded_edges_count_window_size_3_undirected_3_step_rw_matrix.npy',
              row_column_indices_value_path=config['word2vec']['negative_samples_folder']+'encoded_edges_count_window_size_3_undirected_nodes.pickle',
              power=0.5)

print('time in seconds:', common.count_time(start_time))
