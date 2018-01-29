import negative_samples_generator as nsg
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('config.ini')


word_count_path = config['graph']['dicts_and_encoded_texts_folder'] + 'word_count_all.txt'
valid_vocabulary_path = config['graph']['dicts_and_encoded_texts_folder'] + 'valid_vocabulary_min_count_5_vocab_size_10000.txt'


'''Negative Samples Original: list
only for valid word
'''
# count_list = nsg.NegativeSamples.get_valid_vocab_count_list(word_count_path=word_count_path, valid_vocabulary_path=valid_vocabulary_path)
# nsg.Visualization.list_vis(count_list, sort=True, output_path=config['word2vec']['negative_samples_folder']+'png/'+'ns_original.png')
'''Negative Samples Original: matrix
1. sort by word count
2. power=0.75
3. normalization
'''
# count_list = nsg.NegativeSamples.get_valid_vocab_count_list(word_count_path=word_count_path, valid_vocabulary_path=valid_vocabulary_path)
# count_list.sort(reverse=True)
# count_list = [i**0.75 for i in count_list]
# count_list = [float(i)/sum(count_list) for i in count_list]
# matrix = np.array([count_list for i in range(len(count_list))])
# nsg.Visualization.matrix_vis(matrix, output_path=config['word2vec']['negative_samples_folder']+'png/'+'ns_original_matrix.png')


'''Negative Samples Co-occurrence-based: co-occurrence-matrix
'''
nsg.Visualization.multi_cooccurrence_vis(encoded_edges_count_files_folder=config['graph']['graph_folder'],
                                         word_count_path=word_count_path,
                                         valid_vocabulary_path=valid_vocabulary_path,
                                         output_folder=config['graph']['graph_folder']+'png/',
                                         process_num=9)


'''Negative Samples Co-occurrence-based: matrix
1. normalization
2. For small corpus, the matrix is not representative enough.
'''
# # Way 1 remember to unblock code in cooccurrence_vis
# nsg.Visualization.multi_cooccurrence_vis(encoded_edges_count_files_folder=config['graph']['graph_folder'],
#                                          word_count_path=word_count_path,
#                                          valid_vocabulary_path=valid_vocabulary_path,
#                                          output_folder=config['graph']['graph_folder']+'png/',
#                                          process_num=9)
# Way 2
nsg.Visualization.multi_negative_samples_matrix_vis(config['word2vec']['negative_samples_folder'], word_count_path=word_count_path, endswith='_1_step_rw_matrix.npy', process_num=1)


'''Negative Samples Co-occurrence-based: random walk
'''
# nsg.Visualization.multi_negative_samples_matrix_vis(config['word2vec']['negative_samples_folder'], word_count_path=word_count_path, endswith='.npy', process_num=20)
