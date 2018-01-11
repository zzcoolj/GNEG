import negative_samples_generator as nsg
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

# ns_folder=config['word2vec']['negative_samples_folder']
ns_folder = 'output/intermediate_data_for_small_corpus/negative_samples/'
# encoded_edges_count_file_folder=config['graph']['graph_folder']
encoded_edges_count_file_folder = 'output/intermediate_data_for_small_corpus/graph/'

grid_searcher = nsg.NegativeSamplesGenerator(ns_folder=ns_folder,
                                             valid_vocabulary_path=config['graph']['dicts_and_encoded_texts_folder'] +
                                                                   'valid_vocabulary_min_count_5_vocab_size_10000.txt')
grid_searcher.many_to_many(encoded_edges_count_file_folder=encoded_edges_count_file_folder, directed=False, t_max=1,
                           process_num=9)
