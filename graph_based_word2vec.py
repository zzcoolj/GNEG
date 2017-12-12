import os
from word2vec_gensim_modified import Word2Vec
import configparser
import graph_builder_networkx as gbn
config = configparser.ConfigParser()
config.read('config.ini')


# WikiSentences class modified based on the code from https://rare-technologies.com/word2vec-tutorial/
class WikiSentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for sub_folder_name in os.listdir(self.dirname):
            if not sub_folder_name.startswith('.'):
                sub_folder_path = os.path.join(self.dirname, sub_folder_name)
                for fname in os.listdir(sub_folder_path):
                    if not fname.startswith('.'):
                        for line in open(os.path.join(sub_folder_path, fname), 'rb'):
                            yield [word.decode('iso-8859-1') for word in line.split()]


class GridSearch(object):
    def __init__(self, encoded_edges_count_files_folder, index2word_path, translated_ns_dict_output_folder,
                 training_data_folder, merged_word_count_path, valid_vocabulary_path, workers, sg,
                 negative):
        # common parameters
        self.encoded_edges_count_files_folder = encoded_edges_count_files_folder
        self.index2word_path = index2word_path  # same as merged_dict_path
        self.negative = negative
        self.translated_ns_dict_output_folder = translated_ns_dict_output_folder
        self.training_data_folder = training_data_folder
        self.merged_word_count_path = merged_word_count_path
        self.valid_vocabulary_path = valid_vocabulary_path
        self.workers = workers  # number of threads use for one word2vec calculation.
        self.sg = sg  # (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.

        # directed/undirected
        # selected_mode='max'/'min'
        # ns_mode_pyx=1/0  # ns_mode_pyx:  0: original, using cum_table; 1: using graph-based ns_table
        # t=1-5

    def one_search_rw(self, encoded_edges_count_file_path, directed, t, selected_mode, ns_mode_pyx):
        graph = gbn.NXGraph.from_encoded_edges_count_file(path=encoded_edges_count_file_path, directed=directed)
        nodes, matrix = graph.get_t_step_random_walk_stochastic_matrix(t=t)
        ns = gbn.NegativeSamples(matrix=matrix, row_column_indices_value=nodes, merged_dict_path=self.index2word_path,
                                 name_prefix=graph.name_prefix)
        # TODO NOW name unique?
        ns.write_translated_negative_samples_dict(n=self.negative, selected_mode=selected_mode,
                                                  output_folder=self.translated_ns_dict_output_folder)

        sentences = WikiSentences(self.training_data_folder)  # a memory-friendly iterator
        model = Word2Vec(sentences=sentences,
                         index2word_path=self.index2word_path,
                         merged_word_count_path=self.merged_word_count_path,
                         valid_vocabulary_path=self.valid_vocabulary_path,
                         translated_shortest_path_nodes_dict_path=self.translated_ns_dict_output_folder +
                                                                  ns.name_prefix + '_translated_ns_dict.pickle',
                         ns_mode_pyx=ns_mode_pyx,
                         size=100, window=5, min_count=5, max_vocab_size=10000, workers=self.workers, sg=self.sg,
                         negative=self.negative)
        word_vectors = model.wv
        # TODO NOW save wv
        print(word_vectors.evaluate_word_pairs('data/evaluation data/wordsim353/combined.tab'))
        del model

    def grid_search(self):
        # baseline: original word2vec, t and selected_mode has no effect.
        ns_mode_pyx = 0


if __name__ == '__main__':
    gs = GridSearch(encoded_edges_count_files_folder=config['graph']['graph_folder'],
                    index2word_path=config['graph']['dicts_and_encoded_texts_folder'] + 'dict_merged.txt',
                    translated_ns_dict_output_folder=config['graph']['graph_folder'],
                    training_data_folder='data/training data/Wikipedia-Dumps_en_20170420_prep',
                    merged_word_count_path=config['graph']['dicts_and_encoded_texts_folder'] + 'word_count_all.txt',
                    valid_vocabulary_path=config['graph']['dicts_and_encoded_texts_folder'] + 'valid_vocabulary_min_count_5_vocab_size_10000.txt',
                    workers=4, sg=1, negative=20)


