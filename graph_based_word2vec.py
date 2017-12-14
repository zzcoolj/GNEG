import os
from word2vec_gensim_modified import Word2Vec
import pandas as pd
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

import sys
sys.path.insert(0, '../common/')
import multi_processing


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
    """ATTENTION
    Based on the assumption that we already have all 'write_translated_negative_samples_dict's based on different
    parameters combination.
    """
    def __init__(self, training_data_folder, index2word_path, merged_word_count_path, valid_vocabulary_path,
                 workers, sg, negative):
        # common parameters
        self.training_data_folder = training_data_folder
        self.index2word_path = index2word_path  # same as merged_dict_path
        self.merged_word_count_path = merged_word_count_path
        self.valid_vocabulary_path = valid_vocabulary_path
        self.workers = workers  # number of threads use for one word2vec calculation.
        self.sg = sg  # (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.
        self.negative = negative

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
        # TODO LATER a valid word count function in gdp, so as to transfer only one parameter for Word2Vec class.

        model = Word2Vec(sentences=sentences,
                         index2word_path=self.index2word_path,
                         merged_word_count_path=self.merged_word_count_path,
                         valid_vocabulary_path=self.valid_vocabulary_path,
                         translated_shortest_path_nodes_dict_path=ns_path,
                         ns_mode_pyx=ns_mode_pyx,
                         size=100, window=5, min_count=5, max_vocab_size=10000, workers=self.workers, sg=self.sg,
                         negative=self.negative)
        word_vectors = model.wv
        # TODO NOW save wv
        del model

        ''' Result of evaluate_word_pairs contains 3 parts:
        ((0.43915524919358867, 2.3681259690228147e-13),                                     Pearson
        SpearmanrResult(correlation=0.44614214937080449, pvalue=8.8819867392097872e-14),    Spearman 
        28.328611898016998)                                                                 ratio of pairs with unknown 
                                                                                            words (float)
        '''
        evaluation = word_vectors.evaluate_word_pairs('data/evaluation data/wordsim353/combined.tab')
        if ns_path:
            result = [multi_processing.get_file_name(ns_path),
                      evaluation[0][0], evaluation[0][1], evaluation[1][0], evaluation[1][1], evaluation[2]]
        else:
            result = [ns_path, evaluation[0][0], evaluation[0][1], evaluation[1][0], evaluation[1][1], evaluation[2]]
        print(result)
        return result

    def grid_search(self, ns_folder=config['word2vec']['negative_samples_folder']):
        evaluation_result = self.one_search(ns_path=None)  # baseline: original word2vec
        df = pd.DataFrame(columns=['NS file', 'Pearson correlation', 'Pearson pvalue', 'Spearman correlation',
                                   'Spearman pvalue', 'Ration of pairs with OOV'])
        df.loc[0] = evaluation_result

        i = 1
        files = multi_processing.get_files_endswith(data_folder=ns_folder, file_extension='.pickle')
        for file in files:
            evaluation_result= self.one_search(ns_path=file)
            df.loc[i] = evaluation_result
            i += 1

        writer = pd.ExcelWriter('output.xlsx')
        df.to_excel(writer, 'Sheet1')
        writer.save()


if __name__ == '__main__':
    # Fixed parameters for word2vec
    sg = 1  # Only care about skip-gram

    gs = GridSearch(training_data_folder='data/training data/Wikipedia-Dumps_en_20170420_prep',
                    index2word_path=config['graph']['dicts_and_encoded_texts_folder'] + 'dict_merged.txt',
                    merged_word_count_path=config['graph']['dicts_and_encoded_texts_folder'] + 'word_count_all.txt',
                    valid_vocabulary_path=config['graph']['dicts_and_encoded_texts_folder'] + 'valid_vocabulary_min_count_5_vocab_size_10000.txt',
                    workers=5, sg=sg, negative=20)

    gs.grid_search()
