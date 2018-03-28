from gensim import utils
from scipy import stats

import sys
sys.path.insert(0, '../common/')
import common


class StatisticalSignificance(object):

    def __init__(self, keyedVectors):
        self.keyedVectors = keyedVectors

    def evaluate_word_pairs(self, pairs, output_path, delimiter='\t', restrict_vocab=300000):
        """
        Given evaluation data set path (e.g. path to the WordSim-353),
        following the word pairs order in that data set,
        we calculate similarity of each word pair by using our mode.

        ATTENTION: golden similarity given by the evaluation data set is not necessary to this function. Here we use it
                    for the result check.
        """

        ok_vocab = [(w, self.keyedVectors.vocab[w]) for w in self.keyedVectors.index2word[:restrict_vocab]]
        ok_vocab = dict((w.upper(), v) for w, v in reversed(ok_vocab))

        similarity_gold = []
        similarity_model = []
        oov = 0

        original_vocab = self.keyedVectors.vocab
        self.keyedVectors.vocab = ok_vocab

        for line_no, line in enumerate(utils.smart_open(pairs)):
            line = utils.to_unicode(line)
            if line.startswith('#'):
                # May be a comment
                continue
            else:
                try:
                    a, b, sim = [word.upper() for word in line.split(delimiter)]
                    sim = float(sim)
                except:
                    continue
                if a not in ok_vocab or b not in ok_vocab:
                    oov += 1
                    continue
                similarity_gold.append(sim)  # Similarity from the dataset
                similarity_model.append(self.keyedVectors.similarity(a, b))  # Similarity from the model

        self.keyedVectors.vocab = original_vocab

        common.write_to_pickle(similarity_model, output_path)

        spearman = stats.spearmanr(similarity_gold, similarity_model)
        pearson = stats.pearsonr(similarity_gold, similarity_model)
        oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100

        return spearman, pearson, oov_ratio

    @staticmethod
    def coefficient_calculation(similarity_model_1_path, similarity_model_2_path):
        similarity_model_1 = common.read_pickle(similarity_model_1_path)
        similarity_model_2 = common.read_pickle(similarity_model_2_path)

        spearman = stats.spearmanr(similarity_model_1, similarity_model_2)
        pearson = stats.pearsonr(similarity_model_1, similarity_model_2)

        return spearman, pearson

    def write_evaluation_questions_words_result(self, path='data/evaluation data/questions-words.txt'):
        """
        ATTENTION:
        Same questions could appear more than once in different sections.
        e.g. ATHENS GREECE BANGKOK THAILAND appears twice
        """
        accuracy = self.keyedVectors.accuracy(path)  # 4478

        result = {}

        for i in range(len(accuracy) - 1):
            correct = []
            incorrect = []
            correct.extend(accuracy[i]['correct'])
            incorrect.extend(accuracy[i]['incorrect'])

            for question_words in correct:
                key = str(i) + ' ' + ' '.join(question_words)
                if key in result:
                    print('wrong')
                    print(key)
                    exit()
                result[key] = 1

            for question_words in incorrect:
                key = str(i) + ' ' + ' '.join(question_words)
                if key in result:
                    print('wrong')
                    print(key)
                    exit()
                result[key] = 0

        print(len(result), 'this number should be equal to 6032')
        for key, value in result:
            print(key, value)
            exit()


        sem_correct = sum((len(accuracy[i]['correct']) for i in range(5)))
        sem_total = sum((len(accuracy[i]['correct']) + len(accuracy[i]['incorrect'])) for i in range(5))
        sem_acc = 100 * float(sem_correct) / sem_total

        syn_correct = sum((len(accuracy[i]['correct']) for i in range(5, len(accuracy) - 1)))
        syn_total = sum((len(accuracy[i]['correct']) + len(accuracy[i]['incorrect'])) for i in range(5, len(accuracy) - 1))
        syn_acc = 100 * float(syn_correct) / syn_total

        sum_corr = len(accuracy[-1]['correct'])
        sum_incorr = len(accuracy[-1]['incorrect'])
        total = sum_corr + sum_incorr
        total_acc = sum_corr / total * 100

        labels = ['sem_acc', '#sem', 'syn_acc', '#syn', 'total_acc', '#total']
        results = [sem_acc, sem_total, syn_acc, syn_total, total_acc, total]
        return labels, results