from gensim import utils
from scipy import stats


class StatisticalSignificance(object):

    def __init__(self, keyedVectors):
        self.keyedVectors = keyedVectors

    def evaluate_word_pairs(self, pairs, delimiter='\t', restrict_vocab=300000):
        ok_vocab = [(w, self.keyedVectors.vocab[w]) for w in self.keyedVectors.index2word[:restrict_vocab]]
        ok_vocab = dict((w.upper(), v) for w, v in reversed(ok_vocab))

        similarity_gold = []
        similarity_model = []
        oov = 0

        original_vocab = self.keyedVectors.vocab
        self.keyedVectors.vocab = ok_vocab

        print("OK1")
        for line_no, line in enumerate(utils.smart_open(pairs)):
            print("OK2")
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
        spearman = stats.spearmanr(similarity_gold, similarity_model)
        pearson = stats.pearsonr(similarity_gold, similarity_model)
        oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100

        return pearson, spearman, oov_ratio