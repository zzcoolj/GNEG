"""
ATTENTION: All results are based on setting preprocessing_word=False in config.ini.
"""

import unittest
import graph_data_provider as gdp


class TestGraphDataProvider(unittest.TestCase):
    data_folder = 'data/training data/unittest_data/'
    file_extension = '.txt'
    max_window_size = 6
    process_num = 4
    data_type = 'txt'
    dicts_folder = 'output/intermediate data for unittest/dicts_and_encoded_texts/'
    edges_folder = 'output/intermediate data for unittest/edges/'
    graph_folder = 'output/intermediate data for unittest/graph/'
    min_count = 5
    max_vocab_size = 3

    def test_merge_local_dict(self):
        gdp.multiprocessing_write_local_encoded_text_and_local_dict(self.data_folder, self.file_extension,
                                                                    self.dicts_folder, self.process_num)
        result = gdp.merge_local_dict(dict_folder=self.dicts_folder, output_folder=self.dicts_folder)
        self.assertEqual(len(result), 94)

    def test_merge_transferred_word_count(self):
        gdp.multiprocessing_write_transferred_edges_files_and_transferred_word_count(self.dicts_folder,
                                                                                     self.edges_folder,
                                                                                     self.max_window_size,
                                                                                     self.process_num)
        result = gdp.merge_transferred_word_count(word_count_folder=self.dicts_folder, output_folder=self.dicts_folder)
        self.assertEqual(len(result), 94)
        merged_dict = gdp.read_two_columns_file_to_build_dictionary_type_specified(
            file=self.dicts_folder + 'dict_merged.txt', key_type=str,
            value_type=int)
        self.assertEqual(result[merged_dict['on']], 2)
        self.assertEqual(result[merged_dict['00']], 3)
        self.assertEqual(result[merged_dict[',']], 5)

    def get_id2word(self):
        word2id = gdp.read_two_columns_file_to_build_dictionary_type_specified(
            file=self.dicts_folder + 'dict_merged.txt', key_type=str,
            value_type=int)
        id2word = dict()
        for word, id in word2id.items():
            id2word[id] = word
        return id2word

    def test_write_valid_vocabulary(self):
        result = gdp.write_valid_vocabulary(merged_word_count_path=self.dicts_folder + 'word_count_all.txt',
                                            output_path=self.dicts_folder + 'valid_vocabulary_min_count_' + str(
                                                self.min_count) + '.txt',
                                            min_count=self.min_count)
        self.assertEqual(len(result), 6)

        result = gdp.write_valid_vocabulary(merged_word_count_path=self.dicts_folder + 'word_count_all.txt',
                                            output_path=self.dicts_folder + 'valid_vocabulary_min_count_1.txt',
                                            min_count=1)
        self.assertEqual(len(result), 94)

        result = gdp.write_valid_vocabulary(merged_word_count_path=self.dicts_folder + 'word_count_all.txt',
                                            output_path=self.dicts_folder + 'valid_vocabulary_min_count_3.txt',
                                            min_count=3)
        self.assertEqual(len(result), 6 + 3)
        # id2word = self.get_id2word()
        # print([id2word[int(i)] for i in result])

    def test_multiprocessing_merge_edges_count_of_a_specific_window_size(self):
        gdp.write_valid_vocabulary(merged_word_count_path=self.dicts_folder + 'word_count_all.txt',
                                   output_path=self.dicts_folder + 'valid_vocabulary_min_count_' + str(
                                       self.min_count) + '.txt',
                                   min_count=self.min_count)
        result = gdp.multiprocessing_merge_edges_count_of_a_specific_window_size(window_size=50, process_num=5,
                                                                                 min_count=self.min_count,
                                                                                 dicts_folder=self.dicts_folder,
                                                                                 edges_folder=self.edges_folder,
                                                                                 output_folder=self.graph_folder)
        word2id = gdp.read_two_columns_file_to_build_dictionary_type_specified(
            file=self.dicts_folder + 'dict_merged.txt', key_type=str,
            value_type=int)
        print()
        self.assertEqual(result[(str(word2id['and']), str(word2id[',']))], 2)
        self.assertEqual(result[(str(word2id['and']), str(word2id['.']))], 2)
        self.assertEqual(result[(str(word2id['and']), str(word2id['the']))], 1)

        self.assertEqual(result[(str(word2id['the']), str(word2id['of']))], 6)
        self.assertEqual(result[(str(word2id['the']), str(word2id['.']))], 2)
        self.assertEqual(result[(str(word2id['the']), str(word2id['and']))], 3)
        self.assertEqual(result[(str(word2id['the']), str(word2id['in']))], 1)
        self.assertEqual(result[(str(word2id['the']), str(word2id[',']))], 2)

        self.assertEqual(result[(str(word2id['of']), str(word2id['.']))], 3)
        self.assertEqual(result[(str(word2id['of']), str(word2id['the']))], 2)
        self.assertEqual(result[(str(word2id['of']), str(word2id['and']))], 3)
        self.assertEqual(result[(str(word2id['of']), str(word2id['in']))], 2)
        self.assertEqual(result[(str(word2id['of']), str(word2id[',']))], 1)

        self.assertEqual(result[(str(word2id['in']), str(word2id['.']))], 1)
        self.assertEqual(result[(str(word2id['in']), str(word2id['the']))], 5)
        self.assertEqual(result[(str(word2id['in']), str(word2id['and']))], 1)
        self.assertEqual(result[(str(word2id['in']), str(word2id[',']))], 1)

        self.assertEqual(result[(str(word2id[',']), str(word2id['and']))], 2)
        self.assertEqual(result[(str(word2id[',']), str(word2id['in']))], 1)
        self.assertEqual(result[(str(word2id[',']), str(word2id['the']))], 1)

        self.assertEqual(len(result), 20 + 3)  # 3 self loops

    def test_filter_edges(self):
        gdp.write_valid_vocabulary(merged_word_count_path=self.dicts_folder + 'word_count_all.txt',
                                   output_path=self.dicts_folder + 'valid_vocabulary_min_count_' + str(
                                       self.min_count) + '.txt',
                                   min_count=self.min_count)
        gdp.multiprocessing_merge_edges_count_of_a_specific_window_size(window_size=50, process_num=5,
                                                                        min_count=self.min_count,
                                                                        dicts_folder=self.dicts_folder,
                                                                        edges_folder=self.edges_folder,
                                                                        output_folder=self.graph_folder)
        filtered_edges = gdp.filter_edges(min_count=self.min_count,
                                          old_encoded_edges_count_path=
                                          self.graph_folder + 'encoded_edges_count_window_size_6.txt',
                                          max_vocab_size=self.max_vocab_size,
                                          new_valid_vocabulary_folder=self.dicts_folder,
                                          merged_word_count_path=self.dicts_folder + 'word_count_all.txt',
                                          output_folder=self.graph_folder)

        word2id = gdp.read_two_columns_file_to_build_dictionary_type_specified(
            file=self.dicts_folder + 'dict_merged.txt', key_type=str,
            value_type=int)

        self.assertEqual(filtered_edges[(str(word2id['and']), str(word2id['the']))], 1)

        self.assertEqual(filtered_edges[(str(word2id['the']), str(word2id['of']))], 6)
        self.assertEqual(filtered_edges[(str(word2id['the']), str(word2id['and']))], 3)

        self.assertEqual(filtered_edges[(str(word2id['of']), str(word2id['the']))], 2)
        self.assertEqual(filtered_edges[(str(word2id['of']), str(word2id['and']))], 3)

        self.assertEqual(len(filtered_edges), 5 + 2)  # 2 self loops


if __name__ == '__main__':
    unittest.main()
