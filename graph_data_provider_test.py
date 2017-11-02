import unittest
import graph_data_provider_multiprocessing as gdp


class TestGraphDataProvider(unittest.TestCase):
    data_folder = 'data/training data/unittest_data/'
    file_extension = '.txt'
    max_window_size = 6
    process_num = 4
    data_type = 'txt'
    dicts_folder = 'output/intermediate data for unittest/dicts_and_encoded_texts/'
    edges_folder = 'output/intermediate data for unittest/edges/'
    min_count = 5

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

    def test_write_valid_vocabulary(self):
        result = gdp.write_valid_vocabulary(merged_word_count_path=self.dicts_folder + 'word_count_all.txt',
                                            output_path=self.dicts_folder + 'valid_vocabulary_min_count_' + str(
                                                self.min_count) + '.txt',
                                            min_count=self.min_count)
        self.assertEqual(len(result), 6)
        result = gdp.write_valid_vocabulary(merged_word_count_path=self.dicts_folder + 'word_count_all.txt',
                                            output_path=self.dicts_folder + 'valid_vocabulary_min_count_' + str(
                                                self.min_count) + '.txt',
                                            min_count=1)
        self.assertEqual(len(result), 94)

    def test_multiprocessing_merge_edges_count_of_a_specific_window_size(self):
        result = gdp.multiprocessing_merge_edges_count_of_a_specific_window_size(window_size=50, process_num=5,
                                                                                 min_count=self.min_count,
                                                                                 dicts_folder=self.dicts_folder,
                                                                                 edges_folder=self.edges_folder,
                                                                                 output_folder=self.edges_folder)


if __name__ == '__main__':
    unittest.main()
