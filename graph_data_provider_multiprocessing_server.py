import graph_data_provider_multiprocessing
import nltk.data

nltk.data.path = ['/vol/datailes/tools/nlp/nltk_data/2016']

# graph_data_provider_multiprocessing.multiprocessing_all(
#     data_folder='/vol/corpusiles/open/Wikipedia-Dumps/en/20170420/prep/',
#     file_extension='.txt',
#     max_window_size=5,
#     process_num=30,
#     data_type='txt')
# graph_data_provider_multiprocessing.merge_transferred_word_count()
graph_data_provider_multiprocessing.write_valid_vocabulary()
graph_data_provider_multiprocessing.multiprocessing_merge_edges_count_of_a_specific_window_size(window_size=50, process_num=60)
