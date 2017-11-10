import graph_data_provider
import nltk.data

nltk.data.path = ['/vol/datailes/tools/nlp/nltk_data/2016']

# graph_data_provider.prepare_intermediate_data(
#     data_folder='/vol/corpusiles/open/Wikipedia-Dumps/en/20170420/prep/',
#     file_extension='.txt',
#     max_window_size=10,
#     process_num=50)
# graph_data_provider.multiprocessing_merge_edges_count_of_a_specific_window_size(window_size=5, process_num=50)
graph_data_provider.filter_edges(min_count=5,
                                 old_encoded_edges_count_path="output/intermediate data/graph/encoded_edges_count_window_size_5.txt")