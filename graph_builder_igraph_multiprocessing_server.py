import graph_builder_igraph_multiprocessing

import nltk.data
nltk.data.path = ['/vol/datailes/tools/nlp/nltk_data/2016']


# graph_builder_igraph_multiprocessing.multiprocessing_all(
#     xml_data_folder='/vol/corpusiles/restricted/ldc/ldc2008t25/data/xin_eng',
#     xml_file_extension='.xml',
#     xml_node_path='./DOC/TEXT/P',
#     dicts_folder='data/dicts_and_encoded_texts/',
#     local_dict_extension='.dicloc',
#     edges_folder='data/edges/',
#     max_window_size=3,
#     process_num=10)
graph_builder_igraph_multiprocessing.multiprocessing_all(
    data_folder='/vol/corpusiles/open/Wikipedia-Dumps/en/20170420/prep/AA',
    file_extension='.txt',
    dicts_folder='data/dicts_and_encoded_texts/',
    local_dict_extension='.dicloc',
    edges_folder='data/edges/',
    max_window_size=3,
    process_num=10,
    worker=graph_builder_igraph_multiprocessing.write_encoded_text_and_local_dict_for_txt)
graph_builder_igraph_multiprocessing.merge_edges_count_of_a_specific_window_size(edges_folder='data/edges/', window_size=4, output_folder='data/')
G, mergedDict = graph_builder_igraph_multiprocessing.build_graph('data/dicts_and_encoded_texts/merged_dict.txt', 'data/counted_edges.txt')
graph_builder_igraph_multiprocessing.calculate_k_core_and_save_graph(G, mergedDict)