import graph_data_provider_multiprocessing
import nltk.data
nltk.data.path = ['/vol/datailes/tools/nlp/nltk_data/2016']


graph_data_provider_multiprocessing.multiprocessing_all(
    data_folder='/vol/corpusiles/open/Wikipedia-Dumps/en/20170420/prep/',
    file_extension='.txt',
    max_window_size=5,
    process_num=16,
    worker=graph_data_provider_multiprocessing.write_encoded_text_and_local_dict_for_txt)
