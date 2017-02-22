import graph_builder_igraph_multiprocessing

import nltk.data
nltk.data.path = ['/vol/datailes/tools/nlp/nltk_data/2016']

graph_builder_igraph_multiprocessing.multi_processing.master("/vol/corpusiles/restricted/ldc/ldc2008t25/data/xin_eng", 
".xml", "data/", graph_builder_igraph_multiprocessing.worker, process_num=10)
