import string
import os
from collections import Counter
import configparser
from multiprocessing import Pool
import sys
sys.path.insert(0, '../common/')
import common
import multi_processing

config = configparser.ConfigParser()
config.read('config.ini')

'''
Attention:
Each time we create a local dictionary, word order will not be the same (word id is identical).
So each time the merged dictionary will be different: Each time a word may have different id in the merged dictionary.
'''
# TODO NOW explain the meaning of "local" and "transferred".


def multiprocessing_write_local_encoded_text_and_local_dict(data_folder, file_extension, dicts_folder, process_num,
                                                            data_type):
    """1st multiprocessing
    Get dictionary and encoded text of each origin file
    """
    def write_encoded_text_and_local_dict_for_xml(file_path, output_folder):
        """For data in /vol/corpusiles/restricted/ldc/ldc2008t25/data/xin_eng
        """
        print('Processing file %s (%s)...' % (file_path, multi_processing.get_pid()))

        word2id = dict()  # key: word <-> value: index
        id2word = dict()
        encoded_text = []
        puncs = set(string.punctuation)

        for paragraph in common.search_all_specific_nodes_in_xml_known_node_path(file_path,
                                                                                 config['input data']['xml_node_path']):
            for sent in common.tokenize_informal_paragraph_into_sentences(paragraph):
                encoded_sent = []

                # update the dictionary
                for word in common.tokenize_text_into_words(sent, "WordPunct"):

                    # Remove numbers
                    if word.isnumeric():
                        # TODO Maybe distinguish some meaningful numbers, like year
                        continue

                    # Remove punctuations
                    # if all(j.isdigit() or j in puncs for j in word):
                    if all(c in puncs for c in word):
                        continue

                    # Stem word
                    word = common.stem_word(word)

                    # Make all words in lowercase
                    word = word.lower()

                    if word not in word2id:
                        id = len(word2id)
                        word2id[word] = id
                        id2word[id] = word
                    encoded_sent.append(word2id[word])
                encoded_text.append(encoded_sent)

        file_basename = multi_processing.get_file_name(file_path)
        # Write the encoded_text
        if not output_folder.endswith('/'):
            output_folder += '/'
        common.write_list_to_pickle(encoded_text, output_folder + "encoded_text_" + file_basename + ".pickle")
        # Write the dictionary
        common.write_dict_to_file(output_folder + "dict_" + file_basename + ".dicloc", word2id, 'str')

    def write_encoded_text_and_local_dict_for_txt(file_path, output_folder):
        """For data in /vol/corpusiles/open/Wikipedia-Dumps/en/20170420/prep/ (Each line of txt file is one sentence.)
        """

        def sentences():
            for line in open(file_path, 'r', encoding='utf-8'):
                yield line

        print('Processing file %s (%s)...' % (file_path, multi_processing.get_pid()))

        word2id = dict()  # key: word <-> value: index
        id2word = dict()
        encoded_text = []
        puncs = set(string.punctuation)

        for sent in sentences():
            encoded_sent = []
            # update the dictionary
            for word in common.tokenize_text_into_words(sent, "WordPunct"):
                # Remove numbers
                if config.getboolean("graph", "remove_numbers") and word.isnumeric():
                    # TODO Maybe distinguish some meaningful numbers, like year
                    continue
                # Remove punctuations
                # if all(j.isdigit() or j in puncs for j in word):
                if config.getboolean("graph", "remove_punctuations"):
                    if all(c in puncs for c in word):
                        continue
                # Stem word
                if config.getboolean("graph", "stem_word"):
                    word = common.stem_word(word)
                # Make all words in lowercase
                if config.getboolean("graph", "lowercase"):
                    word = word.lower()
                if word not in word2id:
                    id = len(word2id)
                    word2id[word] = id
                    id2word[id] = word
                encoded_sent.append(word2id[word])
            encoded_text.append(encoded_sent)

        file_basename = multi_processing.get_file_name(file_path)
        # names like "AA", "AB", ...
        parent_folder_name = multi_processing.get_file_folder(file_path).split('/')[-1]
        # Write the encoded_text
        if not output_folder.endswith('/'):
            output_folder += '/'
        common.write_list_to_pickle(encoded_text,
                                    output_folder + "encoded_text_" + parent_folder_name + "_" + file_basename + ".pickle")
        # Write the dictionary
        write_dict_to_file(output_folder + "dict_" + parent_folder_name + "_" + file_basename + ".dicloc", word2id)

    kw = {'output_folder': dicts_folder}
    if data_type is 'txt':
        multi_processing.master(files_getter=multi_processing.get_files_endswith_in_all_subfolders,
                                data_folder=data_folder,
                                file_extension=file_extension,
                                worker=write_encoded_text_and_local_dict_for_txt,
                                process_num=process_num,
                                **kw)
    elif data_type is 'xml':
        multi_processing.master(files_getter=multi_processing.get_files_endswith_in_all_subfolders,
                                data_folder=data_folder,
                                file_extension=file_extension,
                                worker=write_encoded_text_and_local_dict_for_xml,
                                process_num=process_num,
                                **kw)


def merge_local_dict(dict_folder, output_folder):
    def read_first_column_file_to_build_set(file):
        d = set()
        with open(file, encoding='utf-8') as f:
            for line in f:
                (key, val) = line.rstrip('\n').split("\t")
                d.add(key)
        return d

    # Take all files in the folder starting with "dict_"
    files = [os.path.join(dict_folder, name) for name in os.listdir(dict_folder)
             if (os.path.isfile(os.path.join(dict_folder, name))
                 and name.startswith("dict_"))]
    all_keys = set()
    for file in files:
        all_keys |= read_first_column_file_to_build_set(file)

    result = dict(zip(all_keys, range(len(all_keys))))
    write_dict_to_file(output_folder + 'dict_merged.txt', result)
    return result


def multiprocessing_write_transferred_edges_files_and_transferred_word_count(local_dicts_folder, edges_folder,
                                                                             max_window_size, process_num):
    """2nd multiprocessing
    Build a transfer dict (by local dictionary and merged dictionary)
    and write a new encoded text by using the transfer dict.
    """

    def get_local_edges_files_and_local_word_count(local_dict_file_path, output_folder, max_window_size):
        def word_count(encoded_text, file_name):
            result = dict(Counter([item for sublist in encoded_text for item in sublist]))
            folder_name = multi_processing.get_file_folder(local_dict_file_path)
            common.write_dict_to_file(folder_name + "/word_count_" + file_name + ".txt", result, 'str')
            return result

        def get_transfer_dict_for_local_dict(local_dict, merged_dict):
            """
            local_dict:
                "hello": 37
            merged_dict:
                "hello": 52
            transfer_dict:
                37: 52
            """
            transfer_dict = {}
            for key, value in local_dict.items():
                transfer_dict[value] = merged_dict[key]
            return transfer_dict

        def write_edges_of_different_window_size(encoded_text, file_basename, output_folder, max_window_size):
            edges = {}

            # Construct edges
            for i in range(2, max_window_size + 1):
                edges[i] = []
            for encoded_sent in encoded_text:
                sentence_len = len(encoded_sent)
                for start_index in range(sentence_len - 1):
                    if start_index + max_window_size < sentence_len:
                        max_range = max_window_size + start_index
                    else:
                        max_range = sentence_len

                    for end_index in range(1 + start_index, max_range):
                        current_window_size = end_index - start_index + 1
                        # encoded_edge = [encoded_sent[start_index], encoded_sent[end_index]]
                        encoded_edge = (encoded_sent[start_index], encoded_sent[end_index])
                        edges[current_window_size].append(encoded_edge)

            # Write edges to files
            if not output_folder.endswith('/'):
                output_folder += '/'
            for i in range(2, max_window_size + 1):
                common.write_list_to_file(
                    output_folder + file_basename + "_encoded_edges_window_size_{0}.txt".format(i), edges[i])

        print('Processing file %s (%s)...' % (local_dict_file_path, multi_processing.get_pid()))

        merged_dict = read_two_columns_file_to_build_dictionary_type_specified(
            file=config['graph']['dicts_and_encoded_texts_folder'] + 'dict_merged.txt', key_type=str, value_type=int)
        local_dict = read_two_columns_file_to_build_dictionary_type_specified(local_dict_file_path, str, int)
        transfer_dict = get_transfer_dict_for_local_dict(local_dict, merged_dict)
        '''
        Local dict and local encoded text must be in the same folder,
        and their names should be look like below:
            local_dict_file_path:       dict_xin_eng_200410.txt
            local_encoded_text_pickle:  pickle_encoded_text_xin_eng_200410
        '''
        # Get encoded_text_pickle path according to local_dict_file_path
        local_encoded_text_pickle = local_dict_file_path.replace("dict_", "encoded_text_")[
                                    :-len(config['graph']['local_dict_extension'])]
        local_encoded_text = common.read_pickle_to_build_list(local_encoded_text_pickle + ".pickle")
        # Translate the local encoded text with transfer_dict
        transferred_encoded_text = []
        for encoded_sent in local_encoded_text:
            transfered_encoded_sent = []
            for encoded_word in encoded_sent:
                transfered_encoded_sent.append(transfer_dict[encoded_word])
            transferred_encoded_text.append(transfered_encoded_sent)

        # TODO Have to write the transferred_encoded_text?

        file_name = multi_processing.get_file_name(local_dict_file_path).replace("dict_", "")
        # Word count
        word_count(transferred_encoded_text, file_name)
        # Write edges files of different window size based on the transfered encoded text
        write_edges_of_different_window_size(transferred_encoded_text, file_name, output_folder, max_window_size)

    kw = {'output_folder': edges_folder, 'max_window_size': max_window_size}
    multi_processing.master(files_getter=multi_processing.get_files_endswith,
                            data_folder=local_dicts_folder,
                            file_extension=config['graph']['local_dict_extension'],
                            worker=get_local_edges_files_and_local_word_count,
                            process_num=process_num,
                            **kw)


def merge_transferred_word_count(word_count_folder=config['graph']['dicts_and_encoded_texts_folder'],
                                 output_folder=config['graph']['dicts_and_encoded_texts_folder']):
    files = multi_processing.get_files_startswith(word_count_folder, "word_count_")
    c = Counter()
    for file in files:
        counter_temp = common.read_two_columns_file_to_build_dictionary_type_specified(file, int, int)
        c += counter_temp
    common.write_dict_to_file(output_folder + "word_count_all.txt", dict(c), 'str')
    return dict(c)


def write_valid_vocabulary(
        merged_word_count_path=config['graph']['dicts_and_encoded_texts_folder'] + 'word_count_all.txt',
        output_path=config['graph']['dicts_and_encoded_texts_folder'] + 'valid_vocabulary_min_count_' + config['graph'][
            'min_count'] + '.txt',
        min_count=int(config['graph']['min_count'])):
    merged_word_count = read_two_columns_file_to_build_dictionary_type_specified(file=merged_word_count_path,
                                                                                 key_type=str, value_type=int)
    valid_vocabulary = []
    for word_id, count in merged_word_count.items():
        if count >= min_count:
            valid_vocabulary.append(word_id)
    common.write_simple_list_to_file(output_path, valid_vocabulary)


def get_counted_edges_worker(edges_files_paths):
    def read_valid_vocabulary(file_path=config['graph']['dicts_and_encoded_texts_folder'] +
                                        'valid_vocabulary_min_count_' + config['graph']['min_count'] + '.txt'):
        result = []
        with open(file_path) as f:
            for line in f:
                line_element = line.rstrip('\n')
                result.append(line_element)
        return result

    def counters_yielder():
        def read_edges_file_with_respect_to_valid_vocabulary(file_path, valid_vocabulary_list):
            d = []
            with open(file_path) as f:
                for line in f:
                    (first, second) = line.rstrip('\n').split("\t")
                    if (first in valid_vocabulary_list) and (second in valid_vocabulary_list):
                        d.append((first, second))
            return d

        for file in edges_files_paths:
            yield Counter(dict(Counter(
                read_edges_file_with_respect_to_valid_vocabulary(file_path=file, valid_vocabulary_list=valid_vocabulary))))

    valid_vocabulary = dict.fromkeys(read_valid_vocabulary())
    total = len(edges_files_paths)
    print(total, "files to be counted.")
    count = 1
    counted_edges = Counter(dict())
    for c in counters_yielder():
        counted_edges += c
        print('%i/%i files processed.' % (count, total), end='\r', flush=True)
        count += 1
    common.write_dict_to_file(config['graph']['edges_folder'] + str(multi_processing.get_pid()) + ".txt",
                              counted_edges, 'tuple')


def multiprocessing_merge_edges_count_of_a_specific_window_size(window_size, process_num,
                                                                edges_folder=config['graph']['edges_folder'],
                                                                output_folder=config['graph']['edges_folder']):
    def counted_edges_from_worker_yielder(folder=edges_folder):
        def read_counted_edges_from_worker(file_path):
            d = {}
            with open(file_path) as f:
                for line in f:
                    elements = line.rstrip('\n').split("\t")
                    key = (elements[0], elements[1])
                    val = int(elements[2])
                    d[key] = val
            return d

        paths = multi_processing.get_files_paths_not_contain(data_folder=folder, not_contain='encoded_edges')
        for path in paths:
            yield Counter(read_counted_edges_from_worker(path))

    # # Get all target edges files' paths to be merged and counted.
    # files = []
    # for i in range(2, window_size + 1):
    #     files_to_add = multi_processing.get_files_endswith(edges_folder, "_encoded_edges_window_size_{0}.txt".format(i))
    #     if not files_to_add:
    #         print('No encoded edges file of window size ' + str(window_size) + '. Reset window size to ' + str(
    #             i - 1) + '.')
    #         window_size = i - 1
    #         break
    #     else:
    #         files.extend(files_to_add)
    #
    # # Each thread processes several target edges files and save their counted_edges.
    # files_list = multi_processing.chunkify(files, process_num)
    with Pool(process_num) as p:
        # p.map(get_counted_edges_worker, files_list)
        # print('All sub-processes done.')

        # Merge all counted_edges from workers and get the final result.
        count = 1
        counted_edges = Counter(dict())
        for c in counted_edges_from_worker_yielder():
            counted_edges += c
            print('%i/%i files processed.' % (count, process_num), end='\r', flush=True)
            count += 1
        common.write_dict_to_file(output_folder + "encoded_edges_count_window_size_" + str(window_size) + ".txt",
                                  counted_edges, 'tuple')
        # Remove all counted_edges from workers.
        files_paths = multi_processing.get_files_paths_not_contain(data_folder=edges_folder,
                                                                   not_contain='encoded_edges')
        for file_path in files_paths:
            print('Remove file %s' % file_path)
            os.remove(file_path)


def write_dict_to_file(file_path, dictionary):
    f = open(file_path, 'w', encoding='utf-8')
    for key, value in dictionary.items():
        f.write('%s\t%s\n' % (key, value))


def read_two_columns_file_to_build_dictionary_type_specified(file, key_type, value_type):
    """
    file:
        en-000000001    Food waste or food loss is food that is discarded or lost uneaten.

    Output:
        {'en-000000001': 'Food waste or food loss is food that is discarded or lost uneaten.'}
    """
    d = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            (key, val) = line.rstrip('\n').split("\t")
            d[key_type(key)] = value_type(val)
    return d


def multiprocessing_all(data_folder, file_extension,
                        max_window_size,
                        process_num, data_type,
                        dicts_folder=config['graph']['dicts_and_encoded_texts_folder'],
                        edges_folder=config['graph']['edges_folder']):
    multiprocessing_write_local_encoded_text_and_local_dict(data_folder, file_extension, dicts_folder, process_num,
                                                            data_type)
    # Get one merged dictionary from all local dictionaries
    merge_local_dict(dict_folder=dicts_folder, output_folder=dicts_folder)
    multiprocessing_write_transferred_edges_files_and_transferred_word_count(dicts_folder, edges_folder, max_window_size, process_num)


# TODO LATER Add weight according to word pair distance in write_edges_of_different_window_size function
# TODO LATER remove edges by their frequency
# TODO LATER remove words by their frequency


# TESTS
# write_edges_of_different_window_size([[0, 11, 12, 13, 14, 15, 3, 16, 17], [1, 2, 3]], 5)


# # One core test (local dictionaries ready)
# # xml
# write_encoded_text_and_local_dict_for_xml("data/test_input_data/test_for_graph_builder_igraph_multiprocessing.xml", 'data/dicts_and_encoded_texts/', "./DOC/TEXT/P")
# merged_dict = merge_local_dict(dict_folder='data/dicts_and_encoded_texts/', output_folder='data/dicts_and_encoded_texts/')
# get_local_edges_files_and_local_word_count('data/dicts_and_encoded_texts/dict_test_for_graph_builder_igraph_multiprocessing.dicloc',
#                                            merged_dict, 'data/edges/', max_window_size=10, local_dict_extension='.dicloc')
# merge_transferred_word_count(word_count_folder='data/dicts_and_encoded_texts/', output_folder='data/dicts_and_encoded_texts/')
# multiprocessing_merge_edges_count_of_a_specific_window_size(edges_folder='data/edges/', window_size=4, output_folder='data/')

# # txt
# write_encoded_text_and_local_dict_for_txt(
#     file_path="data/training data/Wikipedia-Dumps_en_20170420_prep/AA/wiki_01.txt",
#     output_folder='output/intermediate data/dicts_and_encoded_texts')


# # Multiprocessing test
# # xml
# multiprocessing_all(xml_data_folder='/Users/zzcoolj/Code/GoW/data/test_input_data/xin_eng_for_test',
#                     xml_file_extension='.xml',
#                     xml_node_path='./DOC/TEXT/P',
#                     dicts_folder='data/dicts_and_encoded_texts/',
#                     local_dict_extension='.dicloc',
#                     edges_folder='data/edges/',
#                     max_window_size=3,
#                     process_num=3)
# merge_transferred_word_count(word_count_folder='data/dicts_and_encoded_texts/', output_folder='data/dicts_and_encoded_texts/')
# multiprocessing_merge_edges_count_of_a_specific_window_size(edges_folder='data/edges/', window_size=4, output_folder='data/')

# txt
# multiprocessing_all(data_folder='data/training data/Wikipedia-Dumps_en_20170420_prep/',
#                     file_extension='.txt',
#                     max_window_size=3,
#                     process_num=4,
#                     data_type='txt')
# merge_transferred_word_count()
# write_valid_vocabulary()
multiprocessing_merge_edges_count_of_a_specific_window_size(window_size=50, process_num=6)
