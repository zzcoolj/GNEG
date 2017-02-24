import string
import os

import sys
sys.path.insert(0, '../common/')
import common
import multi_processing


def write_encoded_text_and_local_dict(file_path, output_folder, node_path):
    print('Processing file %s (%s)...' % (file_path, multi_processing.get_pid()))

    word2id = dict()  # key: word <-> value: index
    id2word = dict()
    encoded_text = []
    puncs = set(string.punctuation)

    for paragraph in common.search_all_specific_nodes_in_xml_known_node_path(file_path, node_path):
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
    common.write_list_to_pickle(encoded_text, output_folder+"pickle_encoded_text_"+file_basename)
    # Write the dictionary
    common.write_dict_to_file(output_folder+"dict_"+file_basename+".dicloc", word2id)


def write_edges_of_different_window_size(encoded_text, file_basename, output_folder, max_window_size):
    edges = {}

    # Construct edges
    for i in range(2, max_window_size+1):
        edges[i] = []
    for encoded_sent in encoded_text:
        sentence_len = len(encoded_sent)
        for start_index in range(sentence_len-1):
            if start_index + max_window_size < sentence_len:
                max_range = max_window_size+start_index
            else:
                max_range = sentence_len

            for end_index in range(1+start_index, max_range):
                current_window_size = end_index - start_index + 1
                encoded_edge = [encoded_sent[start_index], encoded_sent[end_index]]
                edges[current_window_size].append(encoded_edge)


    # Write edges to files
    if not output_folder.endswith('/'):
        output_folder += '/'
    for i in range(2, max_window_size+1):
        common.write_list_to_file(output_folder+file_basename+"_encoded_edges_window_size_{0}.txt".format(i), edges[i])


def merge_dict(folder):
    # Take all files in the folder starting with "dict_"
    files = [os.path.join(folder, name) for name in os.listdir(folder)
             if (os.path.isfile(os.path.join(folder, name))
                 and name.startswith("dict_"))]
    all_keys = set()
    for file in files:
        all_keys |= common.read_first_column_file_to_build_set(file)

    merged_dict = dict(zip(all_keys, range(len(all_keys))))
    return merged_dict


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


# Solution 1
def get_transfered_encoded_text(local_dict_file_path, merged_dict, output_folder, max_window_size, local_dict_extension):

# Solution 2
# def get_transfered_encoded_text(local_dict_file_path, *merged_dict, output_folder, max_window_size):

    print('Processing2 file %s (%s)...' % (local_dict_file_path, multi_processing.get_pid()))

    local_dict = common.read_two_columns_file_to_build_dictionary_type_specified(local_dict_file_path, str, int)
    transfer_dict = get_transfer_dict_for_local_dict(local_dict, merged_dict)

    '''
    Local dict and local encoded text must be in the same folder,
    and their names should be look like below:
        local_dict_file_path:            /Users/zzcoolj/Code/GoW/data/dict_xin_eng_200410.txt
        local_encoded_text_pickle:  /Users/zzcoolj/Code/GoW/data/pickle_encoded_text_xin_eng_200410
    '''
    # Get encoded_text_pickle path according to local_dict_file_path
    local_encoded_text_pickle = local_dict_file_path.replace("dict_", "pickle_encoded_text_")[:-len(local_dict_extension)]
    local_encoded_text = common.read_pickle_to_build_list(local_encoded_text_pickle)

    # Translate the local encoded text with transfer_dict
    transfered_encoded_text = []
    for encoded_sent in local_encoded_text:
        transfered_encoded_sent = []
        for encoded_word in encoded_sent:
            transfered_encoded_sent.append(transfer_dict[encoded_word])
        transfered_encoded_text.append(transfered_encoded_sent)

    # TODO Have to write the transfered_encoded_text?

    # Write edges files of different window size based on the transfered encoded text
    file_basename = multi_processing.get_file_name(local_dict_file_path)
    write_edges_of_different_window_size(transfered_encoded_text, file_basename, output_folder, max_window_size)


def multiprocessing_write_encoded_text_and_local_dict(data_folder, file_extension, dicts_folder, xml_node_path, process_num):
    # 1st multiprocessing: Get dictionary and encoded text of each origin file
    kw = {'output_folder': dicts_folder, 'node_path': xml_node_path}
    multi_processing.master(data_folder,
                            file_extension,
                            write_encoded_text_and_local_dict,
                            process_num=process_num,
                            **kw)


def multiprocessing_get_edges_files(local_dicts_folder, local_dict_extension, edges_folder, merged_dict, max_window_size, process_num):
    # 2nd multiprocessing: Build a transfer dict (by local dictionary and merged dictionary)
    #                       and write a new encoded text by using the transfer dict.

    # Build a list of merged_dict. Each process could use its own merged dict, don't have to share memory.
    local_dicts_number = len(multi_processing.get_files(local_dicts_folder, local_dict_extension))
    merged_dicts = []
    for i in range(local_dicts_number):
        merged_dicts.append(merged_dict.copy())

    kw2 = {'output_folder': edges_folder, 'max_window_size': max_window_size, 'local_dict_extension': local_dict_extension}
    multi_processing.master2(local_dicts_folder,
                             local_dict_extension,
                             merged_dicts,
                             get_transfered_encoded_text,
                             process_num=process_num,
                             **kw2)


def multiprocessing_all(xml_data_folder, xml_file_extension, xml_node_path,
                        dicts_folder, local_dict_extension,
                        edges_folder, max_window_size,
                        process_num):
    multiprocessing_write_encoded_text_and_local_dict(xml_data_folder, xml_file_extension, dicts_folder, xml_node_path, process_num)

    # Get one merged dictionary from all local dictionaries
    merged_dict = merge_dict(dicts_folder)
    common.write_dict_to_file(dicts_folder + "merged_dict.txt", merged_dict)

    multiprocessing_get_edges_files(dicts_folder, local_dict_extension, edges_folder, merged_dict, max_window_size, process_num)


# TESTS
# print(write_encoded_text_and_local_dict("/Users/zzcoolj/Code/GoW/data/aquaint-2_sample_xin_eng_200512.xml", "./DOC/TEXT/P"))
# print(write_encoded_text_and_local_dict("data/test_for_graph_builder_igraph_multiprocessing.xml", "./DOC/TEXT/P"))

# write_edges_of_different_window_size([[0, 11, 12, 13, 14, 15, 3, 16, 17], [1, 2, 3]], 5)

# # One core test (local dictionaries ready)
# merged_dict = merge_dict('data/dicts/')
# common.write_dict_to_file('data/dicts/merged_dict.txt', merged_dict)
# get_transfered_encoded_text('data/dicts/dict_xin_eng_200410.dicloc', merged_dict, 'data/edges/', 3, '.dicloc')

# Multiprocessing test
multiprocessing_all(xml_data_folder='/Users/zzcoolj/Code/GoW/data/xin_eng_for_test',
                    xml_file_extension='.xml',
                    xml_node_path='./DOC/TEXT/P',
                    dicts_folder='data/dicts_and_encoded_texts/',
                    local_dict_extension='.dicloc',
                    edges_folder='data/edges/',
                    max_window_size=3,
                    process_num=3)


