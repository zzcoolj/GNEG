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
    for i in range(2, max_window_size+1):
        # TODO add folder path before file name
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
def get_transfered_encoded_text(local_dict_file_path, merged_dict, output_folder, max_window_size):

# Solution 2
# def get_transfered_encoded_text(local_dict_file_path, *merged_dict, output_folder, max_window_size):

    print('Processing2 file %s (%s)...' % (local_dict_file_path, multi_processing.get_pid()))

    local_dict = common.read_two_columns_file_to_build_dictionary(local_dict_file_path)
    transfer_dict = get_transfer_dict_for_local_dict(local_dict, merged_dict)

    '''
    Local dict and local encoded text must be in the same folder,
    and their names should be look like below:
        local_dict_file_path:            /Users/zzcoolj/Code/GoW/data/dict_xin_eng_200410.txt
        local_encoded_text_pickle:  /Users/zzcoolj/Code/GoW/data/pickle_encoded_text_xin_eng_200410
    '''
    # Get encoded_text_pickle path according to local_dict_file_path
    local_encoded_text_pickle = local_dict_file_path.replace("dict", "pickle_encoded_text")[:-7]
    local_encoded_text = common.read_pickle_to_build_list(local_encoded_text_pickle)

    # Translate the local encoded text with transfer_dict
    transfered_encoded_text = []
    for encoded_sent in local_encoded_text:
        transfered_encoded_sent = []
        for encoded_word in encoded_sent:
            # TODO dict{int: int}
            transfered_encoded_sent.append(transfer_dict[str(encoded_word)])
        transfered_encoded_text.append(transfered_encoded_sent)

    # Write edges files of different window size based on the transfered encoded text
    file_basename = multi_processing.get_file_name(local_dict_file_path)
    write_edges_of_different_window_size(transfered_encoded_text, file_basename, output_folder, max_window_size)


def multiprocessing_get_merged_dict_and_edges_files(data_folder, file_extension, output_folder, node_path):
    # TODO attention output_folder should contain "/" in the end

    # 1st multiprocessing: Get dictionary and encoded text of each origin file
    kw = {'output_folder': output_folder, 'node_path': node_path}
    multi_processing.master(data_folder,
                            file_extension,
                            write_encoded_text_and_local_dict,
                            process_num=3,
                            **kw)

    # Get one merged dictionary from all local dictionaries
    merged_dict = merge_dict(output_folder)
    common.write_dict_to_file(output_folder+"merged_dict.txt", merged_dict)

    # TODO split into two functions, so as to I can test 2nd multiprocessing directly
    # 2nd multiprocessing: Build a transfer dict (by local dictionary and merged dictionary)
    #                       and write a new encoded text by using the transfer dict.
    # TODO build a list of merged_dict
    files_number = len(multi_processing.get_files(data_folder, file_extension))
    merged_dicts = []
    for i in range(files_number):
        merged_dicts.append(merged_dict.copy())

    kw2 = {'output_folder': output_folder, 'max_window_size': 3}
    multi_processing.master2(output_folder,
                             ".dicloc",
                             merged_dicts,
                             get_transfered_encoded_text,
                             process_num=3,
                             **kw2)

    # TODO for test
    # get_transfered_encoded_text("data/dict_xin_eng_200410.dicloc", merged_dict, output_folder, 3)




# TESTS
# print(write_encoded_text_and_local_dict("/Users/zzcoolj/Code/GoW/data/aquaint-2_sample_xin_eng_200512.xml", "./DOC/TEXT/P"))
# print(write_encoded_text_and_local_dict("data/test_for_graph_builder_igraph_multiprocessing.xml", "./DOC/TEXT/P"))

# write_edges_of_different_window_size([[0, 11, 12, 13, 14, 15, 3, 16, 17], [1, 2, 3]], 5)

# merge_dict("data/")

multiprocessing_get_merged_dict_and_edges_files(data_folder='/Users/zzcoolj/Code/GoW/data/xin_eng_for_test',
                                                file_extension='.xml',
                                                output_folder='data/',
                                                node_path='./DOC/TEXT/P')
