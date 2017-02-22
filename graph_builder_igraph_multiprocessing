import string

import sys
sys.path.insert(0, '../common/')
import common
import multi_processing


word2id = dict()  # key: word <-> value: index
id2word = dict()


def get_encoded_text(file_path, output_folder, node_path):
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

    # Write the dictionary
    file_basename = multi_processing.get_file_name(file_path)
    common.write_dict_to_file(output_folder+"dict_"+file_basename+".txt", word2id)
    return encoded_text, file_basename


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


def worker(file_path, output_folder):
    # TODO attention output_folder should contain "/" in the end
    print('Processing file %s (%s)...' % (file_path, multi_processing.get_pid()))
    encoded_text, file_basename = get_encoded_text(file_path, output_folder, "./DOC/TEXT/P")
    write_edges_of_different_window_size(encoded_text, file_basename, output_folder, 3)


# TESTS
# print(get_encoded_text("/Users/zzcoolj/Code/GoW/data/aquaint-2_sample_xin_eng_200512.xml", "./DOC/TEXT/P"))
# print(get_encoded_text("data/test_for_graph_builder_igraph_multiprocessing.xml", "./DOC/TEXT/P"))

# write_edges_of_different_window_size([[0, 11, 12, 13, 14, 15, 3, 16, 17], [1, 2, 3]], 5)

# write_edges_of_different_window_size(get_encoded_text("/Users/zzcoolj/Code/GoW/data/aquaint-2_sample_xin_eng_200512.xml", "./DOC/TEXT/P"), 5)

multi_processing.master("/Users/zzcoolj/Code/GoW/data/xin_eng_for_test", ".xml", "data/", worker, process_num=3)
