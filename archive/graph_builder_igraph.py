import igraph
import core_dec
from itertools import islice
import string
import pickle
import time
import operator
import sys
sys.path.insert(0, '../common/')
import common
# sys.path.insert(0, '/Users/zzcoolj/anaconda/pkgs')
# sys.path.insert(0, '/Users/zzcoolj/anaconda/lib')
# sys.path.insert(0, '/Users/zzcoolj/anaconda/include')
# import cairo


window_size = 10

# Initialize graph
# Graph is weighted, directed
G = igraph.Graph()

word2id = dict()  # key: word <-> value: index
id2word = dict()


'''
The difference between slide_window_across_a_sentence1 and slide_window_across_a_sentence2 is that:
In slide_window_across_a_sentence1, we connect not only first word to the others but also second word, third word...
e.g. window_size = 3 sentence = [A B C D]
    In the first window [A, B, C], we have connections A->B, A->C & B->C
    In the second window [B, C, D], we have B->C, B->D, C->D
    B->C has been counted twice.

In slide_window_across_a_sentence2, B->C will be counted only once in the second window [B, C, D].
slide_window_across_a_sentence2 is more appropriate
'''


# Sentence has already been encoded, it's a list of vertex indexes.
def slide_window_across_a_sentence1(sentence):

    # window is a list of vertex indexes.
    def extract_graph_information(window_content):
        window_len = len(window_content)
        for start_index in range(window_len):
            for end_index in range(start_index + 1, window_len):
                # We don't care self-loop edges
                if not window_content[start_index] == window_content[end_index]:
                    if G.are_connected(window_content[start_index], window_content[end_index]):
                        G[window_content[start_index], window_content[end_index], "weight"] += 1
                    else:
                        G.add_edge(window_content[start_index], window_content[end_index])
                        G[window_content[start_index], window_content[end_index], "weight"] = 1

    # It is a generator(because it contains yield), not a function
    def window(seq, n):
        # Returns a sliding window (of width n) over data from the iterable
        #   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    if len(sentence) <= window_size:
        extract_graph_information(sentence)
    else:
        for window_content in window(sentence, window_size):
            print("window_content ->", window_content)
            extract_graph_information(window_content)


def slide_window_across_a_sentence2(sentence):
    sentence_len = len(sentence)
    for start_index in range(sentence_len):
        if start_index + window_size < sentence_len:
            for end_index in range(start_index + 1, (start_index+window_size)):
                # We don't care self-loop edges
                if not sentence[start_index] == sentence[end_index]:
                    if G.are_connected(sentence[start_index], sentence[end_index]):
                        G[sentence[start_index], sentence[end_index], "weight"] += 1
                    else:
                        G.add_edge(sentence[start_index], sentence[end_index])
                        G[sentence[start_index], sentence[end_index], "weight"] = 1
        else:
            for end_index in range(start_index + 1, sentence_len):
                # We don't care self-loop edges
                if not sentence[start_index] == sentence[end_index]:
                    if G.are_connected(sentence[start_index], sentence[end_index]):
                        G[sentence[start_index], sentence[end_index], "weight"] += 1
                    else:
                        G.add_edge(sentence[start_index], sentence[end_index])
                        G[sentence[start_index], sentence[end_index], "weight"] = 1


# Input is a list of sentences created by tokenize_text_into_sentences() of common.py
def graph_builder(file_path):
    text = common.tokenize_text_into_sentences(file_path)
    for sent in text:
        encoded_sent = []

        # TODO It's really stupid to remove punctuations here, checkout multiprocessing version.
        # TODO remove punctuation, may not so efficient
        translator = str.maketrans('', '', string.punctuation)
        sent = sent.translate(translator)

        # update the dictionary
        for word in common.tokenize_text_into_words(sent):
            # TODO All words are in lowercase so far, may change after
            word = word.lower()
            if word not in word2id:
                id = len(word2id)
                word2id[word] = id
                id2word[id] = word
                # add new vertex into graph
                G.add_vertex(word)
            encoded_sent.append(word2id[word])

        # update the graph
        slide_window_across_a_sentence2(encoded_sent)


# def draw_graph():
#     nx.draw(G, with_labels=True)
#     plt.show()


def show_detailed_information():
    # print(word2id, "\n************\n",  id2word)
    # print(G)
    # print("degree ->", G.degree(G.vs))
    # print("strength ->", G.strength(G.vs, weights=G.es["weight"]))
    print(G.vs["name"])
    # print(G.vs["k_core"])
    print("k-core weighted ->", sorted_cores_g)


def get_k_core(graph):
    graph.vs["weight"] = graph.strength(graph.vs, weights=graph.es["weight"])
    # print("k-core unweighted ->", core_dec.core_dec(graph, weighted=False))
    return core_dec.core_dec(graph)


def save_sorted_cores_dictionary(sorted_cores_dictionary):
    with open('data/sorted_cores_g.pickle', 'wb') as handle:
        pickle.dump(sorted_cores_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_graph_pickle(graph):
    graph.write_pickle("data/graph.pickle")


def add_k_core_information_to_graph(graph, k_core_list):
    for name, core in k_core_list:
        graph.vs.select(word2id[name])["k_core"] = core


# TODO module(cairo + py3cairo) cairo doesn't work in mac.
# TODO The best guess is that py3cairo, a Python bindings(interface) of cairo is not well configured
# TODO Try draw_graph later in linux or Windows, it's not so important now...
# def draw_graph(g):
#     # g = igraph.Graph([(0, 1), (0, 2), (2, 3), (3, 4), (4, 2), (2, 5), (5, 0), (6, 3), (5, 6)])
#     # igraph.plot(g, "result.pdf", layout=g.layout_kamada_kawai())


def save_graph_svg(g):
    def gen_hex_colour_code(seed):
        import random
        random.seed(seed)
        return "#" + ''.join([random.choice('0123456789ABCDEF') for x in range(6)])

    visual_style = dict()
    visual_style["layout"] = g.layout("kk")
    visual_style["width"] = 2000
    visual_style["height"] = 2000
    visual_style["labels"] = "name"
    visual_style["colors"] = [gen_hex_colour_code(k_core) for k_core in g.vs["k_core"]]
    # visual_style["edge_color"] = ["#D6CFCE" * g.vcount()]
    visual_style["edge_colors"] = ["gray"] * g.ecount()
    g.write_svg("data/test.svg", **visual_style)


start_time = time.time()
graph_builder("/Users/zzcoolj/Code/GoW/data/test_for_graph_builder_igraph.txt")
# graph_builder("../word2vec/data/text8")
print("[graph_builder] done:", common.count_time(start_time))
sorted_cores_g = get_k_core(G)
print("[k_core] done:", common.count_time(start_time))
save_sorted_cores_dictionary(sorted_cores_g)
print("[save_sorted_cores_dictionary] done:", common.count_time(start_time))
add_k_core_information_to_graph(G, sorted_cores_g)
print("[add_k_core_information_to_graph] done:", common.count_time(start_time))
save_graph_pickle(G)
print("[save_graph_pickle] done:", common.count_time(start_time))
save_graph_svg(G)
print("[save_graph_svg] done:", common.count_time(start_time))
print("[MISSION COMPLETE]")

show_detailed_information()
# draw_graph(G)


def build_graph(merged_dict_path, counted_edges_path):
    # Initialize graph
    # Graph is weighted, directed
    G = igraph.Graph()

    # Add vertices
    merged_dict = common.read_two_columns_file_to_build_dictionary_type_specified(merged_dict_path, str, int)
    sorted_merged_dict = sorted(merged_dict.items(), key=operator.itemgetter(1), reverse=False)
    sorted_names = [item[0] for item in sorted_merged_dict]
    G.add_vertices(len(merged_dict))
    # Add name attribute to vertices
    G.vs["name"] = sorted_names
    # TODO delete
    print("Vertices added")

    # Add edges
    counted_edges = common.read_n_columns_file_to_build_list_of_lists_type_specified(counted_edges_path, [int, int, int])
    for source_target_weight in counted_edges:
        # We don't care self-loop edges
        if not source_target_weight[0] == source_target_weight[1]:
            G.add_edge(source_target_weight[0], source_target_weight[1])
            G[source_target_weight[0], source_target_weight[1], "weight"] = source_target_weight[2]
    # TODO delete
    print("Edges added")
    return G, merged_dict


def calculate_k_core_and_save_graph(graph, merged_dict):
    def get_k_core(graph):
        graph.vs["weight"] = graph.strength(graph.vs, weights=graph.es["weight"])
        # print("k-core unweighted ->", core_dec.core_dec(graph, weighted=False))
        return core_dec.core_dec(graph)

    def add_k_core_information_to_graph(graph, k_core_list, mergedDict):
        for name, core in k_core_list:
            graph.vs.select(mergedDict[name])["k_core"] = core

    def save_graph_svg(g):
        def gen_hex_colour_code(seed):
            import random
            random.seed(seed)
            return "#" + ''.join([random.choice('0123456789ABCDEF') for x in range(6)])

        visual_style = dict()
        visual_style["layout"] = g.layout("kk")
        visual_style["width"] = 2000
        visual_style["height"] = 2000
        visual_style["labels"] = "name"
        visual_style["colors"] = [gen_hex_colour_code(k_core) for k_core in g.vs["k_core"]]
        # visual_style["edge_color"] = ["#D6CFCE" * g.vcount()]
        visual_style["edge_colors"] = ["gray"] * g.ecount()
        g.write_svg("data/k_core.svg", **visual_style)

    sorted_cores_g = get_k_core(graph)
    # TODO delete
    print("k core done")
    # save_sorted_cores_dictionary(sorted_cores_g)
    add_k_core_information_to_graph(graph, sorted_cores_g, merged_dict)
    # save_graph_pickle(graph)
    save_graph_svg(graph)


# G, mergedDict = build_graph('data/dicts_and_encoded_texts/merged_dict.txt', 'data/counted_edges.txt')
# calculate_k_core_and_save_graph(G, mergedDict)
