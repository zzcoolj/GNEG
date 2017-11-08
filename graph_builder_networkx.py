import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Process
import time
import sys
sys.path.insert(0, '../common/')
import common


def create_graph_with_token_list(window):
    """ Usage
    For test, 8 times "is"
    create_graph_with_token_list(["This", "is", "is", "is", "is", "is", "is", "is", "is", "a"])
    """
    g = nx.DiGraph()
    g.add_nodes_from(set(window))
    window_size = len(window)
    for start_index in range(window_size):
        for end_index in range(start_index + 1, window_size):
            # !!! We don't care self-loop edges
            if not window[start_index] == window[end_index]:
                if g.has_edge(window[start_index], window[end_index]):
                    g[window[start_index]][window[end_index]]['weight'] += 1
                else:
                    g.add_edge(window[start_index], window[end_index], weight=1)
    return g


def draw_graph(g):
    """
    Takes too much time with big data.
    """
    nx.draw(g, with_labels=True)

    plt.show()


def show_detailed_information(g):
    print("#nodes:", g.number_of_nodes(), "#edges:", g.number_of_edges())


def create_graph_with_weighted_edges(edges_file, directed):
    if directed:
        graph = nx.read_weighted_edgelist(edges_file, create_using=nx.DiGraph(), nodetype=int)
    else:
        graph = nx.read_weighted_edgelist(edges_file, create_using=nx.Graph(), nodetype=int)
    return graph


def get_longest_shortest_path_nodes(g, source, n):
    shortest_path_dict = nx.shortest_path_length(g, source, weight='weight')
    # print(shortest_path_dict)
    sorted_nodes = list(sorted(shortest_path_dict, key=shortest_path_dict.get, reverse=True))
    return sorted_nodes[:n]


def get_shortest_path_lengths_between_all_nodes(g):

    length1 = dict(nx.all_pairs_dijkstra_path_length(g))
    for node in [0, 1, 2, 3, 4]:
        print('1 - {}: {}'.format(node, length1[1][node]))

    length2 = dict(nx.all_pairs_bellman_ford_path_length(g))
    for node in [0, 1, 2, 3, 4]:
        print('1 - {}: {}'.format(node, length2[1][node]))

    ''' From doc
    For dense graphs, this may be faster than the Floyd–Warshall algorithm.
    '''
    length3 = nx.johnson(g, weight='weight')
    for node in [0, 1, 2, 3, 4]:
        print('1 - {}: {}'.format(node, length2[1][node]))

    length4 = nx.floyd_warshall_numpy(g)


def f1():
    print('f1 start')
    start_time = time.time()
    nx.read_gpickle("output/intermediate data/graph/graph.gpickle")
    print('pickle:' + str(common.count_time(start_time)))


def f2():
    print('f2 start')
    start_time = time.time()
    nx.read_weighted_edgelist('output/intermediate data/edges/encoded_edges_count_filtered.txt',
                              create_using=nx.DiGraph(), nodetype=int)
    print('txt:' + str(common.count_time(start_time)))


if __name__ == '__main__':
    # g = create_graph_with_weighted_edges(
    #     edges_file='output/intermediate data for unittest/edges/encoded_edges_count_window_size_6.txt',
    #     directed=True)
    # show_detailed_information(g)
    # print(get_longest_shortest_path_nodes(g, 14, 4))

    p = Process(target=f1())
    p.start()
    q = Process(target=f2())
    q.start()

