import networkx as nx
import matplotlib.pyplot as plt


def create_graph_with_token_list(window):
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
    print(shortest_path_dict)
    sorted_nodes = list(sorted(shortest_path_dict, key=shortest_path_dict.get, reverse=True))
    return sorted_nodes[:n]


if __name__ == '__main__':
    # # For test, 8 times "is"
    # create_graph_with_token_list(["This", "is", "is", "is", "is", "is", "is", "is", "is", "a"])

    g = create_graph_with_weighted_edges(
        edges_file='output/intermediate data for unittest/edges/encoded_edges_count_window_size_6.txt',
        directed=True)
    show_detailed_information(g)
    print(get_longest_shortest_path_nodes(g, 14, 4))
    # multimillionair 03
