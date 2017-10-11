import networkx as nx
import matplotlib.pyplot as plt
from core_dec import core_dec

# Graph is weighted, directed
G = nx.DiGraph()


def extract_graph_information(window):
    G.add_nodes_from(set(window))
    window_size = len(window)
    for start_index in range(window_size):
        for end_index in range(start_index+1, window_size):
            # !!! We don't care self-loop edges
            if not window[start_index] == window[end_index]:
                if G.has_edge(window[start_index], window[end_index]):
                    G[window[start_index]][window[end_index]]['weight'] += 1
                else:
                    G.add_edge(window[start_index], window[end_index], weight=1)


def draw_graph():
    nx.draw(G, with_labels=True)
    plt.show()


def show_detailed_information():
    print("#nodes:", G.number_of_nodes(), "#edges:", G.number_of_edges())

    for n, nbrs in G.adjacency_iter():
        print(n, "-> degree: ", G.degree(n))
        for nbr, eattr in nbrs.items():
            weight = eattr['weight']
            print('(%s, %s, %d)' % (n, nbr, weight))


def k_core_test():
    print("degree ->", G.degree())
    print("weighted degree ->", G.degree(weight='weight'))
    print("k-core ->", nx.core_number(G))
    # print("k-core weighted ->", core_number_weighted())
    print("k-core weighted ->", core_dec(G, False))


# TODO It's not just change degrees=G.degree() to degrees=G.degree(weight='weight').
# TODO Algorithm may only work with unweighted
def core_number_weighted():
    """Return the core number for each vertex.

    A k-core is a maximal subgraph that contains nodes of degree k or more.

    The core number of a node is the largest value k of a k-core containing
    that node.

    Parameters
    ----------
    G : NetworkX graph
       A graph or directed graph

    Returns
    -------
    core_number : dictionary
       A dictionary keyed by node to the core number.

    Raises
    ------
    NetworkXError
        The k-core is not defined for graphs with self loops or parallel edges.

    Notes
    -----
    Not implemented for graphs with parallel edges or self loops.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    References
    ----------
    .. [1] An O(m) Algorithm for Cores Decomposition of Networks
       Vladimir Batagelj and Matjaz Zaversnik, 2003.
       http://arxiv.org/abs/cs.DS/0310049
    """
    if G.is_multigraph():
        raise nx.NetworkXError(
                'MultiGraph and MultiDiGraph types not supported.')

    if G.number_of_selfloops()>0:
        raise nx.NetworkXError(
                'Input graph has self loops; the core number is not defined.',
                'Consider using G.remove_edges_from(G.selfloop_edges()).')

    if G.is_directed():
        import itertools
        def neighbors(v):
            return itertools.chain.from_iterable([G.predecessors_iter(v),
                                                  G.successors_iter(v)])
    else:
        neighbors=G.neighbors_iter
    degrees=G.degree(weight='weight')
    # sort nodes by degree
    nodes=sorted(degrees,key=degrees.get)
    print("degrees:", degrees)
    print("nodes:", nodes)
    # TODO what's bin_boundaries for?
    bin_boundaries=[0]
    curr_degree=0
    for i,v in enumerate(nodes):
        print("i:", i, "v:", v)
        print("\tdegrees[v]:", degrees[v], "\tcurr_degree:", curr_degree)
        if degrees[v]>curr_degree:
            print("\t>")
            bin_boundaries.extend([i]*(degrees[v]-curr_degree))
            print("\tbin_boundaries:", bin_boundaries)
            curr_degree=degrees[v]
            print("\tcurr_degree:", curr_degree)
    node_pos = dict((v,pos) for pos,v in enumerate(nodes))
    # initial guesses for core is degree
    core=degrees
    print("core initial:", core)
    print("bin_boundaries:", bin_boundaries)
    print("node_pos:", node_pos)
    nbrs=dict((v,set(neighbors(v))) for v in G)
    print("nbrs:", nbrs)
    for v in nodes:
        for u in nbrs[v]:
            print("v:", v, "\tu:", u)
            print("core[u]:", core[u], "\tcore[v]", core[v])
            if core[u] > core[v]:
                print("\t >")
                nbrs[u].remove(v)
                print("\tnbrs:", nbrs)
                pos=node_pos[u]
                print("\tpos:", pos)
                bin_start=bin_boundaries[core[u]]
                print("\tcore[u]:", core[u])
                print("\tbin_start:", bin_start)
                node_pos[u]=bin_start
                node_pos[nodes[bin_start]]=pos
                nodes[bin_start],nodes[pos]=nodes[pos],nodes[bin_start]
                bin_boundaries[core[u]]+=1
                core[u]-=1
    return core


def create_graph(file_path, directed):
    if directed:
        graph = nx.read_weighted_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int)
    else:
        graph = nx.read_weighted_edgelist(file_path, create_using=nx.Graph(), nodetype=int)
    return graph


# For test, 8 times "is"
extract_graph_information(["This", "is", "is", "is", "is", "is", "is", "is", "is", "a"])
k_core_test()

# show_detailed_information()
# draw_graph()