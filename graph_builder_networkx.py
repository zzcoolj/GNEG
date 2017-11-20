import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import configparser
import sys
sys.path.insert(0, '../common/')
import common
import multi_processing

config = configparser.ConfigParser()
config.read('config.ini')


class NXGraph:
    def __init__(self, path, gpickle_name=None):
        if path.endswith('.gpickle'):
            self.graph = nx.read_gpickle(path)
        elif path.endswith('.txt'):
            self.graph = self.create_graph_with_weighted_edges(path, directed=True)
            nx.write_gpickle(self.graph, multi_processing.get_file_folder(path) + '/' + gpickle_name)

    @staticmethod
    def create_graph_with_weighted_edges(edges_file, directed):
        if directed:
            graph = nx.read_weighted_edgelist(edges_file, create_using=nx.DiGraph(), nodetype=int)
        else:
            graph = nx.read_weighted_edgelist(edges_file, create_using=nx.Graph(), nodetype=int)
        return graph

    @staticmethod
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

    def draw_graph(self):
        """
        Takes too much time with big data.
        """
        nx.draw(self.graph, with_labels=True)

        plt.show()

    def show_detailed_information(self):
        print("#nodes:", self.graph.number_of_nodes(), "#edges:", self.graph.number_of_edges())

    def get_shortest_path_lengths_between_all_nodes(self, output_folder=config['graph']['graph_folder']):
        """
        From test, these three algorithms below take more than 20 hours (processes have been killed after 20 hours) to
        calculate.
        'floyd_warshall_numpy' takes around 100 minutes to get the result.
        """
        # length1 = dict(nx.all_pairs_dijkstra_path_length(g))
        # length2 = dict(nx.all_pairs_bellman_ford_path_length(g))
        # length3 = nx.johnson(g, weight='weight')
        # for node in [0, 1, 2, 3, 4]:
        #     print('1 - {}: {}'.format(node, length2[1][node]))

        matrix = nx.floyd_warshall_numpy(self.graph)
        np.save(output_folder + 'matrix.npy', matrix, fix_imports=False)
        common.write_to_pickle(self.graph.nodes(), output_folder + 'nodes.pickle')
        return self.graph.nodes, matrix

    @staticmethod
    def get_selected_shortest_path_nodes(n, selected_mode, data_folder=config['graph']['graph_folder']):
        # TODO NOW Deal with inf
        n += 1  # add one more potential results, in case results have self loop node.
        matrix = np.load(data_folder + 'matrix.npy')  # matrix's values are indices of nodes list, not nodes indices
        nodes = common.read_pickle(data_folder + 'nodes.pickle')
        nodes_list = list(nodes)
        if selected_mode == 'min':
            selected_indices = np.argpartition(matrix, n)[:, :n]
        elif selected_mode == 'max':
            selected_indices = np.argpartition(matrix, -n)[:, -n:]
        cleaned_selected_indices = np.empty([selected_indices.shape[0], n - 1], dtype=int)
        # result_nodes = np.empty([selected_indices.shape[0], n - 1], dtype=int)
        shortest_path_nodes_dict = {}
        for i in range(matrix.shape[1]):  # shape[0] = shape[1]
            # e.g. for the first row (i=0), find the index in selected_indices where the value equals 0 (self loop)
            self_loop_index = np.where(selected_indices[i] == i)
            if self_loop_index[0].size == 0:  # no self loop
                shortest_path = matrix[i][selected_indices[i]]
                selected_index_shortest_path_length_dict = dict(zip(selected_indices[i], shortest_path))
                sorted_indices = sorted(selected_index_shortest_path_length_dict,
                                        key=selected_index_shortest_path_length_dict.get)
                if selected_mode == 'min':
                    cleaned_selected_indices[i] = sorted_indices[:n - 1]
                elif selected_mode == 'max':
                    cleaned_selected_indices[i] = sorted_indices[1:]
            else:
                cleaned_selected_indices[i] = np.delete(selected_indices[i], self_loop_index)
            # translate values to nodes indices, and the row's order follows the order of nodes
            shortest_path_nodes_dict[nodes_list[i]] = np.array(nodes)[cleaned_selected_indices[i]].tolist()
        common.write_to_pickle(shortest_path_nodes_dict, data_folder+'shortest_path_nodes_dict.pickle')
        return shortest_path_nodes_dict

    @staticmethod
    def translate_shortest_path_nodes_dict(shortest_path_nodes_dict, index2word_path, output_folder):
        index2word = read_two_columns_file_to_build_dictionary_type_specified(file=index2word_path)
        translated_shortest_path_nodes_dict = {}
        for key, value in shortest_path_nodes_dict.items():
            translated_shortest_path_nodes_dict[index2word[key]] = [index2word[node_id] for node_id in value]
        common.write_to_pickle(translated_shortest_path_nodes_dict,
                               output_folder + 'translated_shortest_path_nodes_dict.pickle')
        return translated_shortest_path_nodes_dict


def read_two_columns_file_to_build_dictionary_type_specified(file, key_type=int, value_type=str):
    """ATTENTION
    This function is different from what in graph_data_provider.
    Here, key is id and token is value, while in graph_data_provider, token is key and id is value.
    """
    d = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            (key, val) = line.rstrip('\n').split("\t")
            d[key_type(val)] = value_type(key)
        return d


if __name__ == '__main__':
    graph = NXGraph(config['graph']['graph_folder'] + 'encoded_edges_count_window_size_5_all_one.txt',
                    gpickle_name='graph.gpickle')
    graph.get_shortest_path_lengths_between_all_nodes(output_folder=config['graph']['graph_folder'])
    # translated_shortest_path_nodes_dict = NXGraph.translate_shortest_path_nodes_dict(
    #     NXGraph.get_selected_shortest_path_nodes(20, selected_mode='min', data_folder=config['graph']['graph_folder']),
    #     config['graph']['dicts_and_encoded_texts_folder']+'dict_merged.txt',
    #     output_folder=config['graph']['graph_folder'])
