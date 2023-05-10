import json
from tqdm import tqdm
import copy
from argparse import ArgumentParser

def convert_dep_const_graph(file_path_in, file_path_out):
    # read json file path
    data_json = read_json_file(file_path_in)
    data_json_new = []
    for index, sent_dict in enumerate(tqdm(data_json)):
        tokens = sent_dict['tokens']
        const_tree = sent_dict['stanford_constituency_tree']
        # list_token is in (token, pos, token_index, direct const node index) 4-tuple form
        # list_const_node is in (const_node, const_index) 2-tuple form
        # list_const_edge is in (const_node_start, const_node_end) 2-tuple form
        list_token, list_const_node, list_const_edge = const_index_rer(const_tree, list(), list(), list(), 0)
        list_token_const_path = const_path_index_rer(const_tree, list(), list())

        if len(list_token) == len(tokens):
            if not check_parser_consistency(tokens, list_token):
                print('wrong', index)
                continue

            adj_dep_matrix = create_adj_matrix(len(tokens))
            adj_dep_edges = []
            dep_head_index = sent_dict['dep_head_index']
            dep_labels = sent_dict['spacy_dep']
            for child_index, (head_index, dep_tag) in enumerate(zip(dep_head_index, dep_labels)):
                adj_dep_matrix[child_index][head_index] = 1
                adj_dep_matrix[head_index][child_index] = 1
                adj_dep_edges.append(((head_index, child_index), dep_tag))

            const_to_token, adj_const_edges = index_const_to_token_span(list_token, list_const_node, list_const_edge)
            token_to_const = index_token_to_const(list_token, const_to_token)
            adj_const_matrix = create_adj_matrix(len(const_to_token))

            for node_start, node_end in list_const_edge:
                adj_const_matrix[node_start][node_end] = 1
                adj_const_matrix[node_end][node_start] = 1

            key_del_list = ['stanford_dep_basic', 'stanford_dep_basic_head', 'stanford_dep_basic_head_list',
                            'stanford_dep_enhance', 'stanford_dep_enhance_head', 'stanford_dep_enhance_head_list']
            for key_del in key_del_list:
                sent_dict.pop(key_del)

            sent_dict['dep_graph_nodes'] = dep_labels
            sent_dict['dep_graph_edges'] = adj_dep_edges
            sent_dict['const_graph_nodes'] = list_token_const_path
            sent_dict['const_graph_edges'] = adj_const_edges
            data_json_new.append(sent_dict)

    print(len(data_json), len(data_json_new))
    # generate json file with const graph
    file_out = open(file_path_out, 'w', encoding='utf-8')
    json.dump(data_json_new, file_out)
    return


def check_parser_consistency(spacy_tokens, stanford_tokens):
    stanford_tokens = [token for token, _, _, _ in stanford_tokens]
    if len(spacy_tokens) != len(stanford_tokens):
        return False
    for x, y in zip(spacy_tokens, stanford_tokens):
        if x != y and x != '(' and x != ')':
            return False
    return True

def create_adj_matrix(n):
    adj_matrix = []
    for i in range(n):
        temp = [0 for j in range(n)]
        adj_matrix.append(temp)
    return adj_matrix

def index_token_to_const(list_token, const_to_token):
    temp_token_node = [[elem] for elem, _, _, _ in list_token]
    for const_index, const_node in enumerate(const_to_token):
        _, token_start, token_end = const_node
        temp_token_node[token_start].append(const_index)
        temp_token_node[token_end-1].append(const_index)

    return temp_token_node

def index_const_to_token_span(list_token, list_const_node, list_const_edge):
    temp_const_node = [(elem) for elem, _ in list_const_node]

    # link token index to const index
    for _, _, token_num, const_num in list_token:
        if isinstance(temp_const_node[const_num], str):
            const_node = temp_const_node[const_num]
            temp_const_node[const_num] = (const_node, token_num, token_num)
        else:
            const_node, index_start, index_end = temp_const_node[const_num]
            if token_num < index_start:
                temp_const_node[const_num] = (const_node, token_num, index_end)
            elif token_num > index_end:
                temp_const_node[const_num] = (const_node, index_start, token_num)
            else:
                print('add duplicated token to const index.. please check token index')
                raise Exception

    const_short_node = copy.copy(temp_const_node)
    for node_start, node_end in list_const_edge:
        if isinstance(const_short_node[node_start], str):
            x0 = const_short_node[node_start]
            x1, x2 = 99999, -99999
        else:
            x0, x1, x2 = const_short_node[node_start]

        if isinstance(const_short_node[node_end], str):
            if const_short_node[node_end] == 'S':
                y1 = y2 = x2
            else:
                y1 = y2 = x2 + 1
        else:
            _, y1, y2 = const_short_node[node_end]

        if min(x1, y1) == 99999 or max(x2, y2) == -99999:
            # this happens when SBAR and S are connected.
            # print('both nodes are empty')
            const_short_node[node_start] = (x0, -1, -1)
        else:
            const_short_node[node_start] = (x0, min(x1, y1), max(x2, y2))


    const_long_node = copy.copy(temp_const_node)
    for node_start, node_end in reversed(list_const_edge):
        if isinstance(const_long_node[node_start], str):
            x0 = const_long_node[node_start]
            x1, x2 = 99999, -99999
        else:
            x0, x1, x2 = const_long_node[node_start]

        if isinstance(const_long_node[node_end], str):
            y1, y2 = 99999, -99999
        else:
            _, y1, y2 = const_long_node[node_end]

        if min(x1, y1) == 99999 or max(x2, y2) == -99999:
            print('both nodes are empty')
            const_long_node[node_start] = (x0, -1, -1)
        else:
            const_long_node[node_start] = (x0, min(x1, y1), max(x2, y2))

    const_final_node = []
    for (x0, x1, x2), (y0, y1, y2) in zip(const_short_node, const_long_node):
        if x0 == y0:
            if x0 in ['NP', 'S', 'SBAR', 'UCP', 'FRAG', 'SBARQ', 'SINV']:
                const_final_node.append((x0, y1, y2))
            else:
                const_final_node.append((x0, x1, x2))
        else:
            print('error format, please check')
            raise AssertionError

    edge_list_new = []
    for x0, x1, x2 in const_final_node:
        if x1 != -1 and x2 != -1:
            if x2 >= len(list_token):
                print('error')
            else:
                edge_list_new.append(((x1, x2), x0))

    for node_start, node_end in list_const_edge:
        x0, x1, x2 = const_final_node[node_start]
        _, y1, y2 = const_final_node[node_end]
        if y1 > x1:
            if y1 != -1 and x1 != -1:
                if y1 >= len(list_token):
                    print('error')
                else:
                    edge_list_new.append(((x1, y1), x0))

    edge_list_new = list(dict.fromkeys(edge_list_new))

    return const_long_node, edge_list_new


def checktype(lst):
    if lst and isinstance(lst, list):
        return all(isinstance(elem, str) for elem in lst)
    else:
        return False


def const_path_index_rer(const_tree, token_list, const_list):
    # check if elem contains only POS and Token information
    # if there is no const type, proceed to add tokens
    if checktype(const_tree):
        # token with index is added...
        token_list.append((const_tree[1], len(token_list), const_list))
        return token_list
    else:
        # get constituency tree types
        const_type = const_tree[0]
        if const_type != 'ROOT' and isinstance(const_type, str):
            const_list.append(const_type)

        # get siblings from constituency tree
        for elem in const_tree:
            if not isinstance(elem, str):
                # recursively adding information from top to down, from left to right
                token_list = const_path_index_rer(elem, token_list, copy.copy(const_list))

        return token_list


def const_index_rer(const_tree, token_list, const_list, const_edge, const_num):
    # check if elem contains only POS and Token information
    # if there is no const type, proceed to add tokens
    if checktype(const_tree):
        # token with index is added...
        token_list.append((const_tree[1], const_tree[0], len(token_list), const_num))
        return token_list, const_list, const_edge
    else:
        # get constituency tree types
        const_type = const_tree[0]
        if const_type != 'ROOT' and isinstance(const_type, str):
            patent_num = len(const_list)
            const_num = patent_num
            # const node with index is added...
            const_list.append((const_type, len(const_list)))
        else:
            patent_num = 0

        # get siblings from constituency tree
        for elem in const_tree:
            if not isinstance(elem, str):
                # check if elem contains only POS and Token information
                # if there is no const type, just skip adding edge
                if not checktype(elem):
                    # check child nodes constituency type
                    const_child_type = elem[0]
                    if const_child_type != 'ROOT' and isinstance(const_child_type, str):
                        child_num = len(const_list)
                        # add edges for constituency nodes
                        if patent_num < child_num:
                            # an edge between const node is added...
                            const_edge.append((patent_num, child_num))

                # recursively adding information from top to down, from left to right
                token_list, const_list, const_edge = \
                    const_index_rer(elem, token_list, const_list, const_edge, const_num)

        return token_list, const_list, const_edge

def read_json_file(file_in):
    # read json file path
    file_json = open(file_in, 'r', encoding='utf-8')
    data_json = json.load(file_json)
    file_json.close()
    return data_json

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--inp", dest="file_in", default='test', help="input file")
    parser.add_argument("--out", dest="file_out", default='test', help="output file")
    args = parser.parse_args()
    file_in, file_out = args.file_in, args.file_out

    # read json file as input
    convert_dep_const_graph(file_in, file_out)