from typing import Dict, List, Iterable, Tuple, Any
import json


class DependencyIndexter:
    def __init__(self, dependency_dict_file):
        file_in = open(dependency_dict_file, 'r', encoding='utf-8')
        json_data = json.load(file_in)
        self.mapping = json_data

    def get_dep_index(self, dep_tag_list):
        index_list = [self.mapping[x] for x in dep_tag_list]
        return index_list


def convert_tags_to_wordpiece_tags(tags: List[str], offsets: List[int]) -> List[str]:
    """
    Converts a series of BIO tags to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.

    This is only used if you pass a `bert_model_name` to the dataset reader below.

    Parameters
    ----------
    tags : `List[str]`
        The BIO formatted tags to convert to BIO tags for wordpieces
    offsets : `List[int]`
        The wordpiece offsets.

    Returns
    -------
    The new BIO tags.
    """
    new_tags = []
    j = 0
    for i, offset in enumerate(offsets):
        tag = tags[i]
        is_o = tag == "O"
        is_start = True
        while j < offset:
            if is_o:
                new_tags.append("O")

            elif tag.startswith("I"):
                new_tags.append(tag)

            elif is_start and tag.startswith("B"):
                new_tags.append(tag)
                is_start = False

            elif tag.startswith("B"):
                _, label = tag.split("-", 1)
                new_tags.append("I-" + label)
            j += 1

    # Add O tags for cls and sep tokens.
    return ["O"] + new_tags + ["O"]


def convert_const_tags_to_wordpiece_const_tags(tags: List[List[str]], offsets: List[int]) -> List[List[str]]:
    new_tags = []
    j = 0
    for i, offset in enumerate(offsets):
        tag = tags[i]
        while j < offset:
            new_tags.append(tag)
            j += 1
    new_tags.insert(0, ["CLS"])
    new_tags.append(["SEP"])
    return new_tags


def convert_dep_tags_to_wordpiece_dep_tags(tags: List[str], offsets: List[int]) -> List[str]:
    new_tags = []
    j = 0
    for i, offset in enumerate(offsets):
        tag = tags[i]
        while j < offset:
            new_tags.append(tag)
            j += 1
    return ["CLS"] + new_tags + ["SEP"]


def convert_dep_adj_to_wordpiece_dep_adj(edges: List[Tuple[int, int]], start_offsets: List[int], offsets: List[int]) -> \
List[str]:
    new_dep_adj = []
    new_mapping = {}
    for i, (start_offset, offset) in enumerate(zip(start_offsets, offsets)):
        new_mapping[i] = [x for x in range(start_offset, offset + 1)]

    for node_i, node_j in edges:
        nodes_i = new_mapping[node_i]
        nodes_j = new_mapping[node_j]
        edges_new = [(nodes_i_new, nodes_j_new) for nodes_i_new in nodes_i for nodes_j_new in nodes_j]
        new_dep_adj.extend(edges_new)
        if len(nodes_i) > 1:
            new_dep_adj.extend(get_edges_among_subwords(nodes_i))
        if len(nodes_j) > 1:
            new_dep_adj.extend(get_edges_among_subwords(nodes_j))

    new_dep_adj = list(set(new_dep_adj))
    return new_dep_adj


def get_edges_among_subwords(subword_list):
    return [(subword_list[i], subword_list[i + 1]) for i in range(len(subword_list) - 1)]


def convert_verb_indices_to_wordpiece_indices(verb_indices: List[int],
                                              offsets: List[int]):  # pylint: disable=invalid-name
    """
    Converts binary verb indicators to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.

    This is only used if you pass a `bert_model_name` to the dataset reader below.

    Parameters
    ----------
    verb_indices : `List[int]`
        The binary verb indicators, 0 for not a verb, 1 for verb.
    offsets : `List[int]`
        The wordpiece offsets.

    Returns
    -------
    The new verb indices.
    """
    j = 0
    new_verb_indices = []
    for i, offset in enumerate(offsets):
        indicator = verb_indices[i]
        while j < offset:
            new_verb_indices.append(indicator)
            j += 1
    # Add 0 indicators for cls and sep tokens.
    return [0] + new_verb_indices + [0]


def get_2nd_order_pairs(edge_list1: List[Tuple[int, int]], edge_list2: List[Tuple[int, int]]):
    list1, list2 = [], []
    for x0, y0 in edge_list1:
        edge_exist = False
        for x1, y1 in edge_list2:
            if x1 == x0 and y1 == y0:
                edge_exist = True
                break
        if not edge_exist:
            list1.append(x0)
            list2.append(y0)
    return list1, list2


def convert_tags_into_copynet_str(token_list: List[str], tag_list: List[str]) -> str:
    extraction_str = ''
    arg1, rel, arg2, arg3, arg4 = '', '', '', '', ''
    for token, tag in zip(token_list, tag_list):
        if tag == 'B-ARG0':
            arg1 = token
        elif tag == 'I-ARG0':
            arg1 += ' ' + token
        elif tag == 'B-ARG1':
            arg2 = token
        elif tag == 'I-ARG1':
            arg2 += ' ' + token
        elif tag == 'B-ARG2':
            arg3 = token
        elif tag == 'I-ARG2':
            arg3 += ' ' + token
        elif tag == 'B-ARG3':
            arg4 = token
        elif tag == 'I-ARG3':
            arg4 += ' ' + token
        elif tag == 'B-V':
            rel = token
        elif tag == 'I-V':
            rel += ' ' + token

    if arg1 != '':
        extraction_str = "<arg1> {} </arg1>".format(arg1)
    if rel != '':
        extraction_str += " <rel> {} </rel>".format(rel)
    if arg2 != '':
        extraction_str += " <arg2> {} </arg2>".format(arg2)
    if arg3 != '':
        extraction_str += " <arg3> {} </arg3>".format(arg3)
    if arg4 != '':
        extraction_str += " <arg4> {} </arg4>".format(arg4)
    return extraction_str
