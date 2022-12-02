import logging
from typing import Dict, List, Iterable, Tuple, Any

import numpy as np
from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, AdjacencyField, ListField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from SMiLe_OIE.dataset_helper import *
import json

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("bert_multi_view_oie")
class SMiLe_OIE_Reader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 domain_identifier: str = None,
                 lazy: bool = False,
                 validation: bool = False,
                 data_type: str = 'oo',
                 verbal_indicator: bool = True,
                 bert_model_name: str = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._dep_tag_indexers = {"dep_tags": SingleIdTokenIndexer(namespace="dependency_labels")}
        self._const_tag_indexers = {"const_tags": SingleIdTokenIndexer(namespace="constituency_labels")}
        self._domain_identifier = domain_identifier
        self._validation = validation
        self._data_type = data_type
        self._verbal_indicator = verbal_indicator

        if bert_model_name is not None:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.lowercase_input = "uncased" in bert_model_name
        else:
            self.bert_tokenizer = None
            self.lowercase_input = False

    def _wordpiece_tokenize_input(self, tokens: List[str]) -> Tuple[List[str], List[int], List[int]]:
        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            if self.lowercase_input:
                token = token.lower()
            word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)
            start_offsets.append(cumulative + 1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)
        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]
        return wordpieces, end_offsets, start_offsets

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info("Filtering to only include file paths containing the %s domain", self._domain_identifier)
        file_in = open(file_path, 'r', encoding='utf-8')
        json_sent = json.load(file_in)
        file_in.close()
        for sentence in json_sent:
            tokens = [Token(x) for x in sentence['tokens']]
            dep_graph_nodes = sentence['dep_graph_nodes']
            dep_graph_edges = sentence['dep_graph_edges']
            const_graph_nodes = sentence['const_graph_nodes']
            const_graph_edges = sentence['const_graph_edges']

            if self._validation:
                if self._verbal_indicator:
                    pos_list = sentence['spacy_pos']
                    for index, pos in enumerate(pos_list):
                        if pos == "VERB":
                            verb_indicator = [0] * len(tokens)
                            verb_indicator[index] = 1
                            yield self.text_to_instance(tokens, verb_indicator, dep_graph_nodes, dep_graph_edges, const_graph_nodes, const_graph_edges, tags=None)
                else:
                    verb_indicator = sentence['verb_label']
                    yield self.text_to_instance(tokens, verb_indicator, dep_graph_nodes, dep_graph_edges, const_graph_nodes, const_graph_edges, tags=None)
            else:
                if len(tokens) > 120:
                    continue
                if self._data_type == 'oo':
                    tags_oo = sentence['tags_oo']
                    tags_oo_v = sentence['tags_oo_v']
                    for tags, verb_indicator in zip(tags_oo, tags_oo_v):
                        yield self.text_to_instance(tokens, verb_indicator, dep_graph_nodes, dep_graph_edges, const_graph_nodes, const_graph_edges, tags=tags)
                elif self._data_type == 'aug':
                    tags_aug = sentence['tags_aug']
                    tags_aug_v = sentence['tags_aug_v']
                    for tags, verb_indicator in zip(tags_aug, tags_aug_v):
                        yield self.text_to_instance(tokens, verb_indicator, dep_graph_nodes, dep_graph_edges, const_graph_nodes, const_graph_edges, tags=tags)

    def text_to_instance(self, tokens: List[Token], verb_label: List[int],
                         dep_graph_nodes: List = None, dep_graph_edges: List = None,
                         const_graph_nodes: List = None, const_graph_edges: List = None,
                         tags: List[str] = None) -> Instance:
        dep_edges_tuple = [(item[0][0], item[0][1]) for item in dep_graph_edges]
        metadata_dict: Dict[str, Any] = {}
        metadata_dict["dep_nodes"] = dep_graph_nodes
        metadata_dict["dep_edges"] = dep_edges_tuple

        const_graph_nodes = [item[2] for item in const_graph_nodes]
        const_edges_tuple = [(item[0][0], item[0][1]) for item in const_graph_edges
                             if len(const_graph_nodes) > item[0][1] >= item[0][0] >= 0
                             and (item[0][1]-item[0][0]) < 10]
        metadata_dict["const_nodes"] = const_graph_nodes
        metadata_dict["const_edges"] = const_edges_tuple

        if self.bert_tokenizer is not None:
            wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input([t.text for t in tokens])
            new_verbs = convert_verb_indices_to_wordpiece_indices(verb_label, offsets)
            metadata_dict["offsets"] = start_offsets
            text_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces], token_indexers=self._token_indexers)
            verb_indicator = SequenceLabelField(new_verbs, text_field)
            # convert word label dependency labels to wordpiece labels
            dep_graph_nodes = convert_dep_tags_to_wordpiece_dep_tags(dep_graph_nodes, offsets)
            dep_edges_tuple = convert_dep_adj_to_wordpiece_dep_adj(dep_edges_tuple, start_offsets, offsets)
            const_graph_nodes = convert_const_tags_to_wordpiece_const_tags(const_graph_nodes, offsets)
            const_edges_tuple = convert_dep_adj_to_wordpiece_dep_adj(const_edges_tuple, start_offsets, offsets)

        else:
            text_field = TextField(tokens, token_indexers=self._token_indexers)
            verb_indicator = SequenceLabelField(verb_label, text_field)

        dep_field = TextField([Token(t) for t in dep_graph_nodes], token_indexers=self._dep_tag_indexers)
        dep_adj_field = AdjacencyField(dep_edges_tuple, dep_field, padding_value=0)

        const_list = []
        for nodes_list in const_graph_nodes:
            const_field = TextField([Token(t) for t in nodes_list], token_indexers=self._const_tag_indexers)
            const_list.append(const_field)
        const_list_field = ListField(const_list)
        const_adj_field = AdjacencyField(const_edges_tuple, const_list_field, padding_value=0)

        # this is intra-view diversity
        dep_1_start = ArrayField(np.array([x for x, _ in dep_edges_tuple]))
        dep_1_end = ArrayField(np.array([x for _, x in dep_edges_tuple]))
        const_1_start = ArrayField(np.array([x for x, _ in const_edges_tuple]))
        const_1_end = ArrayField(np.array([x for _, x in const_edges_tuple]))

        # we don't need to make index for inter-view intra-node 1st order collaboration
        # this is inter-view inter-node 2nd order collaboration
        dep_3_start, dep_3_const_end = get_2nd_order_pairs(const_edges_tuple, dep_edges_tuple)
        const_3_start, const_3_dep_end = get_2nd_order_pairs(dep_edges_tuple, const_edges_tuple)
        dep_3_start = ArrayField(np.array(dep_3_start))
        dep_3_const_end = ArrayField(np.array(dep_3_const_end))
        const_3_start = ArrayField(np.array(const_3_start))
        const_3_dep_end = ArrayField(np.array(const_3_dep_end))

        fields = {'tokens': text_field, 'verb_indicator': verb_indicator,
                  'dep_nodes': dep_field, 'dep_edges': dep_adj_field,
                  'const_nodes': const_list_field, 'const_edges': const_adj_field,
                  'dep_1_start': dep_1_start, 'dep_1_end': dep_1_end,
                  'const_1_start': const_1_start, 'const_1_end': const_1_end,
                  'dep_3_start': dep_3_start, 'dep_3_const_end': dep_3_const_end,
                  'const_3_start': const_3_start, 'const_3_dep_end': const_3_dep_end}


        if all([x == 0 for x in verb_label]):
            verb = None
            verb_index = None
        else:
            verb_index = verb_label.index(1)
            verb = tokens[verb_index].text

        metadata_dict["words"] = [x.text for x in tokens]
        metadata_dict["verb"] = verb
        metadata_dict["verb_index"] = verb_index
        metadata_dict["validation"] = self._validation

        if tags:
            if self.bert_tokenizer is not None:
                new_tags = convert_tags_to_wordpiece_tags(tags, offsets)
                fields['tags'] = SequenceLabelField(new_tags, text_field)
            else:
                fields['tags'] = SequenceLabelField(tags, text_field)
            metadata_dict["gold_tags"] = tags

        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)