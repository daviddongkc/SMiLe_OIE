from typing import Iterable
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence
from tqdm import tqdm
import json
from stanfordcorenlp import StanfordCoreNLP
from pyparsing import OneOrMore, nestedExpr
import ast
from argparse import ArgumentParser
import spacy
from spacy.tokens import Doc
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def read_ontonotes_file(file_path):
    ontonotes_reader = Ontonotes()
    print('reading ' + file_path)

    def ontonotes_subset(ontonotes_reader: Ontonotes, file_path: str, ) -> Iterable[OntonotesSentence]:
        yield from ontonotes_reader.sentence_iterator(file_path)

    sentences = ontonotes_subset(ontonotes_reader, file_path)
    sentence_list = []
    for sentence in tqdm(sentences):
        tokens = [t for t in sentence.words]
        if sentence.srl_frames:
            for (_, tags) in sentence.srl_frames:
                verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                sentence_list.append({'tokens': tokens, 'verb_label': verb_indicator, 'tags': tags,
                                      'pred_str': sentence.pred})
    return sentence_list


def get_dep_tree(data_json):
    nlp = spacy.load('en_core_web_trf', disable=['ner', 'textcat'])
    for i, sentence in enumerate(tqdm(data_json)):
        token_list = sentence['tokens']

        spacy_doc = spacy.tokens.Doc(nlp.vocab, words=token_list)
        for _, proc in nlp.pipeline:
            spacy_doc = proc(spacy_doc)

        token_string_list, index_list, pos_list, tag_list, dep_list, head_list, head_index_list = [], [], [], [], [], [], []
        for token in spacy_doc:
            token_string, token_index, token_pos, token_tag, token_dep = token.orth_, token.i, token.pos_, token.tag_, token.dep_
            token_string_list.append(token_string)
            index_list.append(token_index)
            pos_list.append(token_pos)
            tag_list.append(token_tag)
            dep_list.append(token_dep)
            token_head = token.head
            token_head_string, token_head_index = token_head.orth_, token_head.i
            head_list.append(token_head_string)
            head_index_list.append(token_head_index)

        if len(token_list) != len(token_string_list):
            print('error: number', i, '   ', len(token_list), 'after', len(token_string_list))
        else:
            sentence['spacy_pos'] = pos_list
            sentence['spacy_tag'] = tag_list
            sentence['spacy_dep'] = dep_list
            sentence['dep_head'] = head_list
            sentence['dep_head_index'] = head_index_list

    return data_json

def extract_constuency(data_json):
    server = StanfordCoreNLP('http://localhost', port=9000)
    pros = {'annotators': 'parse', 'pinelineLanguage': 'en', 'tokenize.whitespace': 'true'}

    for i, sentence in enumerate(tqdm(data_json)):
        tokens = sentence['tokens']
        text = ' '.join(tokens)
        try:
            sentence_json = json.loads(server.annotate(text, properties=pros))
        except:
            print('sentence {} is not working for Stanford NLP'.format(i))
            continue

        parse_tree_str = sentence_json['sentences'][0]['parse']
        parse_tree = OneOrMore(nestedExpr()).parseString(parse_tree_str)
        parse_tree_list_string = str(parse_tree)
        parse_tree_list = ast.literal_eval(parse_tree_list_string)
        dep_basic_list = sentence_json['sentences'][0]['basicDependencies']
        dep_enhance_list = sentence_json['sentences'][0]['enhancedPlusPlusDependencies']
        dep_basic_list = sorted(dep_basic_list, key=lambda d: d['dependent'])
        dep_enhance_list = sorted(dep_enhance_list, key=lambda d: d['dependent'])

        dep_label_list, dep_head_list, dep_head_index_list = [], [], []
        for dep_basic in dep_basic_list:
            dep_label = dep_basic['dep']
            dep_head = dep_basic['governorGloss']
            dep_head_index = dep_basic['governor'] - 1
            if dep_head_index < 0:
                dep_head_index = len(dep_label_list)
            dep_label_list.append(dep_label)
            dep_head_list.append(dep_head)
            dep_head_index_list.append(dep_head_index)

        dep_enh_label_list, dep_enh_head_list, dep_enh_head_index_list = [], [], []
        for dep_enhance in dep_enhance_list:
            dep_label = dep_enhance['dep']
            dep_head = dep_enhance['governorGloss']
            dep_head_index = dep_enhance['governor'] - 1
            current_index = dep_enhance['dependent'] - 1
            if dep_head_index < 0:
                dep_head_index = current_index
            if current_index == len(dep_enh_label_list):
                dep_enh_label_list.append(dep_label)
                dep_enh_head_list.append(dep_head)
                dep_enh_head_index_list.append(dep_head_index)
            else:
                temp1 = dep_enh_label_list.pop()
                temp2 = dep_enh_head_list.pop()
                temp3 = dep_enh_head_index_list.pop()
                if isinstance(temp1, str):
                    dep_enh_label_list.append([temp1, dep_label])
                    dep_enh_head_list.append([temp2, dep_head])
                    dep_enh_head_index_list.append([temp3, dep_head_index])
                else:
                    temp1.append(dep_label)
                    temp2.append(dep_head)
                    temp3.append(dep_head_index)
                    dep_enh_label_list.append(temp1)
                    dep_enh_head_list.append(temp2)
                    dep_enh_head_index_list.append(temp3)

        if not len(tokens) == len(dep_label_list) == len(dep_enh_label_list):
            if len(dep_label_list) == 0:
                continue
            if len(dep_label_list) < len(tokens):
                temp_label = sentence['spacy_dep'][len(dep_label_list):]
                temp_head = sentence['dep_head'][len(dep_label_list):]
                temp_index = sentence['dep_head_index'][len(dep_label_list):]
                dep_label_list.extend(temp_label)
                dep_head_list.extend(temp_head)
                dep_head_index_list.extend(temp_index)

            if len(dep_enh_label_list) < len(tokens):
                temp_label = sentence['spacy_dep'][len(dep_enh_label_list):]
                temp_head = sentence['dep_head'][len(dep_enh_label_list):]
                temp_index = sentence['dep_head_index'][len(dep_enh_label_list):]
                dep_enh_label_list.extend(temp_label)
                dep_enh_head_list.extend(temp_head)
                dep_enh_head_index_list.extend(temp_index)

        sentence['stanford_constituency_tree'] = parse_tree_list
        sentence['stanford_dep_basic'] = dep_label_list
        sentence['stanford_dep_basic_head'] = dep_head_list
        sentence['stanford_dep_basic_head_list'] = dep_head_index_list
        sentence['stanford_dep_enhance'] = dep_enh_label_list
        sentence['stanford_dep_enhance_head'] = dep_enh_head_list
        sentence['stanford_dep_enhance_head_list'] = dep_enh_head_index_list

    return data_json

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--inp", dest="file_in", default='test', help="input CONLL format file")
    parser.add_argument("--out", dest="file_out", default='test', help="output json file")
    args = parser.parse_args()
    file_in, file_out = args.file_in, args.file_out

    # convert ontonotes format into json format for post-processing
    print("converting conll format into json with labels")
    data_json = read_ontonotes_file(file_in)

    print("Using spacy to get dependency tree and its labels")
    data_json = get_dep_tree(data_json)

    # You need to run the stanford core nlp service first.
    # use Stanford NLP to extract constituency tree and another version of dependency tree
    print("Using stanfort NLP to get constituency/dependency tree and its labels")
    data_json_new = extract_constuency(data_json)

    file_out_json = open(file_out, 'w', encoding='utf-8')
    print(len(data_json_new))
    checked_json = []
    for sent_dict in data_json_new:
        if len(sent_dict) == 16:
            checked_json.append(sent_dict)
    print(len(checked_json))
    json.dump(checked_json, file_out_json)