""" Usage:
        predict_conll --in=INPUT_FILE --out=OUTPUT_FILE
"""
# revert 5/12/2019

import sys
import os

sys.path.append(os.getcwd())
# learn more about python imports and relative paths
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.models.archival import archive_model
from allennlp.predictors import Predictor
import json
import os
import numpy as np
import pandas as pd
from docopt import docopt
from allennlp.common.util import import_submodules
from tqdm import tqdm


def align_probs(instance_result, labels):
    # method to align model output probabilities with the extracted tags
    probs = instance_result['class_probabilities']
    words = instance_result['words']
    tags = instance_result['tags']

    # demarcating the predicate index based on the index of the verb in the sentence. May need to
    # reconsider if the predicate form does not appear in the sentence
    pred_id = words.index(instance_result['verb'])
    # max_probs = np.amax(probs, axis = 1)
    # max_index = np.argmax(probs,axis=1)
    # max_label = map(lambda x: labels[x], max_index)
    tag_index = [labels.index(x) for x in tags]
    tag_probs = [probs[i, j] for i, j in enumerate(tag_index)]
    pred_ids = [pred_id] * len(tags)
    word_ids = list(range(len(tags)))
    df_dict = {'word_id': word_ids, 'words': words, 'pred_id': pred_ids, 'tags': tags, 'probs': tag_probs}
    # df = pd.DataFrame.from_dict(df_dict)
    return df_dict


if __name__ == "__main__":
    import_submodules('simile_oie')
    args = docopt(__doc__)
    input_fn = args["--in"]
    output_fn = args["--out"]

    labels = []
    # define the path to the model and define the conll file to write evaluation to
    # define path to sentences to run the model over
    # sentence_file = '../../data/raw_sentences/lsoie_wiki_test.sents.txt'
    sentence_file = "/home/kc/NLP/simile_oie/data/raw_sentences/lsoie_wiki_test.sents.txt"
    fn = open(sentence_file, "r", encoding='utf-8')
    lines = fn.readlines()

    model_path = input_fn
    output_path = output_fn
    # clear output file
    if os.path.exists(output_path):
        os.remove(output_path)

    with open(model_path + 'vocabulary/labels.txt', "r") as vocab:
        for label in vocab:
            labels.append(label.rstrip())

    # including this redirection in order to load the best model only
    if os.path.exists(model_path + 'model.tar.gz'):
        os.remove(model_path + 'model.tar.gz')

    archive_model(model_path, 'best.th')
    print('best model archived')

    archive = load_archive(model_path + 'model.tar.gz', cuda_device=0)

    predictor = Predictor.from_archive(archive, 'oie')
    # iterate through sentences
    instance_iterator = 0

    # sentences = 'tests/fixtures/oie_test.jsonl'
    print('starting to predict on sents')
    for line in tqdm(lines):
        with open(output_path, 'a') as f:
            inp = {"sentence": line.strip()}
            # run model on sentence
            result = predictor.predict_json(inp)
            for instance_result in result:
                df = align_probs(instance_result, labels)
                # write to conll file
                lines = [str(wi) + '\t' + str(w) + '\t' + str(pi) + '\t' + str(t) + '\t' + str(p) for
                         wi, w, pi, t, p in
                         zip(df['word_id'], df['words'], df['pred_id'], df['tags'], df['probs'])]
                for line in lines:
                    f.write(line)
                    f.write('\n')
                f.write('\n')
