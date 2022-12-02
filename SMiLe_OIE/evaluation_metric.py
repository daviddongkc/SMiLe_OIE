from overrides import overrides
from allennlp.training.metrics.metric import Metric
import os
from collections import defaultdict
import numpy as np
from benchmark.oie_readers.tabReader import TabReader
from benchmark.matcher import Matcher
from benchmark.benchmark import Benchmark

@Metric.register("exact_match")
class Carb(Metric):
    """
    Computes scores according to carb framework
    """
    def __init__(self, output_path: str = None, dev_set: str = None):
        super(Carb, self).__init__()
        self._all_sentences = []
        self._all_tokens = []
        self._all_predictions = []
        self._all_confidences = []
        self._all_predicate_id = []
        self._dev_set = dev_set
        self._epoch_num = 0
        self.map_dic = {"O": "O", "B-V": "P-B", "I-V": "P-I", "B-ARG1": "A1-B", "I-ARG1": "A1-I", "B-ARG2": "A2-B",
                        "I-ARG2": "A2-I", "B-ARG3": "A3-B", "I-ARG3": "A3-I", "B-ARG4": " A4-B", "I-ARG4": "A4-I",
                        "B-ARG5": "A5-B", "I-ARG5": "A5-I", "B-ARG0": "A0-B", "I-ARG0": "A0-I"}
        if output_path is not None and output_path is not '':
            self._output_path = output_path+'/predictions'
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)


    def __call__(self, tokens: list = None, prediction: list = None, predicate_id: list = None, confidence: list = None):
        if len(tokens) == len(prediction) == len(predicate_id) == len(confidence):
            sent_list = [' '.join(x) for x in tokens]
            self._all_sentences.extend(sent_list)
            self._all_tokens.extend(tokens)
            self._all_predictions.extend(prediction)
            self._all_confidences.extend(confidence)
            self._all_predicate_id.extend(predicate_id)
        else:
            print("check prediction output")
            raise Exception

    def flatten_ret_dic(self, dic, num_of_dups):
        """
        Given a dictionary of dictionaries, flatten it
        to a dictionary of lists
        """
        ret = defaultdict(lambda: [])
        for sent, preds_dic in dic.items():
            for pred, exs in preds_dic.items():
                ret[sent].extend(exs * num_of_dups)
        return ret

    def get_metric(self, reset: bool = False):
        if reset:
            ret = defaultdict(lambda: {})
            sents = []
            for tokens, predictions, predicate_id, confidences in zip(self._all_tokens,  self._all_predictions, self._all_predicate_id, self._all_confidences):
                extractions = [(self.map_dic[x], y) for x, y in zip(predictions, confidences)]
                cur_sent = " ".join(tokens)
                sents.append(cur_sent)
                if predicate_id:
                    pred_word = tokens[predicate_id]
                    ret[cur_sent][pred_word] = (((predicate_id, pred_word), extractions),)
                else:
                    try:
                        predicate_id = predictions.index("B-V")
                        pred_word = tokens[predicate_id]
                        ret[cur_sent][pred_word] = (((predicate_id, pred_word), extractions),)
                    except:
                        continue

            dic, sents = self.flatten_ret_dic(ret, 1), list(set(sents))

            extraction_list = []
            for sent in sents:
                for ((pred_ind, pred_word), labels) in dic[sent]:
                    cur_args, cur_arg, probs = [], [], []
                    tokens = sent.split(' ')
                    # collect args
                    for (label, prob), word in zip(labels, tokens):
                        if label.startswith("A"):
                            cur_arg.append(word)
                            probs.append(prob)
                        elif cur_arg:
                            cur_args.append(cur_arg)
                            cur_arg = []
                    # Create extraction
                    if cur_args:
                        extraction_list.append(Extraction(tokens, pred_word, cur_args, probs))

            output_txt_file = self._output_path + "/predictions_epoch_{}.txt".format(self._epoch_num)
            with open(output_txt_file, 'w') as fout:
                fout.write('\n'.join([str(ex) for ex in extraction_list]))

            predicted = TabReader()
            predicted.read(output_txt_file)

            b = Benchmark(self._dev_set)
            output_pr_file = output_txt_file.replace('.txt', '.dat')
            print("Writing PR curve of {} to {}".format(predicted.name, output_pr_file))
            final_F1, my_auc, best_F1, auc = b.compare(predicted=predicted.oie, matchingFunc=Matcher.syntacticHeadMatch, output_fn=output_pr_file)

            self._epoch_num += 1
            self.reset()
            return {'f1': best_F1, 'auc': auc}

        else:
            return {'f1': 0.0, 'auc': 0.0}

    @overrides
    def reset(self):
        self._all_sentences = []
        self._all_tokens = []
        self._all_predictions = []
        self._all_confidences = []
        self._all_predicate_id = []



class Extraction:
    """
    Store and print an OIE extraction
    """
    def __init__(self, sent, pred, args, probs, calc_prob=lambda x: np.mean(np.log(np.clip(x, 1e-5, 1 - 1e-5)))):
        """
        sent - Tokenized sentence - list of strings
        pred - Predicate word
        args - List of arguments (each a string)
        probs - list of float in [0,1] indicating the probability
               of each of the items in the extraction
        calc_prob - function which takes a list of probabilities for each of the
                    items and computes a single probability for the joint occurence of this extraction.
        """
        probs = np.array(probs)
        self.sent = sent
        self.calc_prob = calc_prob
        self.probs = probs
        self.prob = self.calc_prob(self.probs)
        self.pred = pred
        self.args = args

    def __str__(self):
        """
        Format (tab separated):
        Sent, prob, pred, arg1, arg2, ...
        """
        return '\t'.join(map(str, [' '.join(self.sent), self.prob, self.pred, '\t'.join([' '.join(arg) for arg in self.args])]))