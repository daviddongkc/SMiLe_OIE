'''
Usage:
   benchmark --gold=GOLD_OIE --out=OUTPUT_FILE --tabbed=TABBED_OIE
'''

# import nltk
# nltk.download('stopwords')

import docopt
import string
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score
import re
import logging
logging.basicConfig(level=logging.INFO)
import pandas as pd
from sklearn import metrics

# from benchmark.oie_readers.stanfordReader import StanfordReader
# from benchmark.oie_readers.ollieReader import OllieReader
# from benchmark.oie_readers.reVerbReader import ReVerbReader
# from benchmark.oie_readers.clausieReader import ClausieReader
# from benchmark.oie_readers.openieFourReader import OpenieFourReader
# from benchmark.oie_readers.propsReader import PropSReader
from benchmark.oie_readers.tabReader import TabReader
from benchmark.oie_readers.goldReader import GoldReader
from benchmark.matcher import Matcher

class Benchmark:
    ''' Compare the gold OIE dataset against a predicted equivalent '''
    def __init__(self, gold_fn):
        ''' Load gold Open IE, this will serve to compare against using the compare function '''
        gr = GoldReader()
        gr.read(gold_fn)
        self.gold = gr.oie

    def compare(self, predicted, matchingFunc, output_fn):
        ''' Compare gold against predicted using a specified matching function. Outputs PR curve to output_fn '''

        y_true, y_scores = [], []
        total_gold, total_unmatched, total_predict = 0, 0, 0
        correctTotal, unmatchedCount = 0, 0
        predicted = Benchmark.normalizeDict(predicted)
        gold = Benchmark.normalizeDict(self.gold)
        for sent, goldExtractions in list(gold.items()):
            if sent not in predicted:
                # The extractor didn't find any extractions for this sentence
                unmatchedCount += len(goldExtractions)
                correctTotal += len(goldExtractions)
                continue

            predictedExtractions = predicted[sent]
            total_gold += len(goldExtractions)
            total_predict += len(predictedExtractions)

            for goldEx in goldExtractions:
                correctTotal += 1
                found = False
                for predictedEx in predictedExtractions:
                    if matchingFunc(goldEx, predictedEx, ignoreStopwords=True, ignoreCase=True):
                        y_true.append(1)
                        y_scores.append(predictedEx.confidence)
                        predictedEx.matched.append(output_fn)
                        found = True
                        break
                if not found:
                    total_unmatched += 1

            for predictedEx in [x for x in predictedExtractions if (output_fn not in x.matched)]:
                # Add false positives
                y_true.append(0)
                y_scores.append(predictedEx.confidence)

        print('the number of unmatched is: ', total_unmatched)

        try:
            recall = (total_gold - total_unmatched) / total_gold
            precision = (total_gold - total_unmatched) / total_predict
            my_F1 = 2.0 * recall * precision / (recall + precision)
            my_auc = roc_auc_score(y_true, y_scores)
            print("my F1: ", my_F1, "my AUC: ", my_auc)
        except ZeroDivisionError:
            my_F1, my_auc = 0.0, 0.0


        recallMultiplier = (correctTotal - unmatchedCount)/float(correctTotal)

        # recall on y_true, y  (r')_scores computes |covered by extractor| / |True in what's covered by extractor|
        # to get to true recall we do r' * (|True in what's covered by extractor| / |True in gold|) = |true in what's covered| / |true in gold|
        p, r = Benchmark.prCurve(np.array(y_true), np.array(y_scores), recallMultiplier=recallMultiplier)
        # write PR to file
        with open(output_fn, 'w') as fout:
            fout.write('{0}\t{1}\n'.format("Precision", "Recall"))
            for cur_p, cur_r in sorted(zip(p, r), key=lambda cur_p_cur_r: cur_p_cur_r[1]):
                fout.write('{0}\t{1}\n'.format(cur_p, cur_r))

        data = np.array([p, r]).transpose()
        df = pd.DataFrame(data=data, columns=["p", "r"])
        df['f1'] = 2 * (df['r'] * df['p']) / (df['r'] + df['p'])
        best_F1 = df['f1'].max()
        print('max f1 is ' + str(df['f1'].max()))
        df = df[df['r'] > 0]
        r = tuple(list(df['r']))
        p = tuple(list(df['p']))
        try:
            auc = metrics.auc(df['r'].values, df['p'].values)
        except ValueError:
            auc = 0.0
        print('auc is ' + str(auc))

        return my_F1, my_auc, best_F1, auc




    @staticmethod
    def prCurve(y_true, y_scores, recallMultiplier):
        # Recall multiplier - accounts for the percentage examples unreached by
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        recall = recall * recallMultiplier
        return precision, recall

    # Helper functions:
    @staticmethod
    def normalizeDict(d):
        return dict([(Benchmark.normalizeKey(k), v) for k, v in list(d.items())])

    @staticmethod
    def normalizeKey(k):
        return Benchmark.removePunct(str(Benchmark.PTB_unescape(k.replace(' ',''))))

    @staticmethod
    def PTB_escape(s):
        for u, e in Benchmark.PTB_ESCAPES:
            s = s.replace(u, e)
        return s

    @staticmethod
    def PTB_unescape(s):
        for u, e in Benchmark.PTB_ESCAPES:
            s = s.replace(e, u)
        return s

    @staticmethod
    def removePunct(s):
        return Benchmark.regex.sub('', s)

    # CONSTANTS
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    # Penn treebank bracket escapes
    # Taken from: https://github.com/nlplab/brat/blob/master/server/src/gtbtokenize.py
    PTB_ESCAPES = [('(', '-LRB-'),
                   (')', '-RRB-'),
                   ('[', '-LSB-'),
                   (']', '-RSB-'),
                   ('{', '-LCB-'),
                   ('}', '-RCB-'),]


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    logging.debug(args)

    if args['--tabbed']:
        predicted = TabReader()
        predicted.read(args['--tabbed'])

    b = Benchmark(args['--gold'])
    out_filename = args['--out']

    logging.info("Writing PR curve of {} to {}".format(predicted.name, out_filename))
    b.compare(predicted=predicted.oie, matchingFunc=Matcher.syntacticHeadMatch, output_fn=out_filename)
