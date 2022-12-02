""" Usage:
   pr_plot --in=DIR_NAME --out=OUTPUT_FILENAME 

Options:
  --in=DIR_NAME            Folder in which to search for *.dat files, all of which should be in a P/R column format (outputs from benchmark.py)
  --out=OUTPUT_FILENAME    Output filename, filetype will determine the format. Possible formats: pdf, pgf, png


"""
from sklearn import metrics
import os
import ntpath
import numpy as np
from glob import glob
from docopt import docopt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level = logging.INFO)
import pandas as pd
def trend_name(path):
    ''' return a system trend name from dat file path '''
    head, tail = ntpath.split(path)
    ret = tail or ntpath.basename(head)
    return ret.split('.')[0]

# def get_pr(path):
#     ''' get PR curve from file '''
#     with open(path) as fin:
#         # remove header line
#         fin.readline()
#         [p, r] = list(zip(*[[float(x) for x in line.strip().split('\t')] for line in fin]))
#         return p, r

def get_pr(path):
    ''' get PR curve from file '''
    with open(path) as fin:
        # remove header line
        fin.readline()
        [p, r] = list(zip(*[[float(x) for x in line.strip().split('\t')] for line in fin]))
        p = np.array(p)
        r = np.array(r)
        df = pd.DataFrame(data=np.array([p, r]).transpose(), columns=["p", "r"])
        return df

if __name__ == '__main__':
    args = docopt(__doc__)
    input_folder = args['--in']
    output_file = args['--out']
    
    # plot graphs for all *.dat files in input path
    files = glob(os.path.join(input_folder, '*.dat'))
    for _file in files:
        name = trend_name(_file)
        print(_file)
        df = get_pr(_file)
        df['f1'] = 2 * (df['r'] * df['p']) / (df['r'] + df['p'])
        print('max f1 is ' + str(df['f1'].max()))
        df = df[df['r'] > 0]
        r = tuple(list(df['r']))
        p = tuple(list(df['p']))
        auc = metrics.auc(df['r'].values,df['p'].values)
        print('auc is ' + str(auc))

        plt.plot(r, p, label=name)

    # Set figure properties and save
    logging.info("Plotting P/R graph to {}".format(output_file))
    plt.ylim([0.0,1.0])
    plt.xlim([0.0, 0.5])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    plt.savefig(output_file)
