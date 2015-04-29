"""
Draw distribution of the probability between positive and negative pair

Usage:
  proba_mat_speechTurn.py <video_list> <matrix_path> <st_seg> <reference_speaker>
  proba_mat_speechTurn.py -h | --help
"""

from docopt import docopt
from pyannote.algorithms.tagging import ArgMaxDirectTagger
from pyannote.parser import MDTMParser
from mediaeval_util.repere import align_st_ref
from sklearn.externals import joblib
import numpy as np

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

if __name__ == '__main__':
    args = docopt(__doc__)

    l_true = []
    l_false = []

    for videoID in open(args['<video_list>']).read().splitlines():

        st_to_ref = align_st_ref(args['<st_seg>'], args['<reference_speaker>'], videoID)

        for line in open(args['<matrix_path>']+'/'+videoID+'.mat').read().splitlines():
            st1, st2, proba = line.split(' ')
            if st1 in st_to_ref and st2 in st_to_ref:
                if st_to_ref[st1] == st_to_ref[st2]:
                    l_true.append(proba)
                else:
                    l_false.append(proba)

    print len(l_false), len(l_true)

    l_range = list(drange(0.0, 1.0, 0.01))

    hist_1 = np.histogram(l_true, l_range)    
    hist_0 = np.histogram(l_false, l_range)

    for i in range(len(l_range)-1):
        print str(l_range[i]).replace('.', ','),
        print str(round(float(hist_0[0][i])/float(len(l_false))*100,2)).replace('.', ','),
        print str(round(float(hist_1[0][i])/float(len(l_true))*100,2)).replace('.', ',')


