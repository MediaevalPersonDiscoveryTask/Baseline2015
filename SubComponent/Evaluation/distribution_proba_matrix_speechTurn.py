"""
Distribution of the probability between positive and negative pair of speech turns

Usage:
  proba_mat_speechTurn.py <video_list> <matrix_path> <st_seg> <reference_speaker>
  proba_mat_speechTurn.py -h | --help
"""

from docopt import docopt
from pyannote.algorithms.tagging import ArgMaxDirectTagger
from pyannote.core.matrix import LabelMatrix
from mediaeval_util.repere import align_st_ref, drange
import numpy as np

if __name__ == '__main__':
    args = docopt(__doc__)

    l_true = []
    l_false = []

    for videoID in open(args['<video_list>']).read().splitlines():

        st_to_ref = align_st_ref(args['<st_seg>']+'/'+videoID+'.MESeg', args['<reference_speaker>'], videoID)

        m = LabelMatrix.load(args['<matrix_path>']+'/'+videoID+'.mat')
        # compute score between speech turn and save it
        for s1, t1 in m.get_rows():
            for s2, t2 in m.get_columns():
                if t1 in st_to_ref and t2 in st_to_ref and st_to_ref[t1] == st_to_ref[t2]:
                    l_true.append(m[(s1,t1), (s2,t2)])
                else:
                    l_false.append(m[(s1,t1), (s2,t2)])

    print len(l_false), len(l_true)

    l_range = list(drange(0.0, 1.0, 0.01))

    hist_1 = np.histogram(l_true, l_range)    
    hist_0 = np.histogram(l_false, l_range)

    for i in range(len(l_range)-1):
        print str(l_range[i]).replace('.', ','),
        print str(round(float(hist_0[0][i])/float(len(l_false))*100,2)).replace('.', ','),
        print str(round(float(hist_1[0][i])/float(len(l_true))*100,2)).replace('.', ',')


