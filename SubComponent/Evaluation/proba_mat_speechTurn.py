"""
Draw distribution of the probability between positive and negative pair

Usage:
  proba_mat_speechTurn.py <video_list> <matrix_path> <st_seg> <reference_speaker>
  proba_mat_speechTurn.py -h | --help
"""

from docopt import docopt
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

        st_seg = []
        for line in open(args['<st_seg>']+videoID+'.mdtm').read().splitlines():
            v, p, start, dur, spk, na, na, st = line.split(' ')
            st_seg.append([float(start), float(start)+float(dur), st])
        ref_spk = []
        for line in open(args['<reference_speaker>']+videoID+'.atseg').read().splitlines():
            v, startTime, endTime, spkName = line.split(' ') 
            ref_spk.append([float(startTime), float(endTime), spkName])
        ref_spk.sort()

        st_vs_ref = align_st_ref(st_seg, ref_spk)

        for line in open(args['<matrix_path>']+videoID+'.mat').read().splitlines():
            st1, st2, proba = line.split(' ')

            if st1 in st_vs_ref and st2 in st_vs_ref:
                proba = float(proba)
                if st_vs_ref[st1] == st_vs_ref[st2]:
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


