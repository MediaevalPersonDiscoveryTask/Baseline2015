"""
Distribution of the probability between positive and negative pair of speech turns

Usage:
  proba_mat_speechTurn.py <video_list> <faceTracking> <l2MatrixPath> <facePositionReferencePath>
  proba_mat_speechTurn.py -h | --help
"""

from docopt import docopt
from pyannote.algorithms.tagging import ArgMaxDirectTagger
from pyannote.parser import MDTMParser
from mediaeval_util.repere import align_st_ref, drange
from sklearn.externals import joblib
import numpy as np

if __name__ == '__main__':
    args = docopt(__doc__)

    l_true = []
    l_false = []

    for videoID in open(args['<video_list>']).read().splitlines():

        facetracks = {}
        for line in open(args['<faceTracking>']+'/'+videoID+'.facetrack').read().splitlines():
            frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
            facetracks.setdefault(frameID, {})
            facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h
        ref_f = read_ref_facetrack_position(args['<facePositionReferencePath>']+'/'+videoID+'.position', 0)
        facetrack_vs_ref = align_facetrack_ref(ref_f, facetracks)

        for line in open(args['<matrix_path>']+'/'+videoID+'.mat').read().splitlines():
            ft1, ft2, proba = line.split(' ')
            ft1, ft2 = int(ft1), int(ft2)
            if ft1 in facetrack_vs_ref and ft2 in facetrack_vs_ref:
                proba = float(proba)
                if facetrack_vs_ref[ft1] == facetrack_vs_ref[ft2]:
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


