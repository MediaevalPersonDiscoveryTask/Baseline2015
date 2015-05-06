"""
Learn a classifier model to compute the probability that a facetrack is speaking

Usage:
  learn_model_proba_speaking_face.py <videoList> <faceTrackingPosition> <descFaceSelection> <facePositionReferencePath> <modelFaceSelection>
  learn_model_proba_speaking_face.py -h | --help
"""

from docopt import docopt
import numpy as np
from mediaeval_util.repere import read_ref_facetrack_position, align_facetrack_ref, align_st_ref
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import random

if __name__ == '__main__':
    # read arguments       
    args = docopt(__doc__)

    X, Y = [], []
    for videoID in open(args['<videoList>']).read().splitlines():
        # find the name corresponding to facetracks
        ref_f = read_ref_facetrack_position(args['<facePositionReferencePath>'], videoID, 3)
        facetracks = {}
        l_facetrack_used_to_learn_model = set([])
        for line in open(args['<faceTrackingPosition>']+'/'+videoID+'.facetrack').read().splitlines():
            frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
            facetracks.setdefault(frameID, {})
            facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h
            if frameID in ref_f:
                l_facetrack_used_to_learn_model.add(faceID)            
        facetrack_vs_ref = align_facetrack_ref(ref_f, facetracks)

        # read visual descriptors
        desc = {}
        for line in open(args['<descFaceSelection>']+'/'+videoID+'.desc'):
            l = line[:-1].split(' ')
            faceID = int(l[1])
            if faceID in l_facetrack_used_to_learn_model:
                X.append(map(float, l[1:]))
                if faceID in facetrack_vs_ref: Y.append(1)
                else: Y.append(0)

    # train model
    clf = LogisticRegression()
    clf.fit(X, Y)
    # save model     
    joblib.dump(clf, args['<modelFaceSelection>']) 
