"""
Learn a normalisation model to compute svs matrix

Usage:
  9_learn_normalisation_model.py <videoList> <faceTrackPosition> <l2MatrixPath> <facePositionReferencePath> <modell2ToProba>
  9_learn_normalisation_model.py -h | --help
"""

from docopt import docopt
from mediaeval_util.repere import read_ref_facetrack_position, align_facetrack_ref
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np
from scipy import spatial
import pickle

if __name__ == '__main__':
    # read args
    args = docopt(__doc__)

    X = []
    Y = []
    for videoID in open(args['<videoList>']).read().splitlines():
        print videoID

        # find alignement between facetrack and reference
        facetracks = {}
        l_faceID = set([])
        for line in open(args['<faceTrackPosition>']+'/'+videoID+'.facetrack').read().splitlines():
            frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
            facetracks.setdefault(frameID, {})
            facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h
            l_faceID.add(faceID)
        ref_f = read_ref_facetrack_position(args['<facePositionReferencePath>'], videoID, 0)
        facetrack_vs_ref = align_facetrack_ref(ref_f, facetracks)

        # mapping between indice in the matrix and faceID
        l_faceID = sorted(l_faceID)
        N = len(l_faceID)
        face_to_indice = {}
        for i in range(N):
            face_to_indice[l_faceID[i]] = i

        # read matrix
        l2 = pickle.load(open(args['<l2MatrixPath>']+'/'+videoID+'.mat', "rb" ))

        l2 = spatial.distance.squareform(l2, checks=False)

        # file X and Y list
        for i in range(N):
            for j in range(i+1, N):
                if l_faceID[i] in facetrack_vs_ref and l_faceID[j] in facetrack_vs_ref:
                    X.append([l2[i][j]])
                    if facetrack_vs_ref[l_faceID[i]] == facetrack_vs_ref[l_faceID[j]]:
                        Y.append(1)
                    else:
                        Y.append(0)

    # train model and save it
    clf = CalibratedClassifierCV(LogisticRegression(), method='sigmoid')
    clf.fit(X, Y) 
    joblib.dump(clf, args['<modell2ToProba>']) 
