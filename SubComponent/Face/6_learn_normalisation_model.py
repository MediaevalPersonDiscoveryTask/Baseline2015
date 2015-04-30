"""
Learn a normalisation model to compute svs matrix

Usage:
  6_learn_normalisation_model.py <videoList> <faceTracking> <l2MatrixPath> <facePositionReferencePath> <modell2ToProba>
  6_learn_normalisation_model.py -h | --help
"""

from docopt import docopt
from mediaeval_util.repere import read_ref_facetrack_position, align_facetrack_ref
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

if __name__ == '__main__':
    # read args
    args = docopt(__doc__)

    X = []
    Y = []
    for videoID in open(args['<videoList>']).read().splitlines():
        print videoID
        # find alignement between facetrack and reference
        facetracks = {}
        for line in open(args['<faceTracking>']+'/'+videoID+'.facetrack').read().splitlines():
            frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
            facetracks.setdefault(frameID, {})
            facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h
        ref_f = read_ref_facetrack_position(args['<facePositionReferencePath>']+'/'+videoID+'.position', 0)
        facetrack_vs_ref = align_facetrack_ref(ref_f, facetracks)

        # read matrix
        for line in open(args['<l2MatrixPath>']+'/'+videoID+'.mat').read().splitlines():
            ft1, ft2, nbHoGFaceID1, nbHoGFaceID2, distCenterFaceID1, distCenterFaceID2, l2Distance = line.split(' ')
            ft1, ft2 = int(ft1), int(ft2)
            if ft1 in facetrack_vs_ref and ft2 in facetrack_vs_ref:
                X.append([float(l2Distance)])
                if facetrack_vs_ref[ft1] == facetrack_vs_ref[ft2]:
                    Y.append(1)
                else:
                    Y.append(0)

    # train model
    clf = CalibratedClassifierCV(LogisticRegression(), method='sigmoid')
    clf.fit(X, Y) 
    # save model
    joblib.dump(clf, args['<modell2ToProba>']) 
