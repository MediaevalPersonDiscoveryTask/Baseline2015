"""
Learn a normalisation model to compute svs matrix

Usage:
  6_learn_normalisation_model.py <video_list> <facetrack_pos> <matrix_path> <reference_head_position_path> <output_model_file>
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
    for videoID in open(args['<video_list>']).read().splitlines():

        # find alignement between facetrack and reference
        facetracks = {}
        for line in open(args['<facetrack_pos>']+'/'+videoID+'.facetrack').read().splitlines():
            frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
            facetracks.setdefault(frameID, {})
            facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h
        ref_f = read_ref_facetrack_position(args['<reference_head_position_path>']+'/'+videoID+'.position', 0)
        facetrack_vs_ref = align_facetrack_ref(ref_f, facetracks)

        # read matrix
        for line in open(args['<matrix_path>']+'/'+videoID+'.mat').read().splitlines():
            ft1, ft2, dist = line.split(' ')
            if ft1 in facetrack_vs_ref and ft2 in facetrack_vs_ref:
                X.append([float(dist)])
                if facetrack_vs_ref[ft1] == facetrack_vs_ref[ft2]:
                    Y.append(1)
                else:
                    Y.append(0)

    # train model
    clf = CalibratedClassifierCV(LogisticRegression(), method='sigmoid')
    clf.fit(X, Y) 
    # save model
    joblib.dump(clf, args['<output_model_file>']) 
