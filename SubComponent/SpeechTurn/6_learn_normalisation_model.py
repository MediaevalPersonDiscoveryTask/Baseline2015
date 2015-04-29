"""
Learn a normalisation model to compute svs matrix

Usage:
  6_learn_normalisation_model.py <video_list> <segmentation_path> <matrix_path> <reference_path> <output_model_file>
  6_learn_normalisation_model.py -h | --help
"""

from docopt import docopt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from mediaeval_util.repere import align_st_ref

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    X, Y = [], []
    for line in open(args['<video_list>']).read().splitlines():
        videoID = line.split('\t')[0]
        # find alignement between speech turne and reference
        st_to_ref = align_st_ref(args['<segmentation_path>'], args['<reference_path>'], videoID)
        # read BIC matrix
        for line in open(args['<matrix_path>']+'/'+videoID+'.mat').read().splitlines():
            st1, st2, BIC_dist = line.split(' ')
            if st1 in st_to_ref and st2 in st_to_ref:
                X.append([float(BIC_dist)])
                if st_to_ref[st1] == st_to_ref[st2]:
                    Y.append(1)
                else:
                    Y.append(0)
    # train model
    clf = CalibratedClassifierCV(LogisticRegression(), method='sigmoid')
    clf.fit(X, Y) 
    # save model
    joblib.dump(clf, args['<output_model_file>']) 
