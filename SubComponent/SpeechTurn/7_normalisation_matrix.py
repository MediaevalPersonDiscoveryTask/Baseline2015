"""
Compute speaker versus speaker distance

Usage:
  normalisation_matrix.py <videoID> <BICMatrix> <modelBICToProba> <probaMatrix>
  normalisation_matrix.py -h | --help
"""

from docopt import docopt
from sklearn.externals import joblib
from pyannote.core.matrix import LabelMatrix

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    # open model
    clas = joblib.load(args['<modelBICToProba>']) 

    # load BIC matrix
    m = LabelMatrix.load(args['<BICMatrix>'])

    # compute proba between speech turn
    for s1, t1 in m.get_rows():
        for s2, t2 in m.get_columns():
            new_score = clas.predict_proba([[m[(s1,t1), (s2,t2)]]])[0][1]
            m[(s1,t1), (s2,t2)] = new_score

    # save matrix
    m.save(args['<probaMatrix>'])
