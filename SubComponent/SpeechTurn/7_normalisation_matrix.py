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

    # compute proba between speech turn and save matrix
    fout = open(args['<probaMatrix>'], 'w')
    for s1, t1 in m.get_rows():
        for s2, t2 in m.get_columns():
            if t2 > t1:
                new_score = clas.predict_proba([[m[(s1,t1), (s2,t2)]]])[0][1]
                m[(s1,t1), (s2,t2)] = new_score
                fout.write(str(t1)+" "+str(t2)+" "+str(new_score)+'\n')
    fout.close()
