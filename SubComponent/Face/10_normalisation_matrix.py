"""
Compute speaker versus speaker distance

Usage:
  normalisation_matrix.py <l2Matrix> <modell2ToProba> <probaMatrix_out>
  normalisation_matrix.py -h | --help
"""

from docopt import docopt
from sklearn.externals import joblib
import pickle
import numpy as np

if __name__ == '__main__':
    args = docopt(__doc__)

    # open model
    clas = joblib.load(args['<modell2ToProba>']) 

    # compute score between speech turn and save it
    y = pickle.load(open(args['<l2Matrix>'], "rb" ) )

    for i in range(len(y)):
        if not np.isnan(y[i]):
            y[i] = clas.predict_proba([[y[i]]])[0][1]
        else:
            y[i] = 0.0
    y = y.astype(np.float16)

    # save matrix
    pickle.dump(y, open(args['<probaMatrix_out>'], "wb" ))

