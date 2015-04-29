"""
Compute speaker versus speaker distance

Usage:
  normalisation_matrix.py <input_mat> <model_file> <output_mat>
  normalisation_matrix.py -h | --help
"""

from docopt import docopt
from sklearn.externals import joblib

if __name__ == '__main__':
    args = docopt(__doc__)

    # open model
    clas = joblib.load(args['<model_file>']) 

    # compute score between speech turn and save it
    fout = open(args['<output_mat>'], 'w')
    for line in open(args['<input_mat>']).read().splitlines():
        ft1, ft2, dist = line.split(' ')
        fout.write(ft1+' '+ft2+' '+str(clas.predict_proba([[float(dist)]])[0][1])+'\n')
    fout.close()
