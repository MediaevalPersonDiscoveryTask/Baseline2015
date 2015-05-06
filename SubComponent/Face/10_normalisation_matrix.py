"""
Compute speaker versus speaker distance

Usage:
  normalisation_matrix.py <l2Matrix> <modell2ToProba> <probaMatrix>
  normalisation_matrix.py -h | --help
"""

from docopt import docopt
from sklearn.externals import joblib

if __name__ == '__main__':
    args = docopt(__doc__)

    # open model
    clas = joblib.load(args['<modell2ToProba>']) 

    # compute score between speech turn and save it
    fout = open(args['<probaMatrix>'], 'w')
    for line in open(args['<l2Matrix>']).read().splitlines():
        ft1, ft2, nbHoGFaceID1, nbHoGFaceID2, distCenterFaceID1, distCenterFaceID2, l2Distance = line.split(' ')
        fout.write(ft1+' '+ft2+' '+str(clas.predict_proba([[float(l2Distance)]])[0][1])+'\n')
    fout.close()
