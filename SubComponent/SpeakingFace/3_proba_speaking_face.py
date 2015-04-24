"""
compute probability that a facetrack is speaking

Usage:
  proba_speaking_face.py <descriptors> <input_model_file> <output_mat>
  proba_speaking_face.py -h | --help
"""

from docopt import docopt
from sklearn.externals import joblib

if __name__ == '__main__':
    args = docopt(__doc__)

    clf = joblib.load(args['<input_model_file>']) 

    fout = open(args['<output_mat>'], 'w')
    for line in open(args['<descriptors>']).read().splitlines():
        l = line.split(' ')
        fout.write(l[0]+' '+l[1]+' '+str(clf.predict_proba([map(float, l[2:])])[0][1])+'\n')
    fout.close()