"""
compute probability that a facetrack is speaking

Usage:
  proba_speaking_face.py <descFaceSelection> <rawfacetrackPosition> <rawfacetracks> <facetrackPosition> <facetracks> <modelFaceSelection> [--thr=<t>]
  proba_speaking_face.py -h | --help
Options:
  --thr=<t>     threshold on score [default: 0.4]  
"""

from docopt import docopt
from sklearn.externals import joblib
import numpy as np

if __name__ == '__main__':
    args = docopt(__doc__)

    # load classifier model
    clf = joblib.load(args['<modelFaceSelection>']) 
    thr = float(args['--thr'])
    desc = {}
    l_faceID_to_save = []
    for line in open(args['<descFaceSelection>']):
        l = line[:-1].split(' ')
        if clf.predict_proba([map(float, l[1:])])[0][1] > thr:
            l_faceID_to_save.append(l[0])

    fout = open(args['<facetrackPosition>'], 'w')
    for line in open(args['<rawfacetrackPosition>']):
        if line.split(' ')[1] in l_faceID_to_save:
            fout.write(line)
    fout.close()

    fout = open(args['<facetracks>'], 'w')
    for line in open(args['<rawfacetracks>']):
        if line.split(' ')[6] in l_faceID_to_save:
            fout.write(line)
    fout.close()