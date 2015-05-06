"""
compute probability that a facetrack is speaking

Usage:
  proba_speaking_face.py <videoID> <descFaceSelection> <rawfacetrackPosition> <rawfacetracks> <facetrackPosition> <facetracks> <modelFaceSelection>
  proba_speaking_face.py -h | --help
"""

from docopt import docopt
from sklearn.externals import joblib
import numpy as np

if __name__ == '__main__':
    args = docopt(__doc__)

    # load classifier model
    clf = joblib.load(args['<modelFaceSelection>']) 

    desc = {}
    l_faceID_to_save = []
    for line in open(args['<descFaceSelection>']):
        l = line[:-1].split(' ')
        if clf.predict([map(float, l[1:])])[0] == 1:
            print l[1]
            l_faceID_to_save.append(l[1])

    fout = open(args['<facetrackPosition>'], 'w')
    for line in open(args['<rawfacetrackPosition>']):
        print line.split(' ')[1]
        if line.split(' ')[1] in l_faceID_to_save:
            print 'ok1'
            fout.write(line)
    fout.close()

    fout = open(args['<facetracks>'], 'w')
    for line in open(args['<rawfacetracks>']):
        print line.split(' ')[6]
        if line.split(' ')[6] in l_faceID_to_save:
            print 'ok2'
            fout.write(line)
    fout.close()