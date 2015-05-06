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
        if clf.predict([map(float, l[1:])]) == 1:
            l_faceID_to_save.append(l[1])

    fout = open(args['<facetrackPosition>'])
    for line in open(args['<rawfacetrackPosition>']):
        if line.split(' ')[1] in l_faceID_to_save
            fout.write(line)
    fout.close()

    fout = open(args['<facetracks>'])
    for line in open(args['<rawfacetracks>']):
        if line.split(' ')[6] in l_faceID_to_save
            fout.write(line)
    fout.close()