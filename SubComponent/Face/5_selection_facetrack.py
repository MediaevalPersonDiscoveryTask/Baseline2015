"""
Select facetrack

Usage:
  selection_facetrack.py <descFaceSelection> <rawfacetrackPosition> <rawfacetracks> <facetrackPosition> <facetracks> <modelFaceSelection> [--thr=<t>] [--minDuration=<md>]
  selection_facetrack.py -h | --help
Options:
  --thr=<t>           threshold on score [default: 0.4] 
  --minDuration=<md>  minimum duration of a facetrack in second [default: 0.2] 
"""

from docopt import docopt
from sklearn.externals import joblib
import numpy as np

if __name__ == '__main__':
    args = docopt(__doc__)

    # load classifier model
    clf = joblib.load(args['<modelFaceSelection>']) 
    thr = float(args['--thr'])
    minDuration = float(args['--minDuration'])

    facetrack_duration = {}
    for line in open(args['<rawfacetracks>']).read().splitlines():
        v, startTime, endTime, startFrame, endFrame, trackID, faceID, conf = line.split(' ')
        facetrack_duration[faceID] = float(endTime)-float(startTime)

    l_faceID_to_save = []
    for line in open(args['<descFaceSelection>']):
        l = line[:-1].split(' ')
        if clf.predict_proba([map(float, l[1:])])[0][1] > thr and facetrack_duration[l[0]] >= minDuration:
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