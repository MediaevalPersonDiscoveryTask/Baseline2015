"""
compute probability that a facetrack is speaking

Usage:
  proba_speaking_face.py <descriptors> <st_seg> <idx> <input_model_file> <output_mat>
  proba_speaking_face.py -h | --help
"""

from docopt import docopt
from mediaeval_util.repere import IDXHack
from sklearn.externals import joblib
import numpy as np

if __name__ == '__main__':
    args = docopt(__doc__)

    clf = joblib.load(args['<input_model_file>']) 

    frame2time = IDXHack(args['<idx>'])

    st_seg = []
    for line in open(args['<st_seg>']).read().splitlines():
        v, p, start, dur, spk, na, na, st = line.split(' ')
        st_seg.append([float(start), float(start)+float(dur), st])

    proba_st = {}
    for line in open(args['<descriptors>']).read().splitlines():
        l = line.split(' ')
        frameID = int(l[0])
        timestamp = frame2time(frameID, 0.0)
        faceID = int(l[1])
        proba = clf.predict_proba([map(float, l[2:])])[0][1]
        print faceID, timestamp, proba
        for startTime, endTime, st in st_seg:
            if startTime <= timestamp and timestamp <= endTime:
                break
        proba_st.setdefault(st, {})
        proba_st[st].setdefault(faceID, []).append(proba)


    fout = open(args['<output_mat>'], 'w')
    for st in proba_st:
        for faceID, l_proba in proba_st[st].items:
            proba = np.median(np.array(l_proba))
            fout.write(st+' '+faceID+' '+str(proba)+'\n')
    fout.close()
    