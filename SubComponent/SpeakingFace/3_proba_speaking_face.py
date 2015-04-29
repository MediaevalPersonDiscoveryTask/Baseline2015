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

    # load classifier model
    clf = joblib.load(args['<input_model_file>']) 

    # read speech turn segmentation
    st_seg = {}
    for line in open(args['<st_seg>']).read().splitlines():
        v, p, start, dur, spk, na, na, st = line.split(' ')
        st_seg[st] = [float(start), float(start)+float(dur)]

    # fct to convert frameID to timestamp 
    frame2time = IDXHack(args['<idx>'])

    # read visual descriptors
    visual_desc = {}
    for line in open(args['<descriptors>']):
        l = line[:-1].split(' ')
        faceID = int(l[1])
        timestamp = frame2time(int(l[0]), 0.0)
        for st, time in st_seg.items():
            if timestamp >= time[0] and timestamp <= time[1]:
                visual_desc.setdefault(st, {})
                visual_desc[st].setdefault(faceID, []).append(map(float, l[2:]))

    # compute a descriptor with information of speaker segment and compute the corresponding probability that the face is speaking
    proba_st_facetrack = {}
    for st in visual_desc:
        proba_st_facetrack[st] = {}

        startTime, endTime = st_seg[st]
        spk_duration = (endTime-startTime)*25
        l = []
        for faceID in visual_desc[st]:
            l.append(float(len(visual_desc[st][faceID])))
        best_facetrack_cooc_duration = max(l)

        for faceID in visual_desc[st]:
            proba_st_facetrack[st][faceID] = []
            duration = float(len(visual_desc[st][faceID]))
            propdur_best_duration_faceID = duration/best_facetrack_cooc_duration
            prop_dur_spk_dur = duration/spk_duration
            for desc in visual_desc[st][faceID]:
                proba_st_facetrack[st][faceID].append(clf.predict_proba([[propdur_best_duration_faceID, prop_dur_spk_dur]+desc])[0][1])

    # save matrix
    fout = open(args['<output_mat>'], 'w')
    for st in proba_st_facetrack:
        for faceID in proba_st_facetrack[st]:
            proba = np.median(np.array(proba_st_facetrack[st][faceID]))            
            fout.write(st+' '+str(faceID)+' '+str(proba)+'\n')
    fout.close()

