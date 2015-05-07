"""
compute probability that a facetrack is speaking

Usage:
  proba_speaking_face.py <videoID> <SpeakingFaceDescriptor> <speechTurnSegmentation> <idx> <modelProbaSpeakingFace> <probaSpeakingFace>
  proba_speaking_face.py -h | --help
"""

from docopt import docopt
from mediaeval_util.repere import IDXHack
from sklearn.externals import joblib
import numpy as np

if __name__ == '__main__':
    args = docopt(__doc__)

    # load classifier model
    clf = joblib.load(args['<modelProbaSpeakingFace>']) 

    # read speech turn segmentation
    st_seg, confs, timeToFrameID = MESegParser(args['<speechTurnSegmentation>'], args['<videoID>'])

    st_seg = {}
    for line in open(args['<speechTurnSegmentation>']).read().splitlines():
        v, p, start, dur, spk, na, na, st = line.split(' ')
        st_seg[st] = [float(start), float(start)+float(dur)]

    # fct to convert frameID to timestamp 
    frame2time = IDXHack(args['<idx>'])

    # read visual descriptors
    visual_desc = {}
    for line in open(args['<SpeakingFaceDescriptor>']):
        l = line[:-1].split(' ')
        faceID = int(l[1])
        timestamp = frame2time(int(l[0]), 0.0)        
        for s, t, st in st_seg.itertracks(label=True):
            if timestamp >= s.start and timestamp <= s.end:
                visual_desc.setdefault(st, {})
                visual_desc[st].setdefault(faceID, []).append(map(float, l[2:]))

    # compute a descriptor with information of speaker segment and compute the corresponding probability that the face is speaking
    proba_st_facetrack = {}
    for s, t, st in st_seg.itertracks(label=True):
        if st in visual_desc:
            proba_st_facetrack[st] = {}
            l = []
            for faceID in visual_desc[st]: l.append(float(len(visual_desc[st][faceID])))
            best_facetrack_cooc_duration = max(l)

            for faceID in visual_desc[st]:
                proba_st_facetrack[st][faceID] = []
                duration = float(len(visual_desc[st][faceID]))
                propdur_best_duration_faceID = duration/best_facetrack_cooc_duration
                prop_dur_spk_dur = duration/(s.duration*25)
                for desc in visual_desc[st][faceID]: proba_st_facetrack[st][faceID].append(clf.predict_proba([[propdur_best_duration_faceID, prop_dur_spk_dur]+desc])[0][1])

    # save matrix
    fout = open(args['<probaSpeakingFace>'], 'w')
    for st in proba_st_facetrack:
        for faceID in proba_st_facetrack[st]:
            proba = np.median(np.array(proba_st_facetrack[st][faceID]))            
            fout.write(st+' '+str(faceID)+' '+str(proba)+'\n')
    fout.close()

