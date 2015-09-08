"""
compute probability that a facetrack is speaking

Usage:
  proba_speaking_face.py <videoID> <faceTrackSegmentation> <SpeakingFaceDescriptor> <speechTurnSegmentation> <idx> <modelProbaSpeakingFace> <probaSpeakingFace>
  proba_speaking_face.py -h | --help
"""

from docopt import docopt
from mediaeval_util.repere import IDXHack, MESegParser
from sklearn.externals import joblib
import numpy as np

if __name__ == '__main__':
    args = docopt(__doc__)

    # load classifier model
    clf = joblib.load(args['<modelProbaSpeakingFace>']) 

    # read speech turn segmentation
    st_seg, confs, timeToFrameID = MESegParser(args['<speechTurnSegmentation>'], args['<videoID>'])
    facetrack, confs, timeToFrameID = MESegParser(args['<faceTrackSegmentation>'], args['<videoID>'])
    faceID_to_TrackID = {}
    for s, trackID, faceID in facetrack.itertracks(label=True):
        faceID_to_TrackID[faceID] = trackID
    # fct to convert frameID to timestamp 
    frame2time = IDXHack(args['<idx>'])

    # read visual descriptors
    visual_desc = {}
    for line in open(args['<SpeakingFaceDescriptor>']):
        l = line[:-1].split(' ')
        trackID_face = faceID_to_TrackID[l[1]]
        timestamp = frame2time(int(l[0]), 0.0)        
        for s, trackID_st, st in st_seg.itertracks(label=True):
            if timestamp >= s.start and timestamp <= s.end:
                visual_desc.setdefault(trackID_st, {})
                visual_desc[trackID_st].setdefault(trackID_face, []).append(map(float, l[2:]))

    # compute a descriptor with information of speaker segment and compute the corresponding probability that the face is speaking
    proba_st_facetrack = {}
    for s, trackID_st, st in st_seg.itertracks(label=True):
        if trackID_st in visual_desc:
            proba_st_facetrack[trackID_st] = {}
            l = []
            for trackID_face in visual_desc[trackID_st]: 
                l.append(float(len(visual_desc[trackID_st][trackID_face])))
            best_facetrack_cooc_duration = max(l)
            for trackID_face in visual_desc[trackID_st]:
                proba_st_facetrack[trackID_st][trackID_face] = []
                duration = float(len(visual_desc[trackID_st][trackID_face]))
                propdur_best_duration_faceID = duration/best_facetrack_cooc_duration
                prop_dur_spk_dur = duration/(s.duration*25)
                for desc in visual_desc[trackID_st][trackID_face]: 
                    proba_st_facetrack[trackID_st][trackID_face].append(clf.predict_proba([[propdur_best_duration_faceID, prop_dur_spk_dur]+desc])[0][1])

    # save matrix
    fout = open(args['<probaSpeakingFace>'], 'w')
    for trackID_st in sorted(proba_st_facetrack):
        for trackID_face in sorted(proba_st_facetrack[trackID_st]):
            proba = np.median(np.array(proba_st_facetrack[trackID_st][trackID_face]))            
            fout.write(str(trackID_st)+' '+str(trackID_face)+' '+str(proba)+'\n')
    fout.close()

