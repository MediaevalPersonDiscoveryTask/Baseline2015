"""
Learn a classifier model to compute the probability that a facetrack is speaking

Usage:
  learn_model_proba_speaking_face.py <video_list> <corpusPath> <dataPath.lst> <facetrack_pos> <descriptor_path> <reference_head_path> <reference_speaker_path> <output_model_file>
  learn_model_proba_speaking_face.py -h | --help
"""

from docopt import docopt
import numpy as np
from mediaeval_util.repere import IDXHack, read_ref_facetrack_position, align_facetrack_ref, align_st_ref
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import random

if __name__ == '__main__':
    # read arguments       
    args = docopt(__doc__)
    # dictionnary with the IDX file
    idxPath = {}
    for path in open(args['<dataPath.lst>']).read().splitlines():
        videoID, wave_file, video_avi_file, video_mpeg_file, trs_file, xgtf_file, idx_file = path.split(' ')
        idxPath[videoID] = args['<corpusPath>']+'/'+idx_file

    X, Y = [], []
    for videoID in open(args['<video_list>']).read().splitlines():
        # find the name corresponding to facetracks
        ref_f = read_ref_facetrack_position(args['<reference_head_path>']+'/'+videoID+'.position', 3)
        facetracks = {}
        l_facetrack_used_to_learn_model = set([])
        for line in open(args['<facetrack_pos>']+'/'+videoID+'.facetrack').read().splitlines():
            frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
            facetracks.setdefault(frameID, {})
            facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h
            if frameID in ref_f:
                l_facetrack_used_to_learn_model.add(faceID)
        facetrack_vs_ref = align_facetrack_ref(ref_f, facetracks)

        # read speaker reference
        ref_spk = {}
        for line in open(args['<reference_speaker_path>']+'/'+videoID+'.atseg').read().splitlines():
            v, startTime, endTime, spkName = line.split(' ') 
            ref_spk[spkName] = [float(startTime), float(endTime)]

        # fct to convert frameID to timestamp 
        frame2time = IDXHack(idxPath[videoID])

        # read visual descriptors
        visual_desc = {}
        for line in open(args['<descriptor_path>']+'/'+videoID+'.desc'):
            l = line[:-1].split(' ')
            faceID = int(l[1])
            if faceID in l_facetrack_used_to_learn_model:
                timestamp = frame2time(int(l[0]), 0.0)
                for spk, time in ref_spk.items():
                    if timestamp >= time[0] and timestamp <= time[1]:
                        visual_desc.setdefault(spk, {})
                        visual_desc[spk].setdefault(faceID, []).append(map(float, l[2:]))

        # compute a descriptor with information of speaker segment
        for spk in visual_desc:
            startTime, endTime = ref_spk[spk]
            spk_duration = (endTime-startTime)*25
            l = []
            for faceID in visual_desc[spk]:
                l.append(float(len(visual_desc[spk][faceID])))
            best_facetrack_cooc_duration = max(l)

            for faceID in visual_desc[spk]:
                facetrack_cooc_duration = float(len(visual_desc[spk][faceID]))
                propdur_best_duration_faceID = facetrack_cooc_duration/best_facetrack_cooc_duration
                prop_dur_spk_dur = facetrack_cooc_duration/spk_duration

                # find if the face correspond to the speaker
                SpeakingFace = 0
                if faceID in facetrack_vs_ref:
                    if facetrack_vs_ref[faceID] == spk:
                        SpeakingFace = 1

                for desc in visual_desc[spk][faceID]:
                    X.append([propdur_best_duration_faceID, prop_dur_spk_dur]+desc)
                    Y.append(SpeakingFace)
    # train model
    clf = CalibratedClassifierCV(LogisticRegression(), method='sigmoid')
    clf.fit(X, Y)
    # save model     
    joblib.dump(clf, args['<output_model_file>']) 
