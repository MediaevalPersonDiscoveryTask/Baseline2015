"""
Learn a classifier model to compute the probability that a facetrack is speaking

Usage:
  learn_model_proba_speaking_face.py <video_list> <corpusPath> <dataPath.lst> <facetrack_pos> <st_seg> <descriptor_path> <reference_head_path> <reference_speaker_path> <output_model_file>
  learn_model_proba_speaking_face.py -h | --help
"""

from docopt import docopt
import numpy as np
from mediaeval_util.repere import IDXHack, read_ref_facetrack_position, align_facetrack_ref, align_st_ref
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.externals import joblib
import math


if __name__ == '__main__':
    args = docopt(__doc__)

    tempo_margin = 3

    idxPath = {}
    for path in open(args['<dataPath.lst>']).read().splitlines():
        videoID, wave_file, video_avi_file, video_mpeg_file, trs_file, xgtf_file, idx_file = path.split(' ')
        idxPath[videoID] = args['<corpusPath>']+'/'+idx_file

    X = []
    Y = []
    for videoID in open(args['<video_list>']).read().splitlines():
        print videoID

        frame2time = IDXHack(idxPath[videoID])

        facetracks = {}
        for line in open(args['<facetrack_pos>']+'/'+videoID+'.facetrack').read().splitlines():
            frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
            facetracks.setdefault(frameID, {})
            facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h
        ref_f = read_ref_facetrack_position(args['<reference_head_path>']+'/'+videoID+'.position', tempo_margin)
        
        ref_spk = []
        for line in open(args['<reference_speaker_path>']+'/'+videoID+'.atseg').read().splitlines():
            v, startTime, endTime, spkName = line.split(' ') 
            ref_spk.append([float(startTime), float(endTime), spkName])
        ref_spk.sort()

        facetrack_vs_ref = align_facetrack_ref(ref_f, facetracks)

        for line in open(args['<descriptor_path>']+'/'+videoID+'.desc').read().splitlines():
            l = line.split(' ')
            frameID = int(l[0])
            timestamp = frame2time(frameID, 0.0)
            faceID = int(l[1])
            desc = map(float, l[2:])

            if faceID in facetrack_vs_ref:

                for startTime, endTime, spkName in ref_spk:
                    if startTime <= timestamp and timestamp <= endTime:
                        break
                SpeakingFace = False
                if faceID in d_align:
                    if d_align[faceID] == spkName:
                        SpeakingFace = True

                SpeakingFace = 0
                if facetrack_vs_ref[faceID] == st_vs_ref[stID]:
                    SpeakingFace = 1

                X.append(desc)
                Y.append(SpeakingFace)

    clf = CalibratedClassifierCV(LogisticRegression(), method='isotonic')
    # train model
    clf.fit(X, Y) 
    # save model
    joblib.dump(clf, args['<output_model_file>']) 
