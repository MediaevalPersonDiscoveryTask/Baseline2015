"""
Learn a classifier model to compute the probability that a facetrack is speaking

Usage:
  learn_model_proba_speaking_face.py <video_list> <dataPath.lst> <source_path> <facetrack_pos> <st_seg> <descriptor_path> <reference_head_path> <reference_speaker_path> <output_model_file> <video_test>
  learn_model_proba_speaking_face.py -h | --help
"""

from docopt import docopt
import numpy as np
from mediaeval_util.repere import IDXHack, read_ref_facetrack_position, align_facetrack_ref, align_st_ref
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron
from sklearn.externals import joblib
import math

if __name__ == '__main__':
    args = docopt(__doc__)

    tempo_margin = 3

    idxPath = {}
    for path in open(args['<dataPath.lst>']).read().splitlines():
        videoID, wave_file, video_avi_file, video_mpeg_file, trs_file, xgtf_file, idx_file = path.split(' ')
        idxPath[videoID] = args['<source_path>']+'/'+idx_file

    X = []
    Y = []
    for videoID in open(args['<video_list>']).read().splitlines():
        print videoID

        facetracks = {}
        for line in open(args['<facetrack_pos>']+'/'+videoID+'.facetrack').read().splitlines():
            frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
            facetracks.setdefault(frameID, {})
            facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h
        ref_f = read_ref_facetrack_position(args['<reference_head_path>']+'/'+videoID+'.position', tempo_margin)
        st_seg = []
        for line in open(args['<st_seg>']+'/'+videoID+'.mdtm').read().splitlines():
            v, p, start, dur, spk, na, na, st = line.split(' ')
            st_seg.append([float(start), float(start)+float(dur), st])
        ref_spk = []
        for line in open(args['<reference_speaker_path>']+'/'+videoID+'.atseg').read().splitlines():
            v, startTime, endTime, spkName = line.split(' ') 
            ref_spk.append([float(startTime), float(endTime), spkName])
        ref_spk.sort()

        facetrack_vs_ref = align_facetrack_ref(ref_f, facetracks)
        st_vs_ref = align_st_ref(st_seg, ref_spk)

        for line in open(args['<descriptor_path>']+'/'+videoID+'.desc').read().splitlines():
            l = line.split(' ')
            stID = l[0]
            faceID = int(l[1])
            desc = map(float, l[2:])
            for d in desc:
                if math.isnan(d):
                    print desc

            if faceID in facetrack_vs_ref and stID in st_vs_ref:
                SpeakingFace = 0
                if facetrack_vs_ref[faceID] == st_vs_ref[stID]:
                    SpeakingFace = 1
                X.append(desc)
                Y.append(SpeakingFace)

    clf = CalibratedClassifierCV(Perceptron(), method='isotonic')
    # train model
    clf.fit(X, Y) 
    # save model
    joblib.dump(clf, args['<output_model_file>']) 


    '''
    clf = joblib.load(args['<output_model_file>']) 


    l_true = []
    l_false = []

    for videoID in open(args['<video_test>']).read().splitlines():
        print videoID
        frame2time = IDXHack(idxPath[videoID])
        facetracks = {}
        for line in open(args['<facetrack_pos>']+'/'+videoID+'.facetrack').read().splitlines():
            frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
            facetracks.setdefault(frameID, {})
            facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h

        ref_f = read_ref_facetrack_position(args['<reference_head_path>']+'/'+videoID+'.position', tempo_margin)

        d_align = align_facetrack_ref(ref_f, facetracks)

        ref_spk = []
        for line in open(args['<reference_speaker_path>']+'/'+videoID+'.atseg').read().splitlines():
            v, startTime, endTime, spkName = line.split(' ') 
            ref_spk.append([float(startTime), float(endTime), spkName])
        ref_spk.sort()

        for line in open(args['<descriptor_path>']+'/'+videoID+'.desc').read().splitlines():
            l = line.split(' ')
            faceID = int(l[0])
            timestamp = frame2time(int(l[1]), 0.0)
            desc = map(float, l[2:])
            desc_ok = True
            for d in desc:
                if math.isnan(d):
                    desc_ok = False
                    break

            if desc_ok:
                for startTime, endTime, spkName in ref_spk:
                    if startTime <= timestamp and timestamp <= endTime:
                        break
                SpeakingFace = False
                if faceID in d_align:
                    if d_align[faceID] == spkName:
                        SpeakingFace = True

                score = clf.decision_function([desc])
                if SpeakingFace:
                    l_true.append(score)
                else:
                    l_false.append(score)


    l_range = list(range(-100000, 100000, 2000))

    hist_1 = np.histogram(l_true, l_range)
    hist_0 = np.histogram(l_false, l_range)

    for i in range(len(l_range)-1):
        print str(l_range[i]).replace('.', ','),
        print str(round(float(hist_0[0][i])/float(len(l_false))*100,2)).replace('.', ','),
        print str(round(float(hist_1[0][i])/float(len(l_true))*100,2)).replace('.', ',')
    '''


