"""
Evaluation of the speaking face detection

Usage:
  proba_mat_speechTurn.py <video_list> <matrix_speaking_face> <facetrack> <facetrackPosition> <reference_head_position> <st_seg> <speaker_ref> <idx_path> <shotSegmentation>
  proba_mat_speechTurn.py -h | --help
"""

from docopt import docopt
from pyannote.algorithms.tagging import ArgMaxDirectTagger
from pyannote.parser import MDTMParser
from mediaeval_util.repere import IDXHack, MESegParser, align_st_ref, read_ref_facetrack_position, align_facetrack_ref
from sklearn.externals import joblib
import numpy as np
import copy

if __name__ == '__main__':
    args = docopt(__doc__)

    nb_ref_speakingFace = 0.0
    nb_hyp_speakingFace = 0.0
    correct_speakingFace = 0.0

    thr_proba = 0.6

    for videoID in open(args['<video_list>']).read().splitlines():
        print videoID
        frames_to_process = []
        for line in open(args['<shotSegmentation>']).read().splitlines():
            v, shot, startTime, endTime, startFrame, endFrame = line.split(' ') 
            if v==videoID:
                for frameID in range(int(startFrame), int(endFrame)+1):
                    frames_to_process.append(frameID)
        
        ref_f_tmp = read_ref_facetrack_position(args['<reference_head_position>'], videoID, 0)
        ref_f = copy.deepcopy(ref_f_tmp)
        for frameID in ref_f_tmp:
            if frameID not in frames_to_process:
                del ref_f[frameID]

        facetracks = {}
        l_ft = []
        for line in open(args['<facetrackPosition>']+'/'+videoID+'.facetrack').read().splitlines():
            frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
            facetracks.setdefault(frameID, {})
            facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h
            if frameID in ref_f:
                l_ft.append(faceID)

        facetrack_vs_ref = align_facetrack_ref(ref_f, facetracks)

        facetrack, confs, timeToFrameID = MESegParser(args['<facetrack>']+videoID+'.MESeg', videoID)
        faceID_to_trackID_face = {}
        trackID_face_to_faceID = {}
        for s, trackID, faceID in facetrack.itertracks(label=True):
            faceID_to_trackID_face[faceID] = trackID
            trackID_face_to_faceID[trackID] = faceID


        st_to_ref = align_st_ref(args['<st_seg>']+'/'+videoID+'.MESeg', args['<speaker_ref>'], videoID)

        st_seg, confs, timeToFrameID = MESegParser(args['<st_seg>']+videoID+'.MESeg', videoID)
        st_to_trackID_st = {}
        trackID_st_to_st = {}
        for s, trackID, st in st_seg.itertracks(label=True):
            st_to_trackID_st[st] = trackID
            trackID_st_to_st[trackID] = st

        speaking_frame = {}
        for line in open(args['<matrix_speaking_face>']+videoID+'.mat').read().splitlines():
            trackID_st, trackID_face, proba = line.split(' ')
            if int(trackID_face_to_faceID[int(trackID_face)]) in l_ft:
                proba = float(proba)
                speaking_frame.setdefault(trackID_st_to_st[int(trackID_st)], [proba, trackID_face_to_faceID[int(trackID_face)]])
                if proba > speaking_frame[trackID_st_to_st[int(trackID_st)]][0]:
                    speaking_frame[trackID_st_to_st[int(trackID_st)]] = [proba, trackID_face_to_faceID[int(trackID_face)]]                

        ref_spk = []
        for line in open(args['<speaker_ref>']).read().splitlines():
            v, startTime, endTime, startFrame, endFrame, t, l, conf = line.split(' ')
            if v == videoID:
                ref_spk.append([float(startTime), float(endTime), l])

        frame2time = IDXHack(args['<idx_path>']+videoID+'.MPG.idx')


        for frameID in ref_f:
            timestamp =  frame2time(frameID, 0.0)

            l_speaking_face = []
            for startTime, endTime, spkName in ref_spk:
                if timestamp >= startTime and timestamp <= endTime:
                    for headName in ref_f[frameID]:
                        if headName == spkName:
                            nb_ref_speakingFace+=1
                            l_speaking_face.append(spkName)

            for s, trackID, st in st_seg.itertracks(label=True):
                if timestamp >= s.start and timestamp <= s.end:
                    if st in speaking_frame and speaking_frame[st][0] >= thr_proba :
                        nb_hyp_speakingFace+=1
                        faceIDSpeaking = int(speaking_frame[st][1])
                        if faceIDSpeaking in facetrack_vs_ref :
                            if facetrack_vs_ref[faceIDSpeaking] in l_speaking_face:
                                correct_speakingFace+=1
                            
            #print

    print nb_ref_speakingFace,
    print nb_hyp_speakingFace,
    print correct_speakingFace,
    print 'precision:', round(correct_speakingFace/nb_hyp_speakingFace,3)*100, '%    ',
    print 'recall:',    round(correct_speakingFace/nb_ref_speakingFace,3)*100, '%    '





