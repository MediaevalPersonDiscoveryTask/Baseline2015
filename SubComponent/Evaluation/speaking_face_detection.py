"""
Evaluation of the speaking face detection

Usage:
  proba_mat_speechTurn.py <video_list> <matrix_speaking_face> <face_tracking> <reference_head_position_path> <st_seg> <speaker_ref> <idx_path> <shotSegmentation>
  proba_mat_speechTurn.py -h | --help
"""

from docopt import docopt
from pyannote.algorithms.tagging import ArgMaxDirectTagger
from pyannote.parser import MDTMParser
from mediaeval_util.repere import align_st_ref, drange
from sklearn.externals import joblib
import numpy as np

if __name__ == '__main__':
    args = docopt(__doc__)

    nb_ref_speakingFace = 0.0
    nb_hyp_speakingFace = 0.0
    correct_speakingFace = 0.0

    for videoID in open(args['<video_list>']).read().splitlines():

        frames_to_process = []
        for line in open(args['<shotSegmentation>']+'/'+videoID+'.shot').read().splitlines():
            videoId, shot, startTime, endTime, startFrame, endFrame = line.split(' ') 
            for frameID in range(int(startFrame), int(endFrame)+1):
                frames_to_process.append(frameID)
                
        ref_f_tmp = read_ref_facetrack_position(args['<reference_head_position_path>']+'/'+videoID+'.position', 0)
        ref_f = copy.deepcopy(ref_f_tmp)
        for frameID in ref_f_tmp:
            if frameID not in frames_to_process:
                del ref_f[frameID]
        facetracks = {}
        for line in open(args['<face_tracking>']+'/'+videoID+'.facetrack').read().splitlines():
            frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
            facetracks.setdefault(frameID, {})
            facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h
        facetrack_vs_ref = align_facetrack_ref(ref_f, facetracks)


        st_vs_ref = align_st_ref(args['<st_seg>'], args['<speaker_ref>'], videoID)
        st_seg = []
        for line in open(args['<st_seg>']+'/'+videoID+'.mdtm')
            v, p, start, dur, spk, na, na, st : line[:-1].split(' ')
            st_seg.append([float(start), float(start)+float(dur), st])

        ref_spk = []
        for line in open(args['<speaker_ref>']+videoID+'.atseg').read().splitlines():
            v, startTime, endTime, spkName = line.split(' ') 
            ref_spk.append([float(startTime), float(endTime), spkName])

        frame2time = IDXHack(args['<idx_path>']+videoID+'.MPG.idx')

        speaking_frame = {}
        for line in open(args['<matrix>']+videoID+'.mat').read().splitlines():
            st, faceID, proba = line.split(' ')
            faceID = int(faceID)
            proba = float(proba)
            speaking_frame.setdefault(st, [proba, faceID])
            if proba > speaking_frame[st][0]:
                speaking_frame[st] = [proba, faceID]

        for frameID in ref_f:
            timestamp =  frame2time(frameID, 0.0)

            l_speaking_face = []
            for startTime, endTime, spkName in ref_spk:
                if timestamp >= startTime and timestamp <= endTime:
                    for headName in ref_f[frameID]:
                        if headName == spkName:
                            nb_ref_speakingFace+=1
                            l_speaking_face.append(spkName)

            for startTime, endTime, st in st_seg:
                if timestamp >= startTime and timestamp <= endTime:
                    if st in speaking_frame and speaking_frame[st][0] >= 0,5
                        faceIDSpeaking = speaking_frame[st][1]
                        nb_hyp_speakingFace+=1
                        if faceIDSpeaking in facetrack_vs_ref and facetrack_vs_ref[faceIDSpeaking] in l_speaking_face:
                            correct_speakingFace+=1

    print 'precision:', round(correct_speakingFace/nb_hyp_speakingFace,3)*100, '%    ',
    print 'recall:',    round(correct_speakingFace/nb_ref_speakingFace,3)*100, '%    '







