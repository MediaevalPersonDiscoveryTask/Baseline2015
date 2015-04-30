"""
Select speaking face base on probability matrix

Usage:
  select_speaking_face.py <video_train_list> <video_test_list> <matrix_path> <output_path> <st_seg> <reference_speaker> <facetrack_pos> <reference_head>
  select_speaking_face.py -h | --help
"""

from docopt import docopt
from mediaeval_util.repere import read_ref_facetrack_position, align_facetrack_ref, align_st_ref, drange
import numpy as np

if __name__ == '__main__':
    # read arguments       
    args = docopt(__doc__)

    l_false, l_true = [], []

    for videoID in open(args['<video_train_list>']).read().splitlines():

        st_vs_ref = align_st_ref(args['<st_seg>'], args['<reference_speaker>'], videoID)

        ref_f = read_ref_facetrack_position(args['<reference_head>']+videoID+'.position', 0)
        facetracks = {}
        l_facetrack_in_annotated_frame = set([])
        for line in open(args['<facetrack_pos>']+videoID+'.facetrack').read().splitlines():
            frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
            facetracks.setdefault(frameID, {})
            facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h
            if frameID in ref_f:
                l_facetrack_in_annotated_frame.add(faceID)            
        facetrack_vs_ref = align_facetrack_ref(ref_f, facetracks)

        for line in open(args['<matrix_path>']+videoID+'.mat').read().splitlines():
            st, faceID, proba = line.split(' ')
            faceID = int(faceID)
            if faceID in l_facetrack_in_annotated_frame:
                proba = float(proba)
                SpeakingFace = False
                if faceID in facetrack_vs_ref and st in st_vs_ref:
                    if facetrack_vs_ref[faceID] == st_vs_ref[st]:
                        SpeakingFace = True
                if SpeakingFace:
                    l_true.append(proba)
                else:
                    l_false.append(proba)

    l_range = list(drange(0.0, 1.0, 0.01))

    hist_1 = np.histogram(l_true, l_range)    
    hist_0 = np.histogram(l_false, l_range)

    ref = float(len(l_true))

    best_thr = 0.0
    best_F = 0.0
    for i in range(len(l_range)-1):
        correct = float(np.sum(hist_1[0][i:]))
        hyp = float(np.sum(hist_0[0][i:]) + np.sum(hist_1[0][i:]))
        P = hyp > 0 and correct/hyp or 0.0
        R = correct / ref
        F = P+R > 0 and (2*P*R)/(P+R) or 0.0
        if F > best_F:
            best_F = F
            best_thr = l_range[i]

    print best_thr, best_F

    nbFaceSelected = 0.0
    nbFaceTotal = 0.0

    for videoID in open(args['<video_test_list>']).read().splitlines():

        l = set([])
        lall = set([])
        
        for line in open(args['<matrix_path>']+videoID+'.mat').read().splitlines():
            st, faceID, proba = line.split(' ')
            faceID = int(faceID)
            proba = float(proba)
            if proba >= best_thr:
                l.add(faceID)
            lall.add(faceID)

        nbFaceSelected += len(l)
        nbFaceTotal += len(lall)

        #fout = open(args['<output_path>']+'/'+videoID+'.lfacetrack', 'w')


