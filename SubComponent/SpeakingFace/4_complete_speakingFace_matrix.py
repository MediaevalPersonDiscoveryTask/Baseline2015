"""
compute distance between speech turns and facetracks that do not co-occure

Usage:
  complete_speakingFace_matrix.py <videoID> <speechTurnSegmentation> <faceTrackSegmentation> <probaSpeechTurns> <probaFacetracks> <probaSpeakingFace> <output_matrix>
  complete_speakingFace_matrix.py -h | --help
"""

from docopt import docopt
from mediaeval_util.repere import IDXHack, MESegParser
import numpy as np
from scipy import spatial, cluster
import pickle


# compute transistive distance
def compute_distance_trans(TrackStID, TrackFaceID, svh, hvh, svs, l_st_cooccuring_face):
    trans_dist1 = 0.0
    for TrackFaceIDTrans in svh[TrackStID]:
        if svh[TrackStID][TrackFaceIDTrans] > 0.5 and hvh[TrackFaceID][TrackFaceIDTrans] > 0.0:
            dist_tmp = (svh[TrackStID][TrackFaceIDTrans] + hvh[TrackFaceID][TrackFaceIDTrans])/2 
            if dist_tmp  > trans_dist1 : trans_dist1 = dist_tmp
    trans_dist2 = 0.0
    for TrackStIDTrans in l_st_cooccuring_face[TrackFaceID]:
        if svh[TrackStIDTrans][TrackFaceID]>0.5 and svs[TrackStID][TrackStIDTrans]>0.0:   # to take into account only head2 that can correspond to st and only distance between head and head2 correspond to sommeting right
            dist_tmp = (svh[TrackStIDTrans][TrackFaceID] + svs[TrackStID][TrackStIDTrans])/2 
            if dist_tmp  > trans_dist2 : trans_dist2 = dist_tmp 
    trans_dist = trans_dist1+trans_dist2
    if trans_dist1 > 0.0 and trans_dist2 > 0.0 : trans_dist /= 2
    return trans_dist

if __name__ == '__main__':
    args = docopt(__doc__)

    svh, hvh, svs, l_st_cooccuring_face = {}, {}, {}, {}

    # read speech turns segmentation
    st_seg, confs, timeToFrameID = MESegParser(args['<speechTurnSegmentation>'], args['<videoID>'])
    for s, t, l in st_seg.itertracks(label=True):
        svs[t], svh[t] = {}, {}

    # read proba between speech turns
    for line in open(args['<probaSpeechTurns>']).read().splitlines():
        t1, t2, p = line.split(' ')        
        svs[int(t1)][int(t2)] = float(p)
        svs[int(t2)][int(t1)] = float(p)

    # read facetracks segmentation
    l_TrackFaceID = []
    face_seg, confs, timeToFrameID = MESegParser(args['<faceTrackSegmentation>'], args['<videoID>'])
    for s, t, l in face_seg.itertracks(label=True):
        l_TrackFaceID.append(t)
        hvh[t], l_st_cooccuring_face[t] = {}, []
    l_TrackFaceID.sort()

    # read proba between facetracks
    hvh_tmp = pickle.load(open(args['<probaFacetracks>'], "rb" ) )
    hvh_tmp = spatial.distance.squareform(hvh_tmp, checks=False)
    for i in range(len(l_TrackFaceID)):
        for j in range(len(l_TrackFaceID)):
            hvh[l_TrackFaceID[i]][l_TrackFaceID[j]] = hvh_tmp[i][j]

    # read proba speaking face
    for line in open(args['<probaSpeakingFace>']).read().splitlines():
        TrackStID, TrackFaceID, proba = line.split(' ')
        svh[int(TrackStID)][int(TrackFaceID)] = float(proba)
        l_st_cooccuring_face[int(TrackFaceID)].append(int(TrackStID))

    fout = open(args['<output_matrix>'], 'w')
    for segSt, TrackStID, StID in st_seg.itertracks(label=True):
        for segFace, TrackFaceID, FaceID in face_seg.itertracks(label=True):
            if TrackFaceID in svh[TrackStID]:
                fout.write(str(TrackStID)+' '+str(TrackFaceID)+' '+str(svh[TrackStID][TrackFaceID])+'\n')
            else:
                fout.write(str(TrackStID)+' '+str(TrackFaceID)+' '+str(compute_distance_trans(TrackStID, TrackFaceID, svh, hvh, svs, l_st_cooccuring_face))+'\n')

    fout.close()