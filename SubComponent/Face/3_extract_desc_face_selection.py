"""
Extract descriptor to find speaking face

Usage:
  extract_desc_speaking_face.py <videoID> <rawFaceTrackPosition> <rawFaceTrackSegmentation> <descFaceSelection_out> [--videoWidth=<vw>] 
  extract_desc_speaking_face.py -h | --help
Options:
  --videoWidth=<of>   width of the video [default: 1024]
"""

from docopt import docopt
import numpy as np
import cv2, cv, math
from mediaeval_util.repere import IDXHack, MESegParser

lk_params = dict( winSize  = (20, 20), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict( maxCorners = 100, qualityLevel = 0.01, minDistance = 4, blockSize = 3)  

if __name__ == "__main__": 
    # read arguments   
    args = docopt(__doc__)

    video_w_center = float(args['--videoWidth'])/2
    face_seg, confs, timeToFrameID = MESegParser(args['<rawFaceTrackSegmentation>'], args['<videoID>'])
    pos = {}
    for line in open(args['<rawFaceTrackPosition>']):
        frameID, faceID, x, y, w, h = line[:-1].split(' ')
        pos.setdefault(faceID, {})
        pos[faceID][int(frameID)] = map(float, [x, y, w, h])
    fout = open(args['<descFaceSelection_out>'], 'w')
    for seg, trackID, faceID in face_seg.itertracks(label=True):
        fout.write(faceID+" "+str(confs[trackID])+" "+str(len(pos[faceID]))+" "+str(float(confs[trackID])/len(pos[faceID])))
        l_size, l_central, l_move = [], [], []
        for frameID in pos[faceID]: 
            x, y, w, h = pos[faceID][frameID]
            x_center = x+w/2
            l_size.append(w*h)
            l_central.append(abs(x_center-video_w_center))
            if frameID-1 in pos[faceID]:
                y_center = y+h/2
                x1, y1, w1, h1 = pos[faceID][frameID-1]
                x1_center = x1+w1/2
                y1_center = y1+h1/2
                l_move.append(abs(x_center-x1_center)+abs(y_center-y1_center))
        if l_move == []:
            l_move = [0.0]
        fout.write(" "+str(np.mean(l_size))+" "+str(np.mean(l_central))+" "+str(np.mean(l_move))+'\n')
    fout.close()
