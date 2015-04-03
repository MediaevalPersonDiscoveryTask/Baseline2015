"""
Face tracking based on the face detection and optical flow for miss detection

Usage:
  face_tracker.py <video> <output_path> <cut_path> <detected_face_path> [--threshold_OF=<of>] [--threshold_cov=<tc>]
  face_tracker.py -h | --help
Options:
  --threshold_OF=<of>       value of the threshold on the optical flow for the tracking [default: 0.3]
  --threshold_cov=<tc>      if the coverage of 2 boxes is higher than threshold_cov, we consider they correspond to the same face track [default: 0.2]
"""


import numpy as np
import math
import cv2, cv
import json, glob, os
from docopt import docopt

import tools

lk_params = dict( winSize  = (20, 20), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 100, 
                       qualityLevel = 0.01,
                       minDistance = 10,
                       blockSize = 3)

def find_new_face(detected_face, face_to_cluster, nb_face_cluster, l_frames_shot, backward, threshold_OF, threshold_cov):
    step = 1
    if backward:                                                # if the tracking is proceed backward
        step = -1

    new_face_find_by_tracking = {}                              # face that are find by the tracking
    for f_nb in l_frames_shot:
        new_face_find_by_tracking[f_nb] = {}

    for f_nb in sorted(l_frames_shot, reverse=backward):
        for i_face, face in detected_face[f_nb].items(): # for face find by the detection
            roi_current = face['roi']
            l_temp = {}
            nbFrameTracking = nb_face_cluster[face_to_cluster[i_face]] + face['sqrt_Neighbors']
            for i in range(1, nbFrameTracking):
                if f_nb+i*step in l_frames_shot and f_nb+i*step-step in l_frames_shot:                # check if the next frame is in the shot     
                    s, nb_p0, nb_p1, move_x, move_y, l_move_x, l_move_y = tools.score_OF(roi_current, l_frames_shot[f_nb+i*step-step], l_frames_shot[f_nb+i*step], lk_params, feature_params)
                    roi_current = tools.ROI(int(roi_current.xmin+move_x), int(roi_current.ymin+move_y), int(roi_current.xmax+move_x), int(roi_current.ymax+move_y)) 

                    if s < threshold_OF:                    
                        break
                    else:
                        best_coverage = 0.0
                        best_i_face = ''
                        for i_face2, face_next in detected_face[f_nb+i*step].items():  
                            roi_next = face_next['roi']
                            coverage = tools.score_coverage_ROI(roi_current, roi_next)  # compute the coverage between the 2 roi
                            if coverage > best_coverage:            
                                best_coverage = coverage  
                                best_i_face = i_face2 

                        if best_coverage >= threshold_cov:                     # if the coverage between 2 boxes is higher than a threshold we considered that these 2 faces correspondant to the same person
                            face_to_cluster[best_i_face] = face_to_cluster[i_face] # rename the second face found at the same place by the name of the current face
                            nb_face_cluster[face_to_cluster[i_face]]+=1
                            for f_nb2, roi in sorted(l_temp.items()):   # add the box to the list of face tracked
                                new_face_find_by_tracking[f_nb2][i_face] = roi
                            break
                        l_temp[f_nb+i*step] = roi_current           # Add face  to the temporary list
                else:
                    nb_face_cluster[face_to_cluster[i_face]]+=1
                    for f_nb2, roi in sorted(l_temp.items()):   # add the box to the list of face tracked
                        new_face_find_by_tracking[f_nb2][i_face] = roi                    

    for f_nb in sorted(l_frames_shot):                          # copy the face find by tracking into detected_face
        for i_face, roi in new_face_find_by_tracking[f_nb].items(): 
            detected_face[f_nb][i_face] = {'roi':roi, 'sqrt_Neighbors':0, 'Neighbors':0}

    return detected_face, face_to_cluster, nb_face_cluster


if __name__ == '__main__':
    arguments = docopt(__doc__)
    video = arguments['<video>']
    path_out = arguments['<output_path>']
    cut_path = arguments['<cut_path>']
    detected_face_path = arguments['<detected_face_path>']
    threshold_OF = arguments['--threshold_OF']
    threshold_OF = float(threshold_OF)
    threshold_cov = arguments['--threshold_cov']
    threshold_cov = float(threshold_cov)

    l_cut = []                                                  # list of shot boundaries
    for line in open(cut_path+'/'+video.split('/')[-1]+'.cut'):
        l_cut.append(int(line[:-1]))
 
    capture = cv2.VideoCapture(video)                           # read the video
    nb_frame = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT)-1)   # total number of frame in the video
    print 'nb_frame', nb_frame

    detected_face = {}                                          # face detected
    for i in range(1, nb_frame+2):
        detected_face[i] = {}

    dic_face_detection = json.load(open(detected_face_path+'/'+video.split('/')[-1]+'.face.json'))
    face_to_cluster = {}                                       # name of the face cluster 
    nb_face_cluster = {}
    nb_face = 1

    for frame in sorted(map(float, dic_face_detection.keys())):
        for face in dic_face_detection[str(int(frame))]:
            detected_face[int(round(frame,0))][nb_face] = {'Neighbors':int(face['n']), 'sqrt_Neighbors':int(math.sqrt(int(face['n']))), 'roi':tools.ROI(int(face['x']), int(face['y']), int(face['w'])+int(face['x']), int(face['h'])+int(face['y']))}
            face_to_cluster[nb_face] = nb_face
            nb_face_cluster[nb_face] = 1
            nb_face+=1                                          # name of the face detected
    
    c_frame = 0                                                 # number of the current frame read in the video
    data_out = {}
    l_frames_shot = {}                                          # list of frame number in the current shot 
    while (c_frame<nb_frame):
        ret, frame = capture.read()                             # read the video
        c_frame = capture.get(cv.CV_CAP_PROP_POS_FRAMES)  
        if frame.any() :                                        # if there an image in frame
            if c_frame in l_cut:                                # if the frame is a shot boundaries, proceed the tracking
                print c_frame
                # tracking forward
                detected_face, face_to_cluster, nb_face_cluster = find_new_face(detected_face, face_to_cluster, nb_face_cluster, l_frames_shot, False, threshold_OF, threshold_cov)
                # tracking backward
                detected_face, face_to_cluster, nb_face_cluster = find_new_face(detected_face, face_to_cluster, nb_face_cluster, l_frames_shot, True, threshold_OF, threshold_cov)
                for f_nb in sorted(l_frames_shot):              # for frame in the current shot
                    data_out[f_nb] = {}
                    for i_face, face in detected_face[f_nb].items():
                        data_out[f_nb][face_to_cluster[i_face]] = {'Neighbors':face['Neighbors'], 'x':face['roi'].xmin, 'y':face['roi'].ymin, 'w':face['roi'].xmax-face['roi'].xmin, 'h':face['roi'].ymax-face['roi'].ymin}
                #l_frames_shot = {}                              # clear the list of frame for the next shot
                l_frames_shot.clear()
            l_frames_shot[c_frame] = frame.copy()               # copy the current frame

    capture.release()                                           # relaese the video

    facetrack = {}
    for c_frame in data_out:
        i_frame = int(round(float(c_frame),0))
        facetrack[i_frame] = {}
        for i_face, face in data_out[c_frame].items():
            facetrack[i_frame][int(i_face)] = face

    fout = open(path_out+'/'+video.split('/')[-1]+'.avi.facetrack', 'w')
    for c_frame in sorted(facetrack):
        fout.write(str(c_frame)+' '+str(len(facetrack[c_frame])))
        for i_face, face in sorted(facetrack[c_frame].items()):
            fout.write(' '+str(i_face)+' '+str(face['x'])+' '+str(face['y'])+' '+str(face['w'])+' '+str(face['h']))
        fout.write('\n')
    fout.close()

