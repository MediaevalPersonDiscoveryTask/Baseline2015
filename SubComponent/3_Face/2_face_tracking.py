"""
Face tracking based on the face detection and optical flow for miss detection

Usage:
  face_tracker.py <video_file> <shot_seg_file> <detection_face_file> <output_file> [--threshold_OF=<of>] [--threshold_cov=<tc>]
  face_tracker.py -h | --help
Options:
  --threshold_OF=<of>       value of the threshold on the optical flow for the tracking [default: 0.3]
  --threshold_cov=<tc>      if the coverage of 2 boxes is higher than threshold_cov, we consider they correspond to the same face track [default: 0.2]
"""

from docopt import docopt
import cv2, cv
import numpy as np
import math
from mediaeval_util.imageTools import OpticalFlow, score_coverage_ROI

lk_params = dict( winSize  = (20, 20), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 100, 
                       qualityLevel = 0.01,
                       minDistance = 10,
                       blockSize = 3)

def find_new_face(l_faces, face_to_cluster, nb_face_cluster, l_frames_shot, backward, threshold_OF, threshold_cov):
    step = 1
    if backward:                                                # if the tracking is proceed backward
        step = -1

    new_face_find_by_tracking = {}                              # face that are find by the tracking
    for f_nb in l_frames_shot:
        new_face_find_by_tracking[f_nb] = {}

    for f_nb in sorted(l_frames_shot, reverse=backward):
        for i_face, face in l_faces[f_nb].items(): # for face find by the detection
            xmin, ymin, xmax, ymax = face['x'], face['y'], face['x']+face['w'], face['y']+face['h']
            l_temp = {}
            nbFrameTracking = nb_face_cluster[face_to_cluster[i_face]] + face['sqrt_Neighbors']
            for i in range(1, nbFrameTracking):
                if f_nb+i*step in l_frames_shot and f_nb+i*step-step in l_frames_shot:                # check if the next frame is in the shot     
                    s, l_move_x, l_move_y = OpticalFlow(l_frames_shot[f_nb+i*step-step][ymin:ymax, xmin:xmax], l_frames_shot[f_nb+i*step][ymin:ymax, xmin:xmax], lk_params, feature_params)
                    xmin = xmin+np.mean(l_move_x)
                    xmax = xmax+np.mean(l_move_x)
                    ymin = ymin+np.mean(l_move_y)
                    ymax = ymax+np.mean(l_move_y)

                    if s < threshold_OF:                    
                        break
                    else:
                        best_coverage = 0.0
                        best_i_face = ''
                        for i_face2, face_next in l_faces[f_nb+i*step].items():  
                            xmin_next, ymin_next, xmax_next, ymax_next = face_next['x'], face_next['y'], face_next['x']+face_next['w'], face_next['y']+face_next['h']
                            coverage = score_coverage_ROI([xmin, ymin, xmax, ymax], [xmin_next, ymin_next, xmax_next, ymax_next])  # compute the coverage between the 2 roi
                            if coverage > best_coverage:            
                                best_coverage = coverage  
                                best_i_face = i_face2 

                        if best_coverage >= threshold_cov:                     # if the coverage between 2 boxes is higher than a threshold we considered that these 2 faces correspondant to the same person
                            face_to_cluster[best_i_face] = face_to_cluster[i_face] # rename the second face found at the same place by the name of the current face
                            nb_face_cluster[face_to_cluster[i_face]]+=1
                            for f_nb2, roi in sorted(l_temp.items()):   # add the box to the list of face tracked
                                new_face_find_by_tracking[f_nb2][i_face] = roi
                            break
                        l_temp[f_nb+i*step] = {'x':xmin, 'y':ymin, 'w':xmax-xmin, 'h':ymax-ymin}          # Add face  to the temporary list
                else:
                    nb_face_cluster[face_to_cluster[i_face]]+=1
                    for f_nb2, roi in sorted(l_temp.items()):   # add the box to the list of face tracked
                        new_face_find_by_tracking[f_nb2][i_face] = roi                    

    for f_nb in sorted(l_frames_shot):                          # copy the face find by tracking into l_faces
        for i_face, roi in new_face_find_by_tracking[f_nb].items(): 
            l_faces[f_nb][i_face] = {'Neighbors':0, 'sqrt_Neighbors':0, 'x':roi['x'], 'y':roi['y'], 'w':roi['w'], 'h':roi['h']}
    return l_faces, face_to_cluster, nb_face_cluster


if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
    # read file with the list of shot
    shot_boundaries = []                                                  # list of shot boundaries
    startFirstShot = +np.inf
    endLastShot = 0
    for line in open(args['<shot_seg_file>']).read().splitlines():
        videoId, shot, startTime, endTime, startFrame, endFrame = line.split(' ') 
        shot_boundaries.append(int(endFrame))
        if int(startFrame)<startFirstShot:
            startFirstShot = int(startFrame)
    endLastShot = max(shot_boundaries)
    # open the video
    capture = cv2.VideoCapture(args['<video_file>'])            # read the video
    nb_frame = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT)-1)   # total number of frame in the video
    # initialize the dictionnary with the list of face detected
    l_faces = {}                                                # face detected
    for i in range(1, nb_frame+2):
        l_faces[i] = {}
    # read face detection
    face_to_cluster = {}                                        # name of the face cluster 
    nb_face_cluster = {}                                        # number of face of a face track
    nb_face = 0
    for line in open(args['<detection_face_file>']).read().splitlines():
        c_frame, x, y, w, h, n = line.split(' ')
        nb_face+=1
        face_to_cluster[nb_face] = nb_face
        nb_face_cluster[nb_face] = 1

        l_faces[int(c_frame)][nb_face] = {'Neighbors':int(n), 
                                          'sqrt_Neighbors':int(round(math.sqrt(int(n)),0)), 
                                          'x':int(x), 
                                          'y':int(y), 
                                          'w':int(w), 
                                          'h':int(h)}

    ret, frame = capture.read()
    c_frame = capture.get(cv.CV_CAP_PROP_POS_FRAMES)
    data_out = {}
    l_frames_shot = {}                                          # frames of the current shot 
    fout = open(args['<output_file>'], 'w')
    while (c_frame<endLastShot):
        ret, frame = capture.read()                             # get the next image
        c_frame = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES))  
        if ret and c_frame>=startFirstShot:                     # if there is an image in the frame
            if c_frame in shot_boundaries:                      # if the frame is a shot boundaries, proceed the tracking
                # tracking forward
                l_faces, face_to_cluster, nb_face_cluster = find_new_face(l_faces, face_to_cluster, nb_face_cluster, l_frames_shot, False, float(args['--threshold_OF']), float(args['--threshold_cov']))
                # tracking backward
                l_faces, face_to_cluster, nb_face_cluster = find_new_face(l_faces, face_to_cluster, nb_face_cluster, l_frames_shot, True, float(args['--threshold_OF']), float(args['--threshold_cov']))
                for f_nb in sorted(l_frames_shot):              # for frame in the current shot
                    for i_face, face in l_faces[f_nb].items():
                        fout.write(str(f_nb)+' '+str(face_to_cluster[i_face])+' '+str(int(round(face['x'], 0)))+' '+str(int(round(face['y'], 0)))+' '+str(int(round(face['w'],0)))+' '+str(int(round(face['h'], 0)))+'\n')
                l_frames_shot.clear()
            l_frames_shot[c_frame] = frame.copy()               # copy the current frame
    capture.release()                                           # relaese the video

