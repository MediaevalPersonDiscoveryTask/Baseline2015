"""
Face tracking based on the face detection and optical flow for miss detection

Usage:
  face_tracker.py <video_file> <shot_seg_file> <detection_face_file> <output_face_tracking_pos> <output_facetracks_segmentation> [--thr_score_OF=<of>] [--thr_coverage=<tc>] [--nbFrameTracking=<nft>] [--idx=<idx>] 
  face_tracker.py -h | --help
Options:
  --thr_score_OF=<of>       value of the threshold on the optical flow for the tracking [default: 0.3]
  --thr_nb_pts_OF=<of>      minimum number of point of interest find by th optical flow [default: 8]
  --thr_coverage=<tc>      if the coverage of 2 boxes is higher than thr_coverage, we consider they correspond to the same face track [default: 0.2]
  --nbFrameTracking=<nft>   number of frame where with try to find the net detection [default: 15]
  --idx=<idx>               mapping between frame number to timestamp
"""

from docopt import docopt
import numpy as np
import math
import cv2, cv
from mediaeval_util.imageTools import OpticalFlow, score_coverage_ROI
from mediaeval_util.repere import IDXHack
import copy

lk_params = dict(winSize  = (20, 20), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners = 100, qualityLevel = 0.01, minDistance = 10, blockSize = 3)

def find_face_detected_at_the_good_place(roi_current, l_face, thr_coverage):
    faceID_next_find = False
    best_coverage = thr_coverage
    for faceID_next, roi_next in l_face.items():  
        coverage = score_coverage_ROI(roi_current, roi_next)  # compute the coverage between the 2 roi
        if coverage > best_coverage:            
            best_coverage = coverage  
            faceID_next_find = faceID_next 
    return faceID_next_find

def faceTracking(faces, faceID_to_facetrackID, l_frames_shot, backward, thr_score_OF, thr_coverage, thr_nb_pts_OF, nbFrameTracking):
    step = backward and -1 or 1
    faces_original = copy.deepcopy(faces)
    for frameID in sorted(l_frames_shot, reverse=backward):
        for faceID in faces_original[frameID]:                              # for face find by the 
            xmin, ymin, xmax, ymax = faces_original[frameID][faceID]
            l_temp = {}
            l_frame_tracking = list(set(range(frameID, frameID+nbFrameTracking*step+step, step)).intersection(l_frames_shot.keys()))
            l_frame_tracking.sort(reverse=backward)
            for frameIDTracking, frameIDTrackingNext in zip(l_frame_tracking, l_frame_tracking[1:]):
                score, l_move_x, l_move_y = OpticalFlow(l_frames_shot[frameIDTracking][ymin:ymax, xmin:xmax], l_frames_shot[frameIDTrackingNext][ymin:ymax, xmin:xmax], lk_params, feature_params)
                if score < thr_score_OF or len(l_move_x) < thr_nb_pts_OF: break # stop tracking due to wrong optical flow
                # update position of the face
                xmin += np.mean(l_move_x)
                xmax += np.mean(l_move_x)
                ymin += np.mean(l_move_y)
                ymax += np.mean(l_move_y)
                faceID_next_find = find_face_detected_at_the_good_place([xmin, ymin, xmax, ymax], faces[frameIDTrackingNext], thr_coverage)
                if faceID_next_find:                                        # stop tracking due to face detected at the same place
                    #print '   find face ', faceID_next_find
                    for frameID2 in l_temp:
                        faces[frameID2][faceID] = l_temp[frameID2]
                    faceID_to_facetrackID[faceID_next_find] = faceID_to_facetrackID[faceID]  # merge the name of the current faceId and the face detected at the same place
                    break
                l_temp[frameIDTrackingNext] = [xmin, ymin, xmax, ymax]      # Add face  to the temporary list

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    fout_pos = open(args['<output_face_tracking_pos>'], 'w')
    fout_seg = open(args['<output_facetracks_segmentation>'], 'w')
    # read file with the list of shot
    frames_to_process = []
    shot_boundaries = []
    faces = {}                                                  # face detected
    for line in open(args['<shot_seg_file>']).read().splitlines():
        videoId, shot, startTime, endTime, startFrame, endFrame = line.split(' ') 
        shot_boundaries.append(int(endFrame))
        for frameID in range(int(startFrame), int(endFrame)+1):
            frames_to_process.append(frameID)
            faces[frameID] = {}
    last_frame_to_process = max(frames_to_process)+10

    # defined function to convert frameID to timestamp
    frame2time = IDXHack(args['--idx'])
    # open the video
    capture = cv2.VideoCapture(args['<video_file>'])
    # read face detection
    faceID_to_facetrackID = {}                                       # name of the face cluster 
    faceID = 1
    for line in open(args['<detection_face_file>']).read().splitlines():
        frameID, x, y, w, h, n = map(int, line.split(' '))
        if frameID in faces:
            faces[frameID][faceID] = [x, y, x+w, y+h]
            faceID_to_facetrackID[faceID] = faceID
            faceID+=1  
    

    frameID = 0                                                 # number of the current frame read in the video
    frames = {}
    frame_to_timestamp = {}                                     # list of frame number in the current shot 
    while (frameID<last_frame_to_process):
        ret, frame = capture.read()                             # read the video
        frameID = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES))
        if ret and frameID in frames_to_process:
            frame_to_timestamp[frameID] = frame2time(frameID, capture.get(cv.CV_CAP_PROP_POS_MSEC)/1000.0)
            frames[frameID] = frame.copy()
            if frameID in shot_boundaries: 
                faceTracking(faces, faceID_to_facetrackID, frames, False, float(args['--thr_score_OF']), float(args['--thr_coverage']), int(args['--thr_nb_pts_OF']), int(args['--nbFrameTracking']))
                #faceTracking(faces, faceID_to_facetrackID, frames, True,  float(args['--thr_score_OF']), float(args['--thr_coverage']), int(args['--thr_nb_pts_OF']), int(args['--nbFrameTracking']))
                # write face position
                seg_face = {}
                for frameID in sorted(frames):                          
                    for faceID in sorted(faces[frameID]):
                        xmin, ymin, xmax, ymax = faces[frameID][faceID]
                        seg_face.setdefault(faceID, []).append(frameID)
                        fout_pos.write(str(frameID)+' '+str(faceID_to_facetrackID[faceID])+' '+str(int(round(xmin, 0)))+' '+str(int(round(ymin, 0)))+' '+str(int(round(xmax-xmin, 0)))+' '+str(int(round(ymax-ymin, 0)))+'\n')
            frames.clear()
    fout_seg.close()
    
    # write face segmentation
    for faceID in sorted(seg_face):                                
        startFrame = min(seg_face[faceID])
        endFrame = max(seg_face[faceID])
        startTime = frame_to_timestamp[startFrame]
        endTime = frame_to_timestamp[endFrame]
        fout_seg.write(str(faceID_to_facetrackID[faceID])+' '+str(startTime)+' '+str(endTime)+' '+str(startFrame)+' '+str(endFrame)+'\n')
    fout_pos.close()

    capture.release()                                           # release the video
