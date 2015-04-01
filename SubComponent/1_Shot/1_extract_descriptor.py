"""
Compute histogram and optical flow between 2 consecutive frames to find shot boundaries

Usage:
  cut_hist_OF_score.py <video_file> <output_file> [--x=<x> --y=<y> --w=<w> --h=<h>]
  cut_hist_OF_score.py -h | --help
Options:
  --x=<x>    position left of the ROI (0 > x > video_width) [default: 0]
  --y=<y>    position bottom of the ROI (0 > y > video_height) [default: 0]
  --w=<w>    position width of the ROI, (0 > w+x > video_width), default width of the video
  --h=<h>    position height of the ROI, (0 > h+y > video_height), default height of the video
"""

import cv2, cv
import json, sys
from docopt import docopt

def calcul_hist(src_np):  
    height, width, depth = src_np.shape
    src = cv.CreateImageHeader((width, height), cv.IPL_DEPTH_8U, 3)
    cv.SetData(src, src_np.tostring(), src_np.dtype.itemsize * 3 * width)
    hsv = cv.CreateImage((width, height), 8, 3)
    cv.CvtColor(src, hsv, cv.CV_BGR2HSV)
    h_plane = cv.CreateMat(height, width, cv.CV_8UC1)
    s_plane = cv.CreateMat(height, width, cv.CV_8UC1)
    cv.Split(hsv, h_plane, s_plane, None, None)
    hist = cv.CreateHist([30, 32], cv.CV_HIST_ARRAY, [[0, 180], [0, 255]], 1)    
    cv.CalcHist([cv.GetImage(i) for i in [h_plane, s_plane]], hist)
    return hist  
    
def score_OF(roi, frame1, frame0, lk_params, feature_params):
    # extract images
    img0 = cv2.cvtColor(frame0[roi.ymin:roi.ymax, roi.xmin:roi.xmax], cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(frame1[roi.ymin:roi.ymax, roi.xmin:roi.xmax], cv2.COLOR_BGR2GRAY)

    # find point on the 2 images
    p0 = cv2.goodFeaturesToTrack(img0, mask = None, **feature_params)    
    p1 = cv2.goodFeaturesToTrack(img1, mask = None, **feature_params)    
    if p0 is None or  p1 is None:
        return -1.0
    
    # find point on the other images
    p1_0, st1_0, err1_0 = cv2.calcOpticalFlowPyrLK(img0, img1, p0, p1, **lk_params)
    p0_1, st0_1, err0_1 = cv2.calcOpticalFlowPyrLK(img1, img0, p1, p0, **lk_params)
    
    # count the number of points find
    nb_c_p0=0.0
    nb_p0=0.0
    nb_c_p1=0.0
    nb_p1=0.0
    for pts0, pts1, s in itertools.izip(p0, p1_0, st1_0):
        nb_p0+=1
        if s[0] == 1:
            nb_c_p0+=1
    for pts0, pts1, s in itertools.izip(p1, p0_1, st0_1):
        nb_p1+=1
        if s[0] == 1:
            nb_c_p1+=1

    return (nb_c_p0/nb_p0+nb_c_p1/nb_p1)/2


if __name__ == '__main__': 
    # read args
    args = docopt(__doc__)
    x = int(args['--x'])
    y = int(args['--y'])
    w = int(args['--w'])
    h = int(args['--h'])

    lk_params = dict(winSize  = (20, 20), 
                     maxLevel = 2, 
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners = 100, 
                          qualityLevel = 0.01,
                          minDistance = 10,
                          blockSize = 3) 
    # open video
    capture = cv2.VideoCapture(args['<video_file>'])
    # extraction video information
    nb_frame = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT)-1)
    video_width = int(capture.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    # read the first frame
    ret, frame = capture.read()
    c_frame = capture.get(cv.CV_CAP_PROP_POS_FRAMES)    
    # get only the ROI
    frame = frame[y:y+h, x:x+w]
    # copy the current frame
    frame_previous = frame.copy()

    fout = open(args['<output_file>'], 'w')
    while (c_frame<nb_frame):
        ret, frame = capture.read()
        frame = frame[y:y+h, x:x+w]
        c_frame = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES))  
        if frame.any():
            # compute and save descriptor 
            histo_current = calcul_hist(frame)    
            previous_current =  calcul_hist(frame_previous) 
            fout.write(c_frame)
            fout.write(' '+str(round(cv.CompareHist(histo_current, previous_current, cv.CV_COMP_CORREL), 3)))
            fout.write(' '+str(round(score_OF(tools.ROI(0, 0, w, h), frame, frame_previous, lk_params, feature_params), 3)))
            fout.write('\n')
            # copy the current frame
            frame_previous = frame.copy()
    fout.close()