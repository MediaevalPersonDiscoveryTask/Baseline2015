"""
Take a video file (<video_file>) in avi format and compute:
 - A score based on the difference of color histogram between 2 consecutive frames
 - A score based on the proportion of point of interested retrieved between 2 consecutive frame
Write this 2 scores into an output file <output_file>.
If x, y, w, h is defined, the score is computed only on a region of interest of the images.

Usage:
  cut_hist_OF_score.py <video_file> <output_file> [--x=<x> --y=<y> --w=<w> --h=<h>] [--idx=<idx>]
  cut_hist_OF_score.py -h | --help
Options:
  --x=<x>      position left of the ROI (0 > x > video_width) [default: 0]
  --y=<y>      position bottom of the ROI (0 > y > video_height) [default: 0]
  --w=<w>      position width of the ROI, (0 > w+x > video_width), default width of the video
  --h=<h>      position height of the ROI, (0 > h+y > video_height), default height of the video
  --idx=<idx>  mapping between frame number to timestamp
"""

from docopt import docopt
import cv2, cv
import itertools 
import numpy as np
from mediaeval_util.repere import IDXHack

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
    
def score_OF(img0, img1, lk_params, feature_params):
    # convert image to gray
    img0_tmp = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1_tmp = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # find point on the 2 images
    p0 = cv2.goodFeaturesToTrack(img0_tmp, **feature_params)    
    p1 = cv2.goodFeaturesToTrack(img1_tmp, **feature_params)    
    if p0 is None and p1 is None:
        return 1.0

    # find point on the other images    
    st1_0, st0_1 = [], []
    if p0 is not None:
        p0_1, st0_1, err0_1 = cv2.calcOpticalFlowPyrLK(img0_tmp, img1_tmp, p0, None, **lk_params)
        st0_1 = list(np.array(st0_1).T[0])
    if p1 is not None:
        p1_0, st1_0, err1_0 = cv2.calcOpticalFlowPyrLK(img1_tmp, img0_tmp, p1, None, **lk_params)
        st1_0 = list(np.array(st1_0).T[0])

    # return proportion of point retrieved
    return np.mean(st0_1 + st1_0)


if __name__ == '__main__': 
    # read args
    args = docopt(__doc__)

    # parameters for optical flow
    lk_params = dict(winSize  = (20, 20), 
                     maxLevel = 2, 
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners = 100, 
                          qualityLevel = 0.01,
                          minDistance = 10,
                          blockSize = 3) 

    # open video
    capture = cv2.VideoCapture(args['<video_file>'])

    # define ROI position
    if args['--w']:
        w = int(args['--w'])
    else:
        w = int(capture.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    if args['--h']:
        h = int(args['--h'])
    else:
        h = int(capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    x = int(args['--x'])
    y = int(args['--y'])

    # read the first frame, take only the ROI and copy as the previous frame
    ret, frame = capture.read()
    frame_previous = frame[y:y+h, x:x+w].copy()
    nb_frame = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT))
    c_frame = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES))

    # defined function from frame to timestamp
    frame2time = IDXHack(args['--idx'])

    # save desc into a file
    fout = open(args['<output_file>'], 'w')
    while (c_frame<nb_frame):
        ret, frame = capture.read()
        c_frame = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES))
        if ret:
            frame = frame[y:y+h, x:x+w]
            # compute and save descriptor 
            fout.write('%06d' %(c_frame))
            fout.write(' '+str(round(cv.CompareHist(calcul_hist(frame), calcul_hist(frame_previous), cv.CV_COMP_CORREL), 3)))
            fout.write(' '+str(round(score_OF(frame_previous, frame, lk_params, feature_params), 3)))
            fout.write(' %09.3f' %(frame2time(c_frame, capture.get(cv.CV_CAP_PROP_POS_MSEC)/1000.0)))
            fout.write('\n')
            # copy the current frame
            frame_previous = frame.copy()
    fout.close()
