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
from mediaeval_util.imageTools import calcul_hist, prop_pts_find_by_optical_flow

def define_roi(x, y, w, h, capture):
    # define ROI position
    x = x and int(x) or 0.0
    y = y and int(y) or 0.0
    w = w and int(w) or int(capture.get(cv.CV_CAP_PROP_FRAME_WIDTH))-x
    h = h and int(h) or int(capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT))-y
    return x, y, w, h

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

    # find_ROI position
    x, y, w, h = define_roi(args['--x'], args['--y'], args['--w'], args['--h'], capture)

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
            fout.write(' %09.3f' %(frame2time(c_frame, capture.get(cv.CV_CAP_PROP_POS_MSEC)/1000.0)))
            fout.write(' %5.3f' %(cv.CompareHist(calcul_hist(frame), calcul_hist(frame_previous), cv.CV_COMP_CORREL)))
            fout.write(' %5.3f' %(prop_pts_find_by_optical_flow(frame_previous, frame, lk_params, feature_params)))
            fout.write('\n')
            # copy the current frame
            frame_previous = frame.copy()
    fout.close()
