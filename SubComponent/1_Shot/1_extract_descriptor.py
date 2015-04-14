"""
Take a video file (<video_file>) in avi format and compute:
 - A score based on the difference of color histogram between 2 consecutive frames
 - A score based on the proportion of point of interested retrieved between 2 consecutive frame
Write this 2 scores into an output file <output_file>.
If x, y, w, h is defined, the score is computed only on a region of interest of the images.

Usage:
  extract_descriptor.py <video_file> <output_file> [--x=<x> --y=<y> --width=<w> --height=<h>] [--idx=<idx>]
  extract_descriptor.py -h | --help
Options:
  --x=<x>         position of the ROI in percentage (0.0 > x > 1.0) [default: 0.05]
  --y=<y>         position of the ROI in percentage (0.0 > y > 1.0) [default: 0.05]
  --width=<w>     position of the ROI in percentage (0.0 > width+x > 1.0) [default: 0.9]
  --height=<h>    position of the ROI in percentage (0.0 > height+y > 1.0) [default: 0.65]
  --idx=<idx>     mapping between frame number to timestamp
"""

from docopt import docopt
import cv2, cv
import itertools 
import numpy as np
from mediaeval_util.repere import IDXHack
from mediaeval_util.imageTools import calcul_hist, prop_pts_find_by_optical_flow

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
    video_width = capture.get(cv.CV_CAP_PROP_FRAME_WIDTH)
    video_height = capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT)   
    xmin = int(video_width  * float(args['--x']))
    ymin = int(video_height * float(args['--y']))
    xmax = int(video_width  * (float(args['--x'])+float(args['--width'])))
    ymax = int(video_height * (float(args['--y'])+float(args['--height'])))

    # read the first frame, take only the ROI and copy as the previous frame
    ret, frame = capture.read()
    frame_previous = frame[ymin:ymax, xmin:xmax].copy()
    nb_frame = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT))

    # defined function from frame to timestamp
    frame2time = IDXHack(args['--idx'])

    # save desc into a file
    fout = open(args['<output_file>'], 'w')
    c_frame = 0
    while (c_frame<nb_frame):
        ret, frame = capture.read()
        c_frame = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES))
        if ret:
            frame = frame[ymin:ymax, xmin:xmax]
            # compute and save descriptor 
            fout.write('%06d' %(c_frame))
            fout.write(' %09.3f' %(frame2time(c_frame, capture.get(cv.CV_CAP_PROP_POS_MSEC)/1000.0)))
            fout.write(' %5.3f' %(cv.CompareHist(calcul_hist(frame), calcul_hist(frame_previous), cv.CV_COMP_CORREL)))
            fout.write(' %5.3f' %(prop_pts_find_by_optical_flow(frame_previous, frame, lk_params, feature_params)))
            fout.write('\n')
            # copy the current frame
            frame_previous = frame.copy()
    fout.close()
