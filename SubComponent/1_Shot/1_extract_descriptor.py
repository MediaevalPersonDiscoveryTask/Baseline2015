"""
Compute histogram and optical flow between 2 consecutive frames to find shot boundaries

Usage:
  cut_hist_OF_score.py <video_name> <video_file> <output_path> [--x=<x> --y=<y> --w=<w> --h=<h>]
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

import tools
    
if __name__ == '__main__': 
    arguments = docopt(__doc__)
    video_name = arguments['<video_name>']
    video_file = arguments['<video_file>']
    output_path = arguments['<output_path>']
    x = int(arguments['--x'])
    y = int(arguments['--y'])
    w = int(arguments['--w'])
    h = int(arguments['--h'])

    lk_params = dict( winSize  = (20, 20), 
                      maxLevel = 2, 
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict( maxCorners = 100, 
                           qualityLevel = 0.01,
                           minDistance = 10,
                           blockSize = 3) 

    capture = cv2.VideoCapture(video_file)
    nb_frame = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT)-1)
    video_width = int(capture.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    c_frame = capture.get(cv.CV_CAP_PROP_POS_FRAMES)    
    ret, frame = capture.read()
    frame = frame[y:y+h, x:x+w]
    frame_previous = frame.copy()

    data_out = {}
    while (c_frame<nb_frame):
        ret, frame = capture.read()
        frame = frame[y:y+h, x:x+w]
        c_frame = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES))  
        if frame.any() :
            histo_current = tools.calcul_hist(frame)    
            previous_current =  tools.calcul_hist(frame_previous) 
            val_hist = cv.CompareHist(histo_current, previous_current, cv.CV_COMP_CORREL)
            val_OF, nb_p0, nb_p1, move_x, move_y, l_move_x, l_move_y = tools.score_OF(tools.ROI(0, 0, w, h), frame, frame_previous, lk_params, feature_params)
            
            data_out[c_frame] = [round(val_hist, 3), round(val_OF, 3)]
            frame_previous = frame.copy()
    
    fout = open(output_path+'/'+video_name+'.cut_desc', 'w')
    for c_frame in sorted(data_out):
        v1, v2 = data_out[c_frame]
        fout.write(c_frame+' '+str(v1)+' '+str(v2)+'\n')
