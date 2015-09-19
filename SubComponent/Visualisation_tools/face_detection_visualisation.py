"""
create a video with bounding box on facetracks

Usage:
  face_detection_visualization.py <videoFile> <facePosition>
  face_detection_visualization.py -h | --help
"""

import numpy as np
import cv2, cv
from docopt import docopt

if __name__ == '__main__':
    args = docopt(__doc__)

    facePosition = {}
    for line in open(args['<facePosition>']).read().splitlines():
        frameID, xmin, ymin, w, h, _ = map(int, line.split(' ')) 
        facePosition.setdefault(frameID, []).append([xmin, ymin, xmin+w, ymin+h])

    capture = cv2.VideoCapture(args['<videoFile>'])
    nb_frame = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT)-1)

    print "read video and save images"
    frameID = 0 
    while (frameID<nb_frame):
        ret, frame = capture.read()                             # read the video
        frameID = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES))     

        if ret:
            if frameID in facePosition:
                for xmin, ymin, xmax, ymax in facePosition[frameID]:
                    cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(255,0,0),2)
        
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
