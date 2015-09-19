"""
create a video with bounding box on facetracks

Usage:
  face_detection_visualization.py <videoFile> <faceTracking>
  face_detection_visualization.py -h | --help
"""

import numpy as np
import cv2, cv
from docopt import docopt

if __name__ == '__main__':
    args = docopt(__doc__)

    faceTracking = {}
    for line in open(args['<faceTracking>']).read().splitlines():
        frameID, trackID, xmin, ymin, w, h = map(int, line.split(' ')) 
        faceTracking.setdefault(frameID, []).append([trackID, xmin, ymin, xmin+w, ymin+h])

    capture = cv2.VideoCapture(args['<videoFile>'])
    nb_frame = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT)-1)

    print "read video and save images"
    frameID = 0 
    while (frameID<nb_frame):
        ret, frame = capture.read()                             # read the video
        frameID = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES))     

        if ret:
            if frameID in faceTracking:
                for trackID, xmin, ymin, xmax, ymax in faceTracking[frameID]:
                    cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(255,0,0),2)
                    cv2.putText(frame, str(trackID) ,(xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),1,10)
        
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
