"""
Face detection based on Viola & Jones method

Usage:
  face_detection.py <video> <output_path> <cascade_file> [--scaleFactor=<sf>] [--minNeighbors=<mn>] [--minSize=<min>] [--startframe=<sf>] [--endframe=<ef>] [--detectionEveryXFrame=<sk> | --pathFileListFrame=<pflf>]
  face_detection.py -h | --help
Options:
  --scaleFactor=<sf>            scaleFactor value [default: 1.1]
  --minNeighbors=<mn>           minNeighbors value [default: 4]
  --minSize=<min>               min size of a face [default: 50]
  --startframe=<ef>             first frame to process [default: 0]
  --endframe=<ef>               last frame to process default last frame
  --detectionEveryXFrame=<sk>   nb frame to skip after each face detection [default: 1]
  --pathFileListFrame=<pflf>    path that contain the file with the list of frame to process

"""

import numpy as np
import cv2, cv, sys, os
import json
from docopt import docopt

import tools

if __name__ == '__main__':
    arguments = docopt(__doc__)
    video = arguments['<video>']
    if not os.path.isfile(video):
        print "error: can't find the video:", video
        sys.exit()
    else:
        print 'video:', video

    path_out = arguments['<output_path>']
    if not os.path.isdir(path_out):
        print "error: can't find folder:", path_out
        sys.exit() 
    else:
        print 'path_out:', path_out

    cascade_file = arguments['<cascade_file>']
    if not os.path.isfile(cascade_file):
        print "error: can't find the cascade:", cascade_file
        sys.exit()
    else:
        print 'cascade_file:', cascade_file

    scaleFactor = arguments['--scaleFactor']
    scaleFactor = float(scaleFactor)
    if scaleFactor <= 1.0 :
        print 'error: scaleFactor must be higher than 1.0'
        sys.exit()
    else:
        print 'scaleFactor:', scaleFactor

    minNeighbors = arguments['--minNeighbors']
    minNeighbors = int(minNeighbors)
    if minNeighbors < 1 :
        print 'error: minNeighbors must be higher than 1'
        sys.exit()  
    else:
        print 'minNeighbors:', minNeighbors

    minSize = arguments['--minSize']
    minSize = (int(minSize), int(minSize))
    if minSize < 1 :
        print 'error: minSize must be higher than 1'
        sys.exit()  
    else:
        print 'minSize: ', minSize

    pathFileListFrame = arguments['--pathFileListFrame']
    if pathFileListFrame:
        if not os.path.isfile(pathFileListFrame+'/'+video.split('/')[-1]+'.frame'):
            print "error: can't find the FileListFrame:", pathFileListFrame+'/'+video.split('/')[-1]+'.frame'
            sys.exit()  
        else:
            print 'FileListFrame:', pathFileListFrame+'/'+video.split('/')[-1]+'.frame'
    l_frame = []
    if pathFileListFrame:
        for line in open(pathFileListFrame+'/'+video.split('/')[-1]+'.frame'):
            l_frame.append(int(line[:-1]))
        print 'proceed only', len(l_frame), 'frames'

    detectionEveryXFrame = arguments['--detectionEveryXFrame']
    detectionEveryXFrame = int(detectionEveryXFrame)    
    if detectionEveryXFrame < 1:
        print 'error: detectionEveryXFrame must be equal or higher than 1'
        sys.exit()
    else:
        print 'proceed face detection every', detectionEveryXFrame, 'frames'


    capture = cv.CaptureFromFile(video)

    startframe = arguments['--startframe']
    endframe = arguments['--endframe']
    startframe = int(startframe)
    if endframe:
        endframe = int(endframe)
    else:
        endframe = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT)-1)

    if startframe > endframe :
        print 'error: startframe must be higher than endframe'
        sys.exit()
    if startframe < -1 or endframe < -1:
        print 'error: startframe and endframe must be higher than 0'
        sys.exit()

    print "proceed frame from", startframe, 'to', endframe


    storage = cv.CreateMemStorage()
    cascade = cv.Load(cascade_file)

    c_frame = 0    
    data_out = {}
    while True:
        image = cv.QueryFrame(capture)
        c_frame = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_POS_FRAMES))
        if c_frame >= startframe:
            process_frame = False
            if (l_frame == [] and c_frame >= startframe and c_frame%detectionEveryXFrame == 0):
                process_frame = True
            elif c_frame in l_frame: 
                process_frame = True

            if process_frame: 
                data_out[c_frame] = [] 
                detected = cv.HaarDetectObjects(image, cascade, storage, scaleFactor, minNeighbors, cv.CV_HAAR_DO_CANNY_PRUNING, minSize)
                for (x,y,w,h),n in detected:  
                    data_out[c_frame].append({'x':str(x), 'y':str(y), 'w':str(w), 'h':str(h), 'n':str(n)})

        if c_frame > endframe:
            break

    with open(path_out+'/'+video.split('/')[-1]+'.face.json', 'w') as f:
        json.dump(data_out, f, indent=4, sort_keys=True) 
