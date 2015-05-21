"""
Face detection based on Viola & Jones method

Usage:
  face_detection.py <videoID> <videoFile> <faceDetection> <haarcascade> [--shotSegmentation=<ss>] [--scaleFactor=<sf>] [--minNeighbors=<mn>] [--minSize=<min>]
  face_detection.py -h | --help
Options:
  --shotSegmentation=<ss>   shot to process
  --scaleFactor=<sf>        scaleFactor value (>1.0) [default: 1.1]
  --minNeighbors=<mn>       minNeighbors value (>1) [default: 4]
  --minSize=<min>           min size of a face in proportion of the video height [default: 0.0868]
"""

from docopt import docopt
import cv2, cv
import numpy as np

if __name__ == '__main__':
    # read args
    args = docopt(__doc__)
    videoID = args['<videoID>']

    capture = cv.CaptureFromFile(args['<videoFile>'])
    minSize = int(round(float(args['--minSize']) * cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT),0))
    minSize = (minSize, minSize)
    # storage for face detected  
    storage = cv.CreateMemStorage()
    # load cascade file
    cascade = cv.Load(args['<haarcascade>'])
    # read video
    # Read shot segmentation
    frames_to_process = []
    if args['--shotSegmentation']:        
        for line in open(args['--shotSegmentation']).read().splitlines():
            v, shot, startTime, endTime, startFrame, endFrame = line.split(' ') 
            if v == videoID:
                for c_frame in range(int(startFrame), int(endFrame)+1):
                    frames_to_process.append(c_frame)
    else:
        nb_frame = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT)-1)
        for c_frame in range(0, nb_frame):
            frames_to_process.append(c_frame)
    last_frame_to_process = max(frames_to_process)
    # save face detection
    c_frame = 0 
    fout = open(args['<faceDetection>'], 'w')
    while (c_frame<last_frame_to_process):
        frame = cv.QueryFrame(capture)
        c_frame = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_POS_FRAMES))
        if frame and c_frame in frames_to_process:
            detected = cv.HaarDetectObjects(frame, cascade, storage, float(args['--scaleFactor']), int(args['--minNeighbors']), cv.CV_HAAR_DO_CANNY_PRUNING, minSize)
            for (x,y,w,h),n in detected: fout.write(str(int(c_frame))+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+' '+str(n)+'\n')
    fout.close()
