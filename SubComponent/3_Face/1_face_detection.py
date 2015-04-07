"""
Face detection based on Viola & Jones method

Usage:
  face_detection.py <video_file> <output_file> <cascade_file> [--shot_segmentation=<ss>] [--scaleFactor=<sf>] [--minNeighbors=<mn>] [--minSize=<min>]
  face_detection.py -h | --help
Options:
  --shot_segmentation=<ss>  shot to process
  --scaleFactor=<sf>        scaleFactor value (>1.0) [default: 1.1]
  --minNeighbors=<mn>       minNeighbors value (>1) [default: 4]
  --minSize=<min>           min size of a face (>1) [default: 50]
"""

from docopt import docopt
import cv2, cv
import numpy as np

if __name__ == '__main__':
    # read args
    args = docopt(__doc__)
    minsize = (int(args['--minSize']), int(args['--minSize']))
    # storage for face detected  
    storage = cv.CreateMemStorage()
    # load cascade file
    cascade = cv.Load(args['<cascade_file>'])
    # read video
    capture = cv.CaptureFromFile(args['<video_file>'])
    # Read shot segmentation
    frames_to_process = []
    if args['--shot_segmentation']:        
        for line in open(args['--shot_segmentation']).read().splitlines():
            videoId, shot, startTime, endTime, startFrame, endFrame = line.split(' ') 
            for c_frame in range(int(startFrame), int(endFrame)+1):
                frames_to_process.append(c_frame)
    else:
        nb_frame = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT)-1)
        for c_frame in range(0, nb_frame):
            frames_to_process.append(c_frame)
    last_frame_to_process = max(frames_to_process)
    # save face detection
    c_frame = 0 
    fout = open(args['<output_file>'], 'w')
    while (c_frame<last_frame_to_process):
        frame = cv.QueryFrame(capture)
        c_frame = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_POS_FRAMES))
        if frame and c_frame in frames_to_process:
            print c_frame
            detected = cv.HaarDetectObjects(frame, 
                                            cascade, 
                                            storage, 
                                            float(args['--scaleFactor']), 
                                            int(args['--minNeighbors']), 
                                            cv.CV_HAAR_DO_CANNY_PRUNING, 
                                            minsize)
            for (x,y,w,h),n in detected:  
                fout.write(str(int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_POS_FRAMES))))
                fout.write(' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+' '+str(n)+'\n')
    fout.close()
