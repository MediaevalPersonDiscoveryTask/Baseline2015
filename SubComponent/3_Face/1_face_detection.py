"""
Face detection based on Viola & Jones method

Usage:
  face_detection.py <video_file> <output_file> <cascade_file> [--scaleFactor=<sf>] [--minNeighbors=<mn>] [--minSize=<min>]
  face_detection.py -h | --help
Options:
  --scaleFactor=<sf>    scaleFactor value (>1.0) [default: 1.1]
  --minNeighbors=<mn>   minNeighbors value (>1) [default: 4]
  --minSize=<min>       min size of a face (>1) [default: 50]
"""

from docopt import docopt
import cv2, cv

if __name__ == '__main__':
    # read args
    args = docopt(__doc__)
    # storage for face detected  
    storage = cv.CreateMemStorage()
    # load cascade file
    cascade = cv.Load(args['<cascade_file>'])
    # read video
    capture = cv.CaptureFromFile(args['<video_file>'])
    nb_frame = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT)-1)
    frame = cv.QueryFrame(capture)
    c_frame = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_POS_FRAMES))

    # save face detection
    fout = open(args['<output_file>'], 'w')
    while (c_frame<nb_frame):
        frame = cv.QueryFrame(capture)
        detected = cv.HaarDetectObjects(frame, 
                                        cascade, 
                                        storage, 
                                        float(args['--scaleFactor']), 
                                        int(args['--minNeighbors']), 
                                        cv.CV_HAAR_DO_CANNY_PRUNING, 
                                        (int(args['--minSize']), int(args['--minSize'])))
        for (x,y,w,h),n in detected:  
            fout.write(str(int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_POS_FRAMES))))
            fout.write(' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+' '+str(n)+'\n')
    fout.close()
