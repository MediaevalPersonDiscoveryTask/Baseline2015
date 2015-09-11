"""
create a video with bounding box on facetracks

Usage:
  extract_images_face_clusters.py <videoID> <videoFile> <output_path> <faceTrackPosition>  <faceTrackSegmentation> <diarization>
  extract_images_face_clusters.py -h | --help
"""

import numpy as np
import cv2, cv
from docopt import docopt
from mediaeval_util.repere import MESegParser

if __name__ == '__main__':
    args = docopt(__doc__)

    print "read facetrack position"
    facetracksPosition = {}
    face_seg, confs, timeToFrameID = MESegParser(args['<faceTrackSegmentation>'], args['<videoID>'])
    trackID_to_facetrackID = {}
    for s, t, l in face_seg.itertracks(label=True):
        trackID_to_facetrackID[int(l)] = int(t)

    for line in open(args['<faceTrackPosition>']).read().splitlines():
        frameID, trackID, xmin, ymin, w, h = map(int, line.split(' ')) 
        facetracksPosition.setdefault(trackID_to_facetrackID[trackID], {})
        facetracksPosition[trackID_to_facetrackID[trackID]][frameID] = xmin, ymin, xmin+w, ymin+h


    print "read facetrack segmentation"
    face_to_save = {}
    face_dia, confs, timeToFrameID = MESegParser(args['<diarization>'], args['<videoID>'])
    for s, t, l in face_dia.itertracks(label=True):
        l_frame = sorted(facetracksPosition[t].keys())
        pos_center_frame = len(l_frame)/2
        center_frameID = l_frame[pos_center_frame]
        xmin, ymin, xmax, ymax = facetracksPosition[t][center_frameID]
        face_to_save.setdefault(center_frameID, []).append([xmin, ymin, xmax, ymax, s, t, l])

    capture = cv2.VideoCapture(args['<videoFile>'])
    nb_frame = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT)-1)

    print "read video and save images"
    frameID = 0 
    while (frameID<nb_frame):
        ret, frame = capture.read()                             # read the video
        frameID = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES))     

        if ret and frameID in face_to_save:
            print frameID, len(face_to_save[frameID]), "images to save"
            for xmin, ymin, xmax, ymax, s, t, l in face_to_save[frameID]:
                name = args['<output_path>']+'/'+l+'_'+str(t)+'_'+str(s.start)+'_'+str(s.end)+'_'+str(xmin)+'_'+str(ymin)+'_'+str(xmax)+'_'+str(ymax)+'.jpg'
                cv2.imwrite(name, frame[ymin:ymax, xmin:xmax])
        


