"""
Usage:
  face_l2_mat.py <video_list> <facetrack_pos> <l2_mat> <reference_head_position_path>
  face_l2_mat.py -h | --help
"""

from docopt import docopt
from mediaeval_util.repere import read_ref_facetrack_position, align_facetrack_ref
import os
import numpy as np

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

if __name__ == '__main__':
    args = docopt(__doc__)

    l_true = []
    l_false = []

    for videoID in open(args['<video_list>']).read().splitlines():

        if os.path.isfile(args['<l2_mat>']+'/'+videoID+'.mat'):

            ref_f = read_ref_facetrack_position(args['<reference_head_position_path>']+'/'+videoID+'.position', 0)
            facetracks = {}
            for line in open(args['<facetrack_pos>']+'/'+videoID+'.facetrack').read().splitlines():
                frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
                facetracks.setdefault(frameID, {})
                facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h
            facetrack_vs_ref = align_facetrack_ref(ref_f, facetracks)

            for line in open(args['<l2_mat>']+'/'+videoID+'.mat').read().splitlines():
                FaceId1, FaceId2, nb_Hog1, nb_Hog2, min_dist_Hog1, min_dist_Hog2, l2 = line.split(' ')
                FaceId1 = int(FaceId1)
                FaceId2 = int(FaceId2)
                l2 = float(l2)
                if FaceId1 in facetrack_vs_ref and FaceId2 in facetrack_vs_ref:
                    if facetrack_vs_ref[FaceId1] == facetrack_vs_ref[FaceId2]:
                        l_true.append(l2)
                    else:
                        l_false.append(l2)

    print len(l_false), len(l_true)

    l_range = list(drange(0.0, 100.0, 1.0))

    hist_1 = np.histogram(l_true, l_range)    
    hist_0 = np.histogram(l_false, l_range)

    for i in range(len(l_range)-1):
        print str(l_range[i]).replace('.', ','),
        print str(round(float(hist_0[0][i])/float(len(l_false))*100,2)).replace('.', ','),
        print str(round(float(hist_1[0][i])/float(len(l_true))*100,2)).replace('.', ',')
