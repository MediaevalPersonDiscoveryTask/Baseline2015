"""
compute probability that a facetrack is speaking

Usage:
  proba_speaking_face.py <video_list> <matrix_path> <st_seg> <reference_speaker> <facetrack_pos> <reference_head>
  proba_speaking_face.py -h | --help
"""

from docopt import docopt
from mediaeval_util.repere import IDXHack, read_ref_facetrack_position, align_facetrack_ref, align_st_ref
from sklearn.externals import joblib
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

        st_vs_ref = align_st_ref(args['<st_seg>'], args['<reference_speaker>'], videoID)

        ref_f = read_ref_facetrack_position(args['<reference_head>']+videoID+'.position', 0)
        facetracks = {}
        l_facetrack_in_annotated_frame = set([])
        for line in open(args['<facetrack_pos>']+videoID+'.facetrack').read().splitlines():
            frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
            facetracks.setdefault(frameID, {})
            facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h
            if frameID in ref_f:
                l_facetrack_in_annotated_frame.add(faceID)            

        facetrack_vs_ref = align_facetrack_ref(ref_f, facetracks)

        for line in open(args['<matrix_path>']+videoID+'.mat').read().splitlines():
            st, faceID, proba = line.split(' ')
            faceID = int(faceID)
            if faceID in l_facetrack_in_annotated_frame:
                proba = float(proba)
                SpeakingFace = 0
                if faceID in facetrack_vs_ref and st in st_vs_ref:
                    if facetrack_vs_ref[faceID] == st_vs_ref[st]:
                        SpeakingFace = 1
                if SpeakingFace == 1:
                    l_true.append(proba)
                else:
                    l_false.append(proba)

    print len(l_false), len(l_true)

    l_range = list(drange(0.0, 1.0, 0.01))

    hist_1 = np.histogram(l_true, l_range)    
    hist_0 = np.histogram(l_false, l_range)

    for i in range(len(l_range)-1):
        print str(l_range[i]).replace('.', ','),
        print str(round(float(hist_0[0][i])/float(len(l_false))*100,2)).replace('.', ','),
        print str(round(float(hist_1[0][i])/float(len(l_true))*100,2)).replace('.', ',')


