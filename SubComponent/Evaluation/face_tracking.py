"""
Usage:
  face_tracking.py <video_list> <facetrack_pos> <reference_head_position_path>
  face_tracking.py -h | --help
"""

from docopt import docopt
from mediaeval_util.repere import read_ref_facetrack_position, align_facetrack_ref

if __name__ == '__main__':
    args = docopt(__doc__)

    nb_hyp = 0.0
    nb_ref = 0.0
    nb_correct=0.0

    for videoID in open(args['<video_list>']).read().splitlines():
        print videoID

        ref_f = read_ref_facetrack_position(args['<reference_head_position_path>']+'/'+videoID+'.position', 0)

        facetracks = {}
        l_ft = []
        for line in open(args['<facetrack_pos>']+'/'+videoID+'.facetrack').read().splitlines():
            frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
            if frameID in ref_f:
                l_ft.append(faceID)
                facetracks.setdefault(frameID, {})
                facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h

        facetrack_vs_ref = align_facetrack_ref(ref_f, facetracks)

        nb_hyp+=len(l_ft)
        for frameID in ref_f:
            nb_ref+=len(ref_f[frameID])

        nb_correct+=len(facetrack_vs_ref)

    print 'precision:', round(nb_correct/nb_hyp,3)*100, '%   recall:', round(nb_correct/nb_ref,3)*100, '%'


