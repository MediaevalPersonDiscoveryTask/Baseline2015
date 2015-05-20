"""
Learn a normalisation model to compute svs matrix

Usage:
  11_diarization.py <videoID> <faceTrackSegmentation> <probaMatrix> <diarization> [--threshold=<t>]
  11_diarization.py -h | --help
Options:
  --threshold=<t>  stop criterion of the agglomerative clustering [default: 0.27]
"""

from docopt import docopt
from mediaeval_util.repere import MESegParser, MESegWriter
import numpy as np
import pickle
from scipy import spatial, cluster

def f(x):
    return 1.0-x

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    # read face segmentation
    l_faceID = []
    face_seg, confs, timeToFrameID = MESegParser(args['<faceTrackSegmentation>'], args['<videoID>'])
    for s, t, l in face_seg.itertracks(label=True):
        l_faceID.append(int(l))
    l_faceID.sort()

    # read matrix
    y = pickle.load(open(args['<probaMatrix>'], "rb" ) )
    f = np.vectorize(f)  # or use a different name if you want to keep the original f
    y = f(y)

    # compute clustering
    Z = cluster.hierarchy.average(y)
    clusters = cluster.hierarchy.fcluster(Z, f(float(args['--threshold'])), criterion='distance')

    # find correspondance between faceID and clusterID
    faceID_cluster = {}
    for i in range(len(clusters)):
        faceID_cluster[l_faceID[i]] = clusters[i]
    
    # rename facetrack by clusterID
    for s, t, l in face_seg.itertracks(label=True):
        face_seg[s, t] = 'c_'+str(faceID_cluster[int(l)])

    # save clustering
    MESegWriter(face_seg, {}, args['<diarization>'], args['<videoID>'], {})
    


