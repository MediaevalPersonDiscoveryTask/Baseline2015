"""
Learn a normalisation model to compute svs matrix

Usage:
  8_diarization.py <videoID> <linearClustering> <probaMatrix> <diarization> [--threshold=<t>]
  8_diarization.py -h | --help
Options:
  --threshold=<t>  stop criterion of the agglomerative clustering [default: 0.28]
"""

from docopt import docopt
from pyannote.core.matrix import LabelMatrix
from mediaeval_util.repere import MESegParser, MESegWriter
import numpy as np
from scipy import spatial, cluster

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)


    st_seg, confs, timeToFrameID = MESegParser(args['<linearClustering>'], args['<videoID>'])
    track_to_indice = {}
    for s, t, l in st_seg.itertracks(label=True):
        track_to_indice[t] = len(track_to_indice)

    # read matrix
    N = len(st_seg.labels())
    X = np.zeros((N, N))

    m = LabelMatrix.load(args['<probaMatrix>'])

    # compute score between speech turn and save it
    for s1, t1 in m.get_rows():
        for s2, t2 in m.get_columns():
            X[track_to_indice[t1]][track_to_indice[t2]] = 1.0-m[(s1,t1), (s2,t2)]

    # compute diarization
    y = spatial.distance.squareform(X, checks=False)
    Z = cluster.hierarchy.average(y)
    clusters = cluster.hierarchy.fcluster(Z, 1.0-float(args['--threshold']), criterion='distance')

    for s, t, l in st_seg.itertracks(label=True):
        st_seg[s, t] = 'c_'+str(clusters[track_to_indice[t]])

    # save clustering
    MESegWriter(st_seg, {}, args['<diarization>'], args['<videoID>'], {})

    