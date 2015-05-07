"""
Learn a normalisation model to compute svs matrix

Usage:
  8_diarization.py <videoID> <linearClustering> <probaMatrix> <diarization> [--threshold=<t>]
  8_diarization.py -h | --help
Options:
  --threshold=<t>  stop criterion of the agglomerative clustering [default: 0.28]
"""

from docopt import docopt
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

    for line in open(args['<probaMatrix>']).read().splitlines():
        t1, t2, p = line.split(' ')
        X[track_to_indice[int(t1)]][track_to_indice[int(t2)]] = 1.0-float(p)

    # compute diarization
    y = spatial.distance.squareform(X, checks=False)
    Z = cluster.hierarchy.average(y)
    clusters = cluster.hierarchy.fcluster(Z, 1.0-float(args['--threshold']), criterion='distance')

    for s, t, l in st_seg.itertracks(label=True):
        st_seg[s, t] = 'c_'+str(clusters[track_to_indice[t]])

    # save clustering
    MESegWriter(st_seg, {}, args['<diarization>'], args['<videoID>'], {})

    