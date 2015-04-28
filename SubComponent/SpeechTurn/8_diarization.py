"""
Learn a normalisation model to compute svs matrix

Usage:
  8_diarization.py <videoID> <st_seg> <matrix> <output_diarization> [--threshold=<t>]
  8_diarization.py -h | --help
Options:
  --threshold=<t>  stop criterion of the agglomerative clustering [default: 0.27]
"""

from docopt import docopt
from pyannote.parser import MDTMParser
import numpy as np
from scipy import spatial, cluster

if __name__ == '__main__':
    args = docopt(__doc__)

    label_to_indice = {}
    seg_st = MDTMParser().read(args['<st_seg>'])(uri=args['<videoID>'], modality="speaker")
    for s, t, l in seg_st.itertracks(label=True):
        label_to_indice[l] = t

    # read matrix
    N = len(label_to_indice)
    X = np.zeros((N, N))
    for line in open(args['<matrix>']).read().splitlines():
        st1, st2, proba = line.split(' ')  
        proba = 1.0-float(proba)
        X[label_to_indice[st1]][label_to_indice[st2]] = proba
        X[label_to_indice[st2]][label_to_indice[st1]] = proba

    y = spatial.distance.squareform(X, checks=False)
    Z = cluster.hierarchy.average(y)
    clusters = cluster.hierarchy.fcluster(Z, 1.0-float(args['--threshold']), criterion='distance')

    st_to_clus = {}
    for c, st in zip(clusters, label_to_indice):
        st_to_clus[st] = c

    fout = open(args['<output_diarization>'], 'w')
    for s, t, l in seg_st.itertracks(label=True):
        fout.write(args['<videoID>']+' 1 '+str(s.start)+' '+str(s.duration)+' speaker na na spk_'+str(st_to_clus[l])+'\n')
    fout.close()
    


