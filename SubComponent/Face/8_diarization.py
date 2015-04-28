"""
Learn a normalisation model to compute svs matrix

Usage:
  8_diarization.py <videoID> <facetrack_seg> <matrix> <output_diarization> [--threshold=<t>]
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

    faceID_to_indice = {}
    indice_to_face = {}
    face_seg = {}
    i=0
    for line in open(args['<facetrack_seg>']).read().splitlines():
        faceID, startTime, endTime, startFrame, endFrame = line.split(' ')        
        faceID_to_indice[faceID] = i
        indice_to_face[i] = faceID
        face_seg[i] = [startTime, endTime]
        i+=1

    # read matrix
    N = len(faceID_to_indice)
    X = np.zeros((N, N))
    for line in open(args['<matrix>']).read().splitlines():
        ft1, ft2, proba = line.split(' ')
        dist = 1.0-float(proba)
        X[faceID_to_indice[ft1]][faceID_to_indice[ft2]] = dist
        X[faceID_to_indice[ft2]][faceID_to_indice[ft1]] = dist

    y = spatial.distance.squareform(X, checks=False)
    Z = cluster.hierarchy.average(y)
    clusters = cluster.hierarchy.fcluster(Z, 1.0-float(args['--threshold']), criterion='distance')

    clusName = {}
    for i in sorted(indice_to_face):
        clusID = clusters[i]
        if clusID not in clus_name:
            clusName[clusID] = indice_to_face[i]
        else:
            clusName[clusID] += ";"+indice_to_face[i]

    fout = open(args['<output_diarization>'], 'w')
    for i in sorted(face_seg):
        startTime, endTime = face_seg[i]
        fout.write(args['<videoID>']+' 1 '+str(startTime)+' '+str(endTime)+' head na na '+str(clusName[clusters[i]])+'\n')
    fout.close()
