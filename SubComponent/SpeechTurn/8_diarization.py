"""
Learn a normalisation model to compute svs matrix

Usage:
  8_diarization.py <videoID> <st_seg> <matrix> <output_diarization> <threshold>
  8_diarization.py -h | --help
"""

from docopt import docopt
from pyannote.parser import MDTMParser
import numpy as np
import copy


def diarization(mat, original_mat, threshold):
    # agglomerative clustering
    max_proba = np.inf
    while(max_proba>threshold):
        max_proba = 0.0
        best_c1, best_c2 = '', ''
        for c1 in mat:
            for c2 in mat[c1]:
                if mat[c1][c2] > max_proba:
                    max_proba = mat[c1][c2]
                    best_c1 = c1
                    best_c2 = c2

        if max_proba>threshold:
            new_c = best_c1+';'+best_c2

            # remove best_c1 best_c2 from mat
            if best_c1 in mat:
                del mat[best_c1]
            if best_c2 in mat:
                del mat[best_c2]
            for clus in mat:
                if best_c1 in mat[clus]:
                    del mat[clus][best_c1]
                if best_c2 in mat[clus]:
                    del mat[clus][best_c2]

            # add new_c in mat
            for c in mat:        
                moyenne = 0.0
                count_d = len(new_c.split(';')) + len(c.split(';'))             
                for track_new_c in new_c.split(';'):
                    for track_c in c.split(';'):   
                        moyenne += original_mat[track_new_c][track_c]
                mat[c][new_c] = moyenne/count_d  
    return mat

if __name__ == '__main__':
    args = docopt(__doc__)

    threshold = float(args['<threshold>'])

    # read matrix
    mat = {}
    original_mat = {}
    for line in open(args['<matrix>']).read().splitlines():
        st1, st2, proba = line.split(' ')  
        proba = float(proba)
        mat.setdefault(st1, {})
        mat[st1][st2] = proba
        original_mat.setdefault(st1, {})
        original_mat[st1][st2] = proba
        original_mat.setdefault(st2, {})
        original_mat[st2][st1] = proba

    mat = diarization(mat, original_mat, threshold)

    st_to_clusters = {}
    for c1 in mat:
        for t1 in c1.split(';'):
            st_to_clusters[t1] = c1
        for c2 in mat[c1]:
            for t2 in c2.split(';'):
                st_to_clusters[t2] = c2

    seg_st = MDTMParser().read(args['<st_seg>'])(uri=args['<videoID>'], modality="speaker")

    fout = open(args['<output_diarization>'], 'w')
    for s, t, l in seg_st.itertracks(label=True):
        fout.write(args['<videoID>']+' 1 '+str(s.start)+' '+str(s.duration)+' speaker na na '+st_to_clusters[l]+'\n')
    fout.close()



