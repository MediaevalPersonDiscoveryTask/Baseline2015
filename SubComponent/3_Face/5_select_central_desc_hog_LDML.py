"""
Select central face descriptor projected by LDML of a face track

Usage:
  select_central_desc_hog_LDML.py <video> <output_path> <hog_path> <file_ldml>
  select_central_desc_hog_LDML.py -h | --help
"""

from docopt import docopt
import sys, os, glob
import numpy as np
import math 

def sqdist(desc1, desc2):
    dim = min(desc1.shape[0], desc2.shape[0])
    dist = 0.0
    for i in range(dim):
        d = desc1[i] - desc2[i]
        dist += d*d
    return dist

if __name__ == '__main__':
    arguments = docopt(__doc__)
    video = arguments['<video>']
    path_out = arguments['<output_path>']
    hog_path = arguments['<hog_path>']
    file_ldml = arguments['<file_ldml>']

    K = 100
    N = 7
    nbins = 8
    cell_sz = 7
    block_sz = 7
    HOG_DIM = N*nbins*cell_sz*block_sz

    file_hog = hog_path+'/'+video.split('/')[-1]+'.face_descriptor'
    fout = open(path_out+'/'+video.split('/')[-1]+'.central_HoG_LDML', 'w')

    L = np.fromfile(file_ldml, sep=' ')
    L = np.array(L)
    L = L.reshape(K, HOG_DIM)  

    desc_face = {}
    for line in open(file_hog):
        l = line[:-1].split(' ')
        i_face = int(float(l[1]))
        desc = map(float, l[2:2+HOG_DIM])
        set_desc = set(desc)

        if set_desc != set(['-1']) and desc != [''] and desc != []:
            desc_ldml = np.dot(np.array(desc, dtype='|S20').astype(np.float), L.T)
            desc_face.setdefault(i_face, []).append(np.array(desc_ldml))
    
    for i_face, l_desc in sorted(desc_face.items()):
        min_dist = 10000.0
        best_desc = []
        for desc1 in l_desc:
            l_d = []
            for desc2 in l_desc:
                l_d.append(sqdist(desc1, desc2))
            d = np.mean(np.array(l_d))
            if d < min_dist:
                min_dist = d
                best_desc = desc1

        fout.write(str(i_face)+' '+str(min_dist))
        for e in best_desc:
            fout.write(' '+str(e))
        fout.write('\n')
    fout.close()
