"""
Select central face descriptor projected by LDML of a face track

Usage:
  select_central_desc_hog_LDML.py <hog> <ldml_matrix> <output_file> 
  select_central_desc_hog_LDML.py -h | --help
"""

from docopt import docopt
import numpy as np

def sqdist(desc1, desc2):
    dim = min(desc1.shape[0], desc2.shape[0])
    dist = 0.0
    for i in range(dim):
        d = desc1[i] - desc2[i]
        dist += d*d
    return dist

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    # size of the descriptor before projection
    N = 7
    nbins = 8
    cell_sz = 7
    block_sz = 7
    HOG_DIM = N*nbins*cell_sz*block_sz
    # size of the descriptor after projection
    K = 100 

    # load projection matrix
    L = np.fromfile(args['<ldml_matrix>'], sep=' ')
    L = np.array(L)
    L = L.reshape(K, HOG_DIM)  

    # read and project descriptor
    desc_face = {}
    for line in open(args['<hog>']).read().splitlines():
        l = line.split(' ')
        desc_ldml = np.dot(np.array(map(float, l[2:2+HOG_DIM]), dtype='|S20').astype(np.float), L.T)
        desc_face.setdefault(int(l[1]), []).append(np.array(desc_ldml))

    # find central descriptor for each face track and save it
    fout = open(args['<output_file>'], 'w')    
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
