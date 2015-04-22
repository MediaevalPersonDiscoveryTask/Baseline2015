"""
Compute head versus head distance

Usage:
  compute_hvh_matrix.py <facetracks_descriptor> <matrix_output>
  compute_hvh_matrix.py -h | --help
"""

from docopt import docopt
import numpy as np

def sqdist(desc1, desc2):
	dist = 0.0
	for i in range(desc1.shape[0]):
		d = desc1[i] - desc2[i]
		dist += d*d
	return dist

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    # read face track descriptors
    dic = {}
    for line in open(args['<facetracks_descriptor>']).read().splitlines():
        l = line.split(' ')
        dic[int(l[0])] = [int(l[1]), float(l[2]), np.array(l[3:], dtype='|S20').astype(np.float)]

    # compute and save distance between face tracks
    fout = open(args['<matrix_output>'], 'w')
    for h1 in sorted(dic):
        for h2 in sorted(dic):
            if h1 < h2:
                fout.write(str(h1)+' '+str(h2))
                fout.write(' '+str(dic[h1][0])+' '+str(dic[h2][0]))             # number of HoG extracted
                fout.write(' '+str(dic[h1][1])+' '+str(dic[h2][1]))             # mean distance of the central desccriptor to other descriptors
                fout.write(' '+str(round(sqdist(dic[h1][2], dic[h2][2]), 2)))   # l2 distance between descriptors
                fout.write('\n')
    fout.close()
                    
                