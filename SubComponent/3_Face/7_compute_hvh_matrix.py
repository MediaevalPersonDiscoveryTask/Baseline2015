"""
Compute head versus head distance

Usage:
  compute_hvh_matrix.py <file_desc_in> <file_hvh_out>
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
    arguments = docopt(__doc__)
    file_desc_in = arguments['<file_desc_in>']
    file_hvh_out = arguments['<file_hvh_out>']

    dic = {}
    for line in open(file_desc_in):
        l = line[:-1].split(' ')
        dic[int(l[0])] = np.array(l[2:], dtype='|S20').astype(np.float)

    fout = open(file_hvh_out, 'w')
    for h1 in sorted(dic):
        for h2 in sorted(dic):
            if h1 < h2:
                fout.write(str(h1)+' '+str(h2)+' '+str(round(sqdist(dic[h1], dic[h2]), 2))+'\n')
    fout.close()
                    
                