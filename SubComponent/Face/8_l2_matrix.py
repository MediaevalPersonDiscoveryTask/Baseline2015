"""
Compute head versus head distance

Usage:
  compute_hvh_matrix.py <faceTrackDescriptor> <l2Matrix> <faceTrackSegmentation>
  compute_hvh_matrix.py -h | --help
"""

from docopt import docopt
import numpy as np
from scipy import spatial
import pickle

def sqdist(desc1, desc2):
	dist = 0.0
	for i in range(desc1.shape[0]):
		d = desc1[i] - desc2[i]
		dist += d*d
	return dist

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    # read face track segmentation
    l_faceID = []
    for line in open(args['<faceTrackSegmentation>']).read().splitlines():
        v, startTime, endTime, startFrame, endFrame, trackID, faceID, conf = line[:-1].split(' ')
        l_faceID.append(int(faceID))
    l_faceID = sorted(l_faceID)

    # read face track descriptors
    dic = {}
    for line in open(args['<faceTrackDescriptor>']).read().splitlines():
        l = line.split(' ')
        dic[int(l[0])] = np.array(l[3:], dtype='|S20').astype(np.float)

    # initialize matrix
    N = len(l_faceID)
    X = np.zeros((N, N))
    X[:] = np.nan

    # compute and save distance between face tracks
    for i in range(N):
        for j in range(i+1, N):
            if l_faceID[i] in dic and l_faceID[j] in dic:
                X[i][j] = sqdist(dic[l_faceID[i]], dic[l_faceID[j]])

    y = spatial.distance.squareform(X, checks=False)
    y = y.astype(np.float16)
    pickle.dump(y, open(args['<l2Matrix>'], "wb" ))

                