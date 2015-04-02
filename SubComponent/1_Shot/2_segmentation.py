"""
Find shot boundaries based on histogram and optical flow between 2 consecutive frames

Usage:
  cut_thr_selection.py <descriptor> <output_file> [--threshold=<t>] [--min_duration=<md>] [--X_sigma=<Xs>] [--mix=<m>]
  cut_thr_selection.py -h | --help
Options:
  --min_duration=<md>  minimum duration (in frame) of a shot (>0) [default: 15]  
  --threshold=<t>      threshold value  (0.0 > t > 1.0), default = mean(score) + X_sigma * std(score)
  --X_sigma=<Xs>       threshold = mean(score) + X_sigma * std(score) [default: 1.0]
  --mix=<m>            mix between the descriptor (mean or product) [default: mean]
"""

from docopt import docopt
from scipy.signal import argrelmax
import numpy as np
from itertools import izip, islice

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
    
    # read descriptor
    x = []
    y = []
    for line in open(args['<descriptor>']).read().splitlines():
        frame, hist, OF = line.split(' ')
        hist = float(hist)
        if hist < 0.0:
            hist = 0.0
        hist = 1.0-hist
        OF = float(OF)
        OF = 1.0-OF

        if args['--mix'] == 'mean':
            score = (float(hist)+float(OF))/2
        elif args['--mix'] == 'product':
            score = float(hist)*float(OF)

        x.append(score)
        y.append(int(frame))

    # threshold in argument or 
    threshold = args['--threshold']
    if threshold:
        float(threshold)
    else:
        threshold = np.mean(x) + float(args['--X_sigma'])*np.std(x)
    print 'threshold used', threshold

    # find local minima
    minima = argrelmax(np.array(x), order=int(args['--min_duration']))

    # select frame in minima with a score lower than the threshold
    # and save segmentation
    fout = open(args['<output_file>'], 'w')
    fout.write('0\n')
    for i in list(minima)[0]:
        if x[i] > threshold:
            fout.write(str(y[i])+'\n')
    fout.write(str(y[-1]))

    
