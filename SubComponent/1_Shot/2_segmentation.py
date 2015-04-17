"""
Find shot boundaries with a descriptor (<descriptor>) based on histogram and optical flow between consecutive frames.
Write frame selected as shot boundarires into <output_file>.

Usage:
  segmentation.py <videoID> <descriptor> <output_file> [--threshold=<t>] [--min_duration=md] [--X_sigma=<Xs>] [--mix_type=<m>] [--uem=<uem>]
  segmentation.py -h | --help
Options:
  --min_duration=<md>  minimum duration (in frame) of a shot (>=2) [default: 50]  
  --threshold=<t>      threshold value  (0.0 > t > 1.0), default = mean(score) - X_sigma * std(score)
  --X_sigma=<Xs>       threshold = mean(score) - X_sigma * std(score) [default: 1.0]
  --mix_type=<m>       mix between the descriptor (mean or product) [default: mean]
  --uem=<uem>          uem file  
"""

from docopt import docopt
from scipy.signal import argrelmin
import numpy as np

def read_uem(uem_file, videoID):
    if not args['--uem']:
        return [[-np.inf, +np.inf]]
    l_uem = []
    for line in open(uem_file).read().splitlines():
        v, p, start, end = line.split(' ')
        if v == videoID:
            l_uem.append([float(start), float(end)])
    return l_uem

def shot_in_uem(l_uem, startTime, endTime):
    for s_uem, e_uem in l_uem:
        if startTime >= s_uem and endTime <= e_uem:
            return True
    return False

def f_mix(mix_type, hist, OF):
    if mix_type == 'mean':
        return (hist+OF)/2
    return hist*OF

def read_descriptor(descriptor_file, mix_type):
    x, y = [], []
    for line in open(descriptor_file).read().splitlines():
        frame, time, hist, OF = line.split(' ')
        x.append(f_mix(mix_type, max(0.0, float(hist)), float(OF)))
        y.append([int(frame), float(time)])
    return x, y

def select_threshold(threshold, x):
    # threshold in argument or computed
    if threshold:
        return float(threshold)
    else:
        return np.mean(x) - float(args['--X_sigma'])*np.std(x)

def find_cut(x, y, threshold):
    # find local minima
    minima = argrelmin(np.array(x), order=int(args['--min_duration'])/2)
    # select frame in minima with a score lower than the threshold
    l_cut = []
    start = y[0]
    for i in list(minima)[0]:
        if x[i] < threshold:
            end = y[i-1]
            l_cut.append([start, end])
            start = y[i]
    end = y[-1]
    l_cut.append([start, end])
    return l_cut

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    # read uem file if exist
    l_uem = read_uem(args['--uem'], args['<videoID>'])

    # read descriptor
    x, y = read_descriptor(args['<descriptor>'], args['--mix_type'])

    #select threshold
    threshold = select_threshold(args['--threshold'], x)
    print 'threshold used', threshold

    # find shot boudaries
    l_cut = find_cut(x, y, threshold)

    # Save segmentation
    nb_shot=0
    fout = open(args['<output_file>'], 'w')
    for start, end in l_cut:
        nb_shot+=1
        if shot_in_uem(l_uem, start[1], end[1]) and end[0]-start[0] >= int(args['--min_duration']):
            fout.write(args['<videoID>'])
            fout.write(' %06d' % (nb_shot))
            fout.write(' %09.3f %09.3f' % (start[1], end[1]))
            fout.write(' %07d %07d' % (start[0], end[0]))
            fout.write('\n')
    fout.close()
