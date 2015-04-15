"""
Compute speaker versus speaker distance

Usage:
  normalisation_matrix.py <videoID> <input_seg> <input_mat> <model_file> <output_mat>
  normalisation_matrix.py -h | --help
"""

from docopt import docopt
from sklearn.externals import joblib
import numpy as np

if __name__ == '__main__':
    arguments = docopt(__doc__)

    # open model
    clas = joblib.load(arguments['<model_file>']) 

    # read segmentation
    seg_st = MDTMParser().read(args['<input_seg>'])(uri=arguments['<videoID>'], modality="speaker")

    dur = {}
    for seg in seg_st.get_timeline():
        st = list(seg_st.get_labels(seg))[0] 
        dur[st] = seg.duration

    # compute score between speech turn and save it
    fout = open(arguments['<output_mat>'], 'w')
    for line in open(args['<input_mat>']).read().splitlines():
        st1, st2, BIC_dist = line.split(' ')
        desc = [float(BIC_dist), min(dur(st1), dur(st2)), max(dur(st1), dur(st2))]
        fout.write(st1+' '+st2+' '+str(clas.predict_proba([desc])[0][1])+'\n')
    fout.close()
