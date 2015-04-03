"""
Learn a normalisation model to compute svs matrix

Usage:
  6_learn_normalisation_model.py <list_video_file> <segmentation_path> <matrix_path> <reference> <output_model_file> [--min_cooc=<mc>]
  6_learn_normalisation_model.py -h | --help
Options:
  --min_cooc=<mc>   minimim duration to consider that a speech turn correspond to a speaker of the reference (>0.0) [default: 0.5]  
"""

from docopt import docopt
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

def align_ref_st(seg_st, ref, min_cooc):
    name_st = []
    for seg in seg_st.get_timeline():
        st = list(seg_st.get_labels(seg))[0] 
        best_cooc = 0.0
        for seg_ref in ref.get_timeline():
            cooc_dur = (seg & seg_ref).duration
            if cooc_dur > best_cooc and cooc_dur >= min_cooc:
                name_st[st] = list(ref.get_labels(seg))[0] 
                best_cooc = cooc_dur

    return name_st


if __name__ == '__main__':
    arguments = docopt(__doc__)
    list_video_file = arguments['<list_video_file>']
    segmentation = arguments['<segmentation>']
    matrix_path = arguments['<matrix_path>']
    output_model_file = arguments['<output_model_file>']
    
    refs = MDTMParser().read(args['<reference>'])
 
    X = []
    Y = []      
    l_video = []
    for video in open(list_video_file):
        l_video.append(video[:-1])

    for video in l_video:
        ref = refs(uri=video, modality="speaker")
        seg_st = MDTMParser().read(args['<segmentation_path>']+'/'+video+'.mdtm')(uri=video, modality="speaker")
        name_st = align_ref_st(seg_st, ref, float(args['--min_cooc']))

        dur = {}
        for seg in seg_st.get_timeline():
            st = list(seg_st.get_labels(seg))[0] 
            dur[st] = seg.duration

        for line in open(args['<matrix_path>']+'/'+video+'.mat').read().splitlines():
            st1, st2, BIC_dist = line.split(' ')
            if st1 in name_st and st2 in name_st:
                X.append([float(BIC_dist), min(dur(st1), dur(st2)), max(dur(st1), dur(st2))])
                if name_st[st1] == name_st[st2]:
                    Y.append(1)
                else:
                    Y.append(0)

    # train model
    clf = LogisticRegression()
    clf.fit(X, Y) 
    # save model
    joblib.dump(clf, output_model_file) 
