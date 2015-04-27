"""
Learn a normalisation model to compute svs matrix

Usage:
  6_learn_normalisation_model.py <video_list> <segmentation_path> <matrix_path> <reference_path> <output_model_file>
  6_learn_normalisation_model.py -h | --help
"""

from docopt import docopt
from pyannote.parser import MDTMParser
from mediaeval_util.repere import parser_atseg
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron
from sklearn.externals import joblib
from pyannote.algorithms.tagging import ArgMaxDirectTagger

if __name__ == '__main__':
    args = docopt(__doc__)

    X = []
    Y = []
    for line in open(args['<video_list>']).read().splitlines():
        videoID = line.split('\t')[0]
        print videoID
        ref = parser_atseg(args['<reference_path>']+'/'+videoID+'.atseg', videoID)
        seg_st = MDTMParser().read(args['<segmentation_path>']+'/'+videoID+'.mdtm')(uri=videoID, modality="speaker")

        direct = ArgMaxDirectTagger()
        named_st = direct(ref, seg_st)

        dur = {}
        for seg in seg_st.get_timeline():
            st = list(seg_st.get_labels(seg))[0] 
            dur[st] = seg.duration

        label_to_track = {}
        for s, t, l in seg_st.itertracks(label=True):
            label_to_track[l] = t

        track_to_name = {}
        for s, t, l in named_st.itertracks(label=True):
            if 'st_' not in l:
                track_to_name[t] = l

        for line in open(args['<matrix_path>']+'/'+videoID+'.mat').read().splitlines():
            st1, st2, BIC_dist = line.split(' ')
            t1 = label_to_track[st1]
            t2 = label_to_track[st2]
            if t1 in track_to_name and t2 in track_to_name:
                #X.append([float(BIC_dist), min(dur[st1], dur[st2]), max(dur[st1], dur[st2])])
                X.append([float(BIC_dist)])
                if track_to_name[t1] == track_to_name[t2]:
                    Y.append(1)
                else:
                    Y.append(0)

    # train model
    clf = CalibratedClassifierCV(Perceptron(), method='isotonic')
    clf.fit(X, Y) 
    # save model
    joblib.dump(clf, args['<output_model_file>']) 
