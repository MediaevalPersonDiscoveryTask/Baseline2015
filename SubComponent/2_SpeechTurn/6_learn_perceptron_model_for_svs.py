"""
Learn a perceptron model to compute svs matrix

Usage:
  learn_perceptron_model_for_svs.py <list_video_file> <features_path> <output_model_file>
  learn_perceptron_model_for_svs.py -h | --help
"""

from docopt import docopt
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

if __name__ == '__main__':
    arguments = docopt(__doc__)
    list_video_file = arguments['<list_video_file>']
    features_path = arguments['<features_path>']
    output_model_file = arguments['<output_model_file>']
    
    X = []
    Y = []      
    l_video = []
    for video in open(list_video_file):
        l_video.append(video[:-1])

    for video in l_video:
        for line in open(features_path+'/'+video+'.features_svs'):            
            n1, n2, BIC_dist, min_dur, max_dur, same_cluster, same_name = line[:-1].split(' ')
            if same_name != '?':                
                X.append([float(BIC_dist), float(min_dur), float(max_dur), int(same_cluster)])
                Y.append(int(same_name))
    print 'nb_desc:', len(X)

    print 'train classifier'
    clf = LogisticRegression()
    clf.fit(X, Y) 

    joblib.dump(clf, output_model_file) 
