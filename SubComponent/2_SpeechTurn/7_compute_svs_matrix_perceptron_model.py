"""
Compute speaker versus speaker distance

Usage:
  compute_svs_matrix_perceptron_model.py <list_video_file> <features_path> <model_file> <output_path>
  compute_svs_matrix_perceptron_model.py -h | --help
"""

from docopt import docopt
from sklearn.externals import joblib
import numpy as np

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

if __name__ == '__main__':
    arguments = docopt(__doc__)
    list_video_file = arguments['<list_video_file>']
    features_path = arguments['<features_path>']
    model_file = arguments['<model_file>']
    output_path = arguments['<output_path>']

    clas = joblib.load(model_file) 


    start_hist = 0.0
    end_hist = 1.02
    step_hist = 0.01
    l_range = list(drange(start_hist, end_hist, step_hist))
    l_score_1 = []
    l_score_0 = []

    for video in open(list_video_file):
        video = video[:-1]
        #fout = open(output_path+'/'+video+'.svs', 'w')
        for line in open(features_path+'/'+video+'.features_svs'):            
            n1, n2, BIC_dist, min_dur, max_dur, same_cluster, same_name = line[:-1].split(' ')
            d = [float(BIC_dist), float(min_dur), float(max_dur), int(same_cluster)]

            #fout.write(n1+' '+n2+' '+str(clas.decision_function([d])[0][1])+'\n')
            if same_name != '?': 
                score = clas.predict_proba([d])[0][1]
                if same_name == '1':
                    l_score_1.append(score)
                else:
                    l_score_0.append(score)


        #fout.close()

    hist_1 = np.histogram(l_score_1, l_range)
    hist_0 = np.histogram(l_score_0, l_range)

    for i in range(len(l_range)-1):
        print str(l_range[i]).replace(',', '.'), 
        f = hist_0[0][i]
        t = hist_1[0][i]
        print str(round(float(f)/float(len(l_score_0))*100,2)).replace('.', ','),
        print str(round(float(t)/float(len(l_score_1))*100,2)).replace('.', ',')

