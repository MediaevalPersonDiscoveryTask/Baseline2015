"""
Linear BIC clustering

Usage:
  linear_bic_clustering.py <videoID> <wave> <input_seg> <output_seg> [--penalty_coef=<pc>] [--gap=<g>]
  linear_bic_clustering.py -h | --help
Options:
  --penalty_coef=<pc>   penalty coefficient for BIC (>0.0) [default: 1.8]
  --gap=<g>             maximum gap between 2 speech turns that can be merged (>0.0) [default: 0.8]  
"""

from docopt import docopt
from pyannote.core import *
from pyannote.features.audio.yaafe import YaafeMFCC
from pyannote.parser import MDTMParser
from pyannote.algorithms.clustering.bic import BICModel

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
    # read speech turn segmentation
    annotation = MDTMParser().read(args['<input_seg>'])(uri=args['<videoID>'], modality="speaker")
    # extract descriptor
    extractor = YaafeMFCC(e=True, coefs=12, De=False, DDe=False, D=False, DD=False)
    audio_features = extractor(args['<wave>'])
    # initialize the model
    model = BICModel(covariance_type='diag', penalty_coef=float(args['--penalty_coef']))
    
    seg_to_merged = True
    while seg_to_merged == True:
        seg_to_merged = False

        l_seg = list(annotation.get_timeline())
        cluster = {}
        for seg in l_seg:
            cluster[seg] = list(annotation.get_labels(seg))[0]

        for i in range(len(l_seg)-1):
            if l_seg[i].end - l_seg[i+1].start <= float(args['--gap']) and cluster[l_seg[i]] != cluster[l_seg[i+1]] :
                if model.get_similarity(cluster[l_seg[i]], cluster[l_seg[i+1]], annotation=annotation, feature=audio_features) > 0:
                    # merge segment
                    c1 = cluster[l_seg[i]]
                    c2 = cluster[l_seg[i+1]]
                    del annotation[l_seg[i]]
                    del annotation[l_seg[i+1]]
                    annotation[Segment(start=min(l_seg[i].start, l_seg[i+1].start), end=max(l_seg[i].end, l_seg[i+1].end))] = c1
                    seg_to_merged = True
                    break
    
    # save segmentation
    with open(args['<output_seg>'], 'w') as f:
        MDTMParser().write(annotation, f=f)

                          