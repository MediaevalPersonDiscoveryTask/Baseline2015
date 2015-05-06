"""
Linear BIC clustering

Usage:
  linear_bic_clustering.py <videoID> <audioFile> <speechTurnSegmentation> <linearClustering> [--penalty_coef=<pc>] [--gap=<g>]
  linear_bic_clustering.py -h | --help
Options:
  --penalty_coef=<pc>   penalty coefficient for BIC (>0.0) [default: 2.4]
  --gap=<g>             maximum gap between 2 speech turns that can be merged (>0.0) [default: 0.8]  
"""

from docopt import docopt
from pyannote.core import *
from pyannote.features.audio.yaafe import YaafeMFCC
from pyannote.algorithms.clustering.bic import BICModel
from mediaeval_util.repere import MESegParser, MESegWriter

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
    
    # read speech turn segmentation
    st_seg, confs, timeToFrameID = MESegParser(args['<speechTurnSegmentation>'], args['<videoID>'])
    
    # extract descriptor
    extractor = YaafeMFCC(e=True, coefs=12, De=False, DDe=False, D=False, DD=False)
    audio_features = extractor(args['<audioFile>'])
    
    # initialize the model
    model = BICModel(covariance_type='diag', penalty_coef=float(args['--penalty_coef']))
    
    # linear clustering    
    seg_to_merged = True
    while seg_to_merged == True:
        seg_to_merged = False
        l_seg = []
        for s, t, l in st_seg.itertracks(label=True): l_seg.append([s, t, l])
        for i in range(len(l_seg)-1):
            if l_seg[i][0].end - l_seg[i+1][0].start <= float(args['--gap']) :
                if model.get_similarity(l_seg[i][2], l_seg[i+1][2], annotation=st_seg, feature=audio_features) > 0:
                    del st_seg[l_seg[i][0], l_seg[i][1]]
                    del st_seg[l_seg[i+1][0], l_seg[i+1][1]]
                    st_seg[Segment(start=l_seg[i][0].start, end=l_seg[i+1][0].end), l_seg[i][1]] = l_seg[i][2]
                    seg_to_merged = True
                    break
    
    # save segmentation
    MESegWriter(st_seg, {}, args['<linearClustering>'], args['<videoID>'], {})

                          