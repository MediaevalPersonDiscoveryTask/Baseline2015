"""
Compute BIC distance between speech turns and save it into a matrix file

Usage:
  compute_BIC_matrix.py <videoID> <audioFile> <linearClustering> <BICMatrix>
  compute_BIC_matrix.py -h | --help
"""

from docopt import docopt
from pyannote.features.audio.yaafe import YaafeMFCC
from mediaeval_util.repere import MESegParser
from pyannote.algorithms.clustering.bic import BICModel
import pickle

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    # read segmentation
    st_seg, confs, timeToFrameID = MESegParser(args['<linearClustering>'], args['<videoID>'])

    print st_seg
    # extract descriptor
    extractor = YaafeMFCC(e=True, coefs=12, De=False, DDe=False, D=False, DD=False)
    audio_features = extractor(args['<audioFile>'])

    # defined model type
    model = BICModel(covariance_type='diag')

    # compute score between speech turns
    m = model.get_track_similarity_matrix(st_seg, audio_features)

    # save matrix
    m.save(args['<BICMatrix>'])
