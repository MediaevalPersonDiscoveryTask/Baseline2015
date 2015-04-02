"""
Compute BIC distance between speech turns and save it into a matrix file

Usage:
  compute_svs_matrix.py <video_name> <wave> <segmentation> <output_matrix_file>
  compute_svs_matrix.py -h | --help
"""

from docopt import docopt
from pyannote.features.audio.yaafe import YaafeMFCC
from pyannote.parser import MDTMParser
from pyannote.algorithms.clustering.bic import BICModel

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    # read segmentation
    seg_speech_turn = MDTMParser().read(args['<segmentation>'])(uri=args['<video_name>'], modality="speaker")

    # extract descriptor
    extractor = YaafeMFCC(e=True, coefs=12, De=False, DDe=False, D=False, DD=False)
    audio_features = extractor(args['<wave>'])

    # defined model type
    model = BICModel(covariance_type='diag')
    labelMatrix = model.get_track_similarity_matrix(seg_speech_turn, audio_features)

    # save matrix
    fout = open(args['<output_matrix_file>'], 'w')
    for s1, t1 in labelMatrix.get_rows():
        for s2, t2 in labelMatrix.get_columns(): 
            if t1 < t2:       
                n1 = list(seg_speech_turn.get_labels(s1))[0]
                n2 = list(seg_speech_turn.get_labels(s2))[0]
                fout.write(str(n1)+' '+str(n2)+' '+str(labelMatrix[(s1,t1), (s2,t2)])+'\n')
    fout.close()                            