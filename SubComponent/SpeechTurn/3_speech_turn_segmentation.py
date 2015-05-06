"""
speech turn segmentation

Usage:
  speech_turn_segmentation.py <videoID> <audioFile> <speechNonSpeechSegmentation> <speechTurnSegmentation> [--penalty_coef=<pc>] [--min_duration=<md>] 
  speech_turn_segmentation.py -h | --help
Options:
  --penalty_coef=<pc>   penalty coefficient for BIC (>0.0) [default: 1.0]
  --min_duration=<md>   minimum duration of a speech turn (>0.0) [default: 1.0]
"""

from docopt import docopt
from pyannote.core import Annotation
from mediaeval_util.repere import MESegParser, MESegWriter
from pyannote.features.audio.yaafe import YaafeMFCC
from pyannote.algorithms.segmentation.bic import BICSegmentation


if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    # read segmentation speech nonspeech
    speech_nonspeech, confs, timeToFrameID = MESegParser(args['<speechNonSpeechSegmentation>'], args['<videoID>'])

    # extract descriptor
    extractor = YaafeMFCC(e=True, coefs=12, De=False, DDe=False, D=False, DD=False)
    audio_features = extractor(args['<audioFile>'])

    # segment audio stream
    segmenter = BICSegmentation(penalty_coef=float(args['--penalty_coef']), min_duration=float(args['--min_duration']))
    speech_turns = segmenter.apply(audio_features, segmentation=speech_nonspeech.subset(['speech']).get_timeline())

    # create a new annotation 
    anno = Annotation(uri=args['<videoID>'], modality='speaker')
    
    # rename speech turn
    nb_st = 0
    for seg1 in speech_turns:
        anno[seg1, nb_st] = 'st_'+str(nb_st)
        nb_st+=1

    # save the segmentation
    MESegWriter(anno, {}, args['<speechTurnSegmentation>'], args['<videoID>'], {})
