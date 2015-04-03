"""
speech turn segmentation

Usage:
  speech_turn_segmentation.py <video_name> <path_to_wave> <seg_speech_nonspeech> <output_path> [--penalty_coef=<pc>] [--min_duration=<md>] 
  speech_turn_segmentation.py -h | --help
Options:
  --penalty_coef=<pc>   penalty coefficient for BIC (>0.0) [default: 1.0]
  --min_duration=<md>   minimum duration of a speech turn (>0.0) [default: 1.0]
"""

from docopt import docopt
from pyannote.core import Annotation
from pyannote.parser import MDTMParser
from pyannote.features.audio.yaafe import YaafeMFCC
from pyannote.algorithms.segmentation.bic import BICSegmentation


if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    # read segmentation speech nonspeech
    speech_nonspeech = MDTMParser().read(args['<seg_speech_nonspeech>']+'/'+args['<video_name>']+'.mdtm')(uri=args['<video_name>'], modality="speaker")

    # extract descriptor
    extractor = YaafeMFCC(e=True, coefs=12, De=False, DDe=False, D=False, DD=False)
    audio_features = extractor(args['<path_to_wave>']+'/'+args['<video_name>']+'.wav')

    # segment audio stream
    segmenter = BICSegmentation(penalty_coef=float(args['<penalty_coef>']), min_duration=float(args['--min_duration']))
    speech_turns = segmenter.apply(audio_features, segmentation=speech_nonspeech.label_timeline('speech'))

    # create a new annotation 
    anno = Annotation(uri=args['<video_name>'], modality='speaker')
    # rename speech turn
    nb_st = 1
    for seg1 in speech_turns:
        anno[seg1] = 'st_'+str(nb_st)
        nb_st+=1

    # save the segmentation
    with open(args['<output_path>']+'/'+args['<video_name>']+'.mdtm', 'w') as f:
        MDTMParser().write(anno, f=f, uri=args['<video_name>'], modality='speaker')
