"""
Speech nonspeech segmentation

Usage:
  speech_nonspeech_segmentation.py <video_name> <path_to_wave> <segmenter_model> <output_path> [--min_dur_speech=<mds>] [--min_dur_non_speech=<mdns>] 
  speech_nonspeech_segmentation.py -h | --help
Options:
  --min_dur_speech=<mds>       minimum duration of a speech segment (>0) [default: 1.0]
  --min_dur_non_speech=<mdns>  minimum duration of a nonspeech segment (>0) [default: 0.8]
"""

from docopt import docopt
from pyannote.features.audio.yaafe import YaafeCompound, YaafeZCR, YaafeMFCC
from pyannote.parser import MDTMParser
import pickle

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    # load model
    segmenter = pickle.load(open(args['<segmenter_model>'], "rb" ) )

    # Extract descriptor
    extractor = YaafeCompound([YaafeZCR(), YaafeMFCC(e=False, De=False, DDe=False, D=True, DD = True)])    
    audio_features = extractor(args['<path_to_wave>']+'/'+args['<video_name>']+'.wav')

    # segment audio signal and write in output_path
    with open(args['<output_path>']+'/'+args['<video_name>']+'.mdtm', 'w') as f:
        MDTMParser().write(segmenter.predict(audio_features, 
                                             min_duration={'speech':float(args['--min_dur_speech']), 
                                                           'non_speech':float(args['--min_dur_non_speech'])
                                                          }
                                            ), 
                           f=f, 
                           uri=video, 
                           modality='speaker')
