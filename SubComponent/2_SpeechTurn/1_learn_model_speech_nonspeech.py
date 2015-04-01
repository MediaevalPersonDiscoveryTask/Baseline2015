"""
learn segmenter model for speech nonspeech segmentation

Usage:
  learn_model_speech_nonspeech.py <path_to_wave> <video_list> <reference> <model_output> <uem_file>
  learn_model_speech_nonspeech.py -h | --help
"""

from docopt import docopt
from pyannote.parser import MDTMParser
from pyannote.parser import UEMParser
from pyannote.features.audio.yaafe import YaafeCompound, YaafeZCR, YaafeMFCC
from pyannote.algorithms.segmentation.hmm import GMMSegmentation 
import pickle

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
    # read ref
    refs = MDTMParser().read(args['<reference>'])
    # segment manually annotated in the reference
    uems = UEMParser().read(args['<uem_file>'])
    # extractor Yaafe
    extractor = YaafeCompound([YaafeZCR(), YaafeMFCC(e=False, De=False, DDe=False, D=True, DD = True)])

    audio_features = []
    ref_speech_nonspeech = []
    for video in open(args['<video_list>']).read().splitlines():
        # extract features
        audio_features = extractor(args['<path_to_wave>']+'/'+video+'.wav')
        audio_features.append(audio_features)

        ref = refs(uri=video, modality="speaker")
        # rename all segment with speech
        mapping = {source: 'speech' for source in ref.labels()}
        ref = ref.translate(mapping)
        # complete gap between segment with nonspeech
        for segment in ref.get_timeline().gaps():
            ref[segment] = 'non_speech'
        # used only segment in uem part
        uem = uems(uri=video)
        ref_speech_nonspeech.append(ref.crop(uem, mode='intersection'))

    # train model
    segmenter = GMMSegmentation(n_components=64, lbg=True)
    segmenter.fit(audio_features, ref_speech_nonspeech)
 
    # save segmenter model
    pickle.dump(segmenter, open(args['<model_output>'], "wb" ) )
