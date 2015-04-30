"""
learn segmenter model for speech nonspeech segmentation

Usage:
  learn_model_speech_nonspeech.py <wavePath> <videoList> <speakerSegmentationReferencePath> <segment.uem> <modelSpeechNonSpeech>
  learn_model_speech_nonspeech.py -h | --help
"""

from docopt import docopt
from mediaeval_util.repere import parser_atseg
from pyannote.parser import UEMParser
from pyannote.features.audio.yaafe import YaafeCompound, YaafeZCR, YaafeMFCC
from pyannote.algorithms.segmentation.hmm import GMMSegmentation 
import pickle

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
    # read ref
    # segment manually annotated in the reference
    uems = UEMParser().read(args['<segment.uem>'])
    # extractor Yaafe
    extractor = YaafeCompound([YaafeZCR(), YaafeMFCC(e=False, De=False, DDe=False, D=True, DD = True)])

    l_features = []
    ref_speech_nonspeech = []

    for videoID in open(args['<videoList>']).read().splitlines():
        # extract features
        features = extractor(args['<wavePath>']+'/'+videoID+'.wav')
        l_features.append(features)
 
        ref = parser_atseg(args['<speakerSegmentationReferencePath>']+'/'+videoID+'.atseg', videoID)

        # rename all segment with speech
        mapping = {source: 'speech' for source in ref.labels()}
        ref = ref.translate(mapping)
        # complete gap between segment with nonspeech
        for segment in ref.get_timeline().gaps():
            ref[segment] = 'non_speech'
        # used only segment in uem part
        uem = uems(uri=videoID)
        ref_speech_nonspeech.append(ref.crop(uem, mode='intersection'))

    # train model
    segmenter = GMMSegmentation(n_components=256, lbg=True)
    segmenter.fit(l_features, ref_speech_nonspeech)
 
    # save segmenter model
    pickle.dump(segmenter, open(args['<modelSpeechNonSpeech>'], "wb" ) )
