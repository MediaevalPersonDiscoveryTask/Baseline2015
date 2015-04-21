"""
learn segmenter model for speech nonspeech segmentation

Usage:
  learn_model_speech_nonspeech.py <source_path> <dataPath.lst> <video_list> <reference_path> <uem_file> <model_output>
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
    uems = UEMParser().read(args['<uem_file>'])
    # extractor Yaafe
    extractor = YaafeCompound([YaafeZCR(), YaafeMFCC(e=False, De=False, DDe=False, D=True, DD = True)])

    audio_features = []
    ref_speech_nonspeech = []

    wavePath = {}
    for path in open(args['<dataPath.lst>']).read().splitlines():
        video, wave_file, video_avi_file, video_mpeg_file, trs_file, xgtf_file, idx_file = path.split(' ')
        wavePath[video] = args['<source_path>']+'/'+wave_file

    for video in open(args['<video_list>']).read().splitlines():
        print video
        # extract features
        audio_features = extractor(wavePath[video])
        audio_features.append(audio_features)
 
        ref = parser_atseg(reference_path+'/'+video+'.atseg', video)

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
    segmenter.fit(audio_features, ref_speech_nonspeech)
 
    # save segmenter model
    pickle.dump(segmenter, open(args['<model_output>'], "wb" ) )
