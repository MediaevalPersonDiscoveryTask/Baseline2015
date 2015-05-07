"""
Evaluation of the speech turn segmentation

Usage:
  eval_speech_turn_segmentation.py <seg_speech_turn_path> <ref> <video_list> <uem>
  eval_speech_turn_segmentation.py -h | --help
"""

from docopt import docopt
from mediaeval_util.repere import MESegParser
from pyannote.parser import MDTMParser
from pyannote.parser import UEMParser
from pyannote.metrics.segmentation import SegmentationPurity, SegmentationCoverage
import pickle, os
import itertools
import numpy as np

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

def fillShortGaps(annotation, duration=1.):
    fillers = annotation.empty()
    for ((s, t, l), (ss, tt, ll)) in pairwise(annotation.itertracks(label=True)):
        if l != ll:
            continue
        gap = s ^ ss
        if gap.duration > duration:
            continue
        fillers[gap] = l
    return annotation.copy().update(fillers).smooth()

if __name__ == '__main__':
    args = docopt(__doc__)

    gap = '0.5'

    parser_uem = UEMParser().read(args['<uem>'])

    pty = SegmentationPurity(collar=0.250)
    cvg = SegmentationCoverage(collar=0.250)

    l_seg = []
    for videoID in open(args['<video_list>']).read().splitlines():
        uem = parser_uem(uri=videoID)

        ref, confs, timeToFrameID = MESegParser(args['<ref>'], videoID)
        ref = fillShortGaps(ref, duration=1.)

        hyp, confs, timeToFrameID = MESegParser(args['<seg_speech_turn_path>']+'/'+videoID+'.MESeg', videoID)

        #hyp = MDTMParser().read(args['<seg_speech_turn_path>']+'/'+videoID+'.mdtm')(uri=videoID, modality="speaker")

        for seg in hyp.get_timeline():
            l_seg.append(seg.duration)                    

        pty(ref, hyp, uem=uem)
        cvg(ref, hyp, uem=uem)

    print round(abs(pty)*100,2), round(abs(cvg)*100,2), 
    print round(np.mean(l_seg),2), round(np.std(l_seg),2), np.percentile(l_seg, 25), np.percentile(l_seg, 50), np.percentile(l_seg, 75) 
