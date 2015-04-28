import sys
from pyannote.parser import MDTMParser, UEMParser
from pyannote.metrics.diarization import DiarizationErrorRate, DiarizationPurity, DiarizationCoverage
from mediaeval_util.repere import parser_atseg

if __name__ == '__main__':
    st_seg_path = sys.argv[1]
    l_video = sys.argv[2]
    ref_path = sys.argv[3]
        
    parser_uem = UEMParser().read(sys.argv[4])

    DER = DiarizationErrorRate()

    Dpur = DiarizationPurity()
    Dcov = DiarizationCoverage()

    for videoID in open(l_video).read().splitlines():
        hyp = MDTMParser().read(st_seg_path+'/'+videoID+'.mdtm')(uri=videoID, modality="speaker")
        ref = parser_atseg(ref_path+'/'+videoID+'.atseg', videoID)
        uem = parser_uem(uri=videoID)
        hyp = hyp.crop(uem, mode='intersection')
        DER(ref, hyp, uem=uem)
        Dpur(ref, hyp, uem=uem)
        Dcov(ref, hyp, uem=uem)

    print 'DER:', round(abs(DER)*100,3), '    purity:', round(abs(Dpur)*100,3), '    coverage:', round(abs(Dcov)*100,3)



