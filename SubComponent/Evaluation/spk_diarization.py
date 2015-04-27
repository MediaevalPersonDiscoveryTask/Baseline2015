import sys
from pyannote.parser import MDTMParser, UEMParser
from pyannote.metrics.diarization import DiarizationErrorRate
from mediaeval_util.repere import parser_atseg

if __name__ == '__main__':


    st_seg_path = sys.argv[1]
    l_video = sys.argv[2]
    ref_path = sys.argv[3]
        
    parser_uem = UEMParser().read(sys.argv[4])


    #l_thr = [0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38]
    l_thr = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    
    for pc in [2.0, 2.2, 2.4, 2.6]:
        for thr in sorted(l_thr):
            DER = DiarizationErrorRate()
            for videoID in open(l_video).read().splitlines():

                hyp = MDTMParser().read(st_seg_path+'/'+videoID+'_'+str(pc)+'_'+str(thr)+'.mdtm')(uri=videoID, modality="speaker")
                ref = parser_atseg(ref_path+'/'+videoID+'.atseg', videoID)
                uem = parser_uem(uri=videoID)

                hyp = hyp.crop(uem, mode='intersection')

                DER(ref, hyp, uem=uem)

            print pc, thr, DER



