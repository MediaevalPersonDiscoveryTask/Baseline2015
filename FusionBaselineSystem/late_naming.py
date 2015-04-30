"""
name speakers by overlaid names and then propagate speakers identity to the best speaking Face in higher a threshold

Usage:
  late_naming.py <videoID> <spk_dia> <st_seg> <face_seg> <mat_speaking_face> <overlaid_names> <shot_seg> <output_file> [--thr_propagation=<tp>]
  late_naming.py -h | --help
Options:
  --thr_propagation=<tp>  minimum score to propagate speaker identity to facetrack [default: 0.5]
"""

from docopt import docopt
from pyannote.parser import MDTMParser, REPEREParser
from pyannote.algorithms.tagging import HungarianTagger, ConservativeDirectTagger
from pyannote.core import Annotation, Segment
from mediaeval_util.repere import parser_vtseg, parser_shot_seg

if __name__ == "__main__":   
    # read arguments
    args = docopt(__doc__)

    # read segmentation file
    sd = MDTMParser().read(args['<spk_dia>'])(uri=args['<videoID>'], modality = 'speaker')
    st = MDTMParser().read(args['<st_seg>'])(uri=args['<videoID>'], modality = 'speaker')
    faces = parser_vtseg(args['<face_seg>'], args['<videoID>'])
    ON = REPEREParser().read(args['<overlaid_names>'])(uri=args['<videoID>'], modality = 'written')
    shots = parser_shot_seg(args['<shot_seg>'], args['<videoID>'])

    # name speakers
    direct = ConservativeDirectTagger()
    one_to_one = HungarianTagger()
    NamedSpk = direct(ON, one_to_one(ON, sd))
    l_to_remove = []
    for s, t, name in NamedSpk.itertracks(label=True):
        if 'st_' in name:
            l_to_remove.append([s, t])
    for s, t in l_to_remove:
        del NamedSpk[s, t]

    # propagate speakers identity to best speakingFace
    dic_trackID_to_st = {}
    dic_st_to_speakingFace = {}
    for s, t, st in st.itertracks(label=True):
        dic_trackID_to_st[t] = st
        dic_st_to_speakingFace[st] = ['', 0.0]

    thr_propagation = float(args['--thr_propagation'])
    for line in open(args['<mat_speaking_face>']).read().splitlines():
        st, faceID, proba = line.split(' ')
        proba = float(proba)
        if proba >= thr_propagation and proba > dic_st_to_speakingFace[st][1]:
            dic_st_to_speakingFace[st] = [faceID, proba]

    faceID_to_name = {}
    for s, t, name in NamedSpk.itertracks(label=True):
        st = dic_trackID_to_st[t]
        if dic_st_to_speakingFace[st][0] != '':
            faceID_to_name[dic_st_to_speakingFace[st][0]] = name

    namedFaces = Annotation(uri=args['<videoID>'])
    for s, t, faceID in faces.itertracks(label=True):
        if faceID in faceID_to_name:
            namedFaces[s, t] = faceID_to_name[faceID]

    # write person visible ans speaking in a shot:
    for s, t, shot in shots.itertracks(label=True):
        NamedSpkShot = NamedSpk.crop(s)
        NamedFaceShot = namedFaces.crop(s)
        PersonShot = set(NamedSpkShot.labels()) & set(NamedFaceShot.labels())
        print shot
        print NamedSpkShot
        print NamedFaceShot
        print PersonShot
        for p in PersonShot:
            print p
            conf = 0.0
            for sSpk in NamedSpkShot.label_timeline(p):
                print '  ', sSpk
                for sON, tON, ON in NamedSpk.itertracks(label=True):
                    if ON == p:
                        print '     ', ON, sON
                        sInter = sON & sSpk
                        if sInter:
                            c = sInter.duration
                        else:
                            sDist = sON ^ sSpk
                            print '     ', sDist.duration
                            

                        '''
                        sInter = sON & sSpk
                        print sInter,
                        if sInter: c = sInter.duration
                        else:
                            sDist = sON ^ sSpk
                            print sDist.duration
                            if sDist.duration > 0.0:
                                c = 1/sDist.duration
                        #c = sON & sSpk and sInter.duration or 1/sDist.duration
                        if c > conf: conf = c
                        print

                        '''
            '''
            for sFace in NamedFaceShot.label_timeline(p):
                for sON, tON, ON in NamedSpk.itertracks(label=True):
                    if ON == p:
                        c = sON & sFace and (sON & sFace).duration or 1/(sON ^ sFace).duration
                        if c > conf: conf = c
            '''
            print shot, p, conf






