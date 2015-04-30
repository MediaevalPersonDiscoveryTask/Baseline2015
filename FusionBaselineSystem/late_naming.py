"""
name speakers by overlaid names and then propagate speakers identity to the best speaking Face in higher a threshold

Usage:
  late_naming.py <video_list> <spk_dia> <st_seg> <face_seg> <mat_speaking_face> <overlaid_names> <shot_seg> <output_label> <output_evidence> [--thr_propagation=<tp>]
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

    evidences = {}
    for videoID in open(args['<video_list>']).read().splitlines():
        print videoID

        # read segmentation file
        sd = MDTMParser().read(args['<spk_dia>']+'/'+videoID+'.mdtm')(uri=videoID, modality = 'speaker')
        st = MDTMParser().read(args['<st_seg>']+'/'+videoID+'.mdtm')(uri=videoID, modality = 'speaker')
        faces = parser_vtseg(args['<face_seg>']+'/'+videoID+'.seg', videoID)
        ON = REPEREParser().read(args['<overlaid_names>'])(uri=videoID, modality = 'written')
        shots = parser_shot_seg(args['<shot_seg>']+'/'+videoID+'.shot', videoID)

        # name speakers
        direct = ConservativeDirectTagger()
        one_to_one = HungarianTagger()
        NamedSpk = direct(ON, one_to_one(ON, sd))
        l_to_remove = []
        for s, t, name in NamedSpk.itertracks(label=True):
            if 'st_' in name: l_to_remove.append([s, t])
        for s, t in l_to_remove: del NamedSpk[s, t]

        # propagate speakers identity to best speakingFace
        dic_trackID_to_st = {}
        dic_st_to_speakingFace = {}
        for s, t, st in st.itertracks(label=True):
            dic_trackID_to_st[t] = st
            dic_st_to_speakingFace[st] = ['', 0.0]

        thr_propagation = float(args['--thr_propagation'])
        for line in open(args['<mat_speaking_face>']+'/'+videoID+'.mat').read().splitlines():
            st, faceID, proba = line.split(' ')
            proba = float(proba)
            if proba >= thr_propagation and proba > dic_st_to_speakingFace[st][1]: dic_st_to_speakingFace[st] = [faceID, proba]

        faceID_to_name = {}
        for s, t, name in NamedSpk.itertracks(label=True):
            st = dic_trackID_to_st[t]
            if dic_st_to_speakingFace[st][0] != '': faceID_to_name[dic_st_to_speakingFace[st][0]] = name

        namedFaces = Annotation(uri=videoID)
        for s, t, faceID in faces.itertracks(label=True):
            if faceID in faceID_to_name: namedFaces[s, t] = faceID_to_name[faceID]

        fout_label = open(args['<output_label>']+'/'+videoID+'.label', 'w')
        # write person visible ans speaking in a shot:
        for sshot, tshot, shot in shots.itertracks(label=True):
            NamedSpkShot = NamedSpk.crop(sshot)
            NamedFaceShot = namedFaces.crop(sshot)
            PersonShot = set(NamedSpkShot.labels()) & set(NamedFaceShot.labels())
            for p in PersonShot:
                conf = 0.0
                for sSpk in NamedSpkShot.label_timeline(p):
                    for sON, tON, name in ON.itertracks(label=True):
                        if name == p:
                            sInter = sON & sSpk
                            if sInter : c = 1.0 + sInter.duration
                            else:
                                sDist = sON ^ sSpk
                                if sDist.duration == 0 : c=1.0
                                else: c = 1/sDist.duration
                            if c > conf: conf = c

                for sFace in NamedFaceShot.label_timeline(p):
                    for sON, tON, name in ON.itertracks(label=True):
                        if name == p:
                            sInter = sON & sFace
                            if sInter : c = 1.0 + sInter.duration
                            else:
                                sDist = sON ^ sFace
                                if sDist.duration == 0 : c=1.0
                                else: c = 1/sDist.duration
                            if c > conf: conf = c
                fout_label.write(videoID+' '+shot+' '+p+' '+str(conf)+'\n')
                evidences.setdefault(p, []).append([conf, videoID, shot])
        fout_label.close()
    
    # select and write evidence
    fout_evidence = open(args['<output_evidence>'], 'w')
    for p in evidences:
        conf, videoID, shot = sorted(evidences[p], reverse=True)[0]
        fout_evidence.write(p+' '+videoID+' '+shot+' image\n')
    fout_evidence.close()



