"""
name speakers by overlaid names and then propagate speakers identity to the best speaking Face in higher a threshold

Usage:
  late_naming.py <video_list> <spk_dia> <face_seg> <mat_speaking_face> <writtenNames> <shot_seg> <output_label> <output_evidence> [--thr_propagation=<tp>]
  late_naming.py -h | --help
Options:
  --thr_propagation=<tp>  minimum score to propagate speaker identity to facetrack [default: 0.5]
"""

from docopt import docopt
from mediaeval_util.repere import MESegParser, ShotSegParser
from pyannote.algorithms.tagging import HungarianTagger, ConservativeDirectTagger
from pyannote.core import Annotation, Segment

if __name__ == "__main__":   
    # read arguments
    args = docopt(__doc__)

    evidences = {}
    for videoID in open(args['<video_list>']).read().splitlines():
        shots = ShotSegParser(args['<shot_seg>'], videoID)
        ON, confs, timeToFrameID = MESegParser(args['<writtenNames>'], videoID)
        for sON, tON, name in ON.itertracks(label=True):
            name = name.lower().replace('-', '_').replace('.', '_')
            for sshot, tshot, shot in shots.itertracks(label=True):
                if sON & sshot:
                    d = (sON & sshot).duration
                    evidences.setdefault(name, [d, videoID, shot])
                    if evidences[name] < d: evidences[name] = [d, videoID, shot]

    fout_label = open(args['<output_label>'], 'w')
    l_p_in_label = set([])
    for videoID in open(args['<video_list>']).read().splitlines():
        print videoID
        # read segmentation file
        sd, confs, timeToFrameID = MESegParser(args['<spk_dia>']+'/'+videoID+'.MESeg', videoID)
        faces, confs, timeToFrameID = MESegParser(args['<face_seg>']+'/'+videoID+'.MESeg', videoID)
        ON, confs, timeToFrameID = MESegParser(args['<writtenNames>'], videoID)
        shots = ShotSegParser(args['<shot_seg>'], videoID)

        # name speakers
        direct = ConservativeDirectTagger()
        one_to_one = HungarianTagger()
        NamedSpk = direct(ON, one_to_one(ON, sd))
        NamedSpk.get_labels
        l_to_remove = []

        for s, t, name in NamedSpk.itertracks(label=True):
            if 'c_' in name: l_to_remove.append([s, t])
        for s, t in l_to_remove: 
            del NamedSpk[s, t]

        
        # propagate speakers identity to best speakingFace
        thr_propagation = float(args['--thr_propagation'])

        dic_trackID_st_to_speakingFace = {}
        for s, t, st in sd.itertracks(label=True):
            dic_trackID_st_to_speakingFace[t] = ['', thr_propagation]
        
        for line in open(args['<mat_speaking_face>']+'/'+videoID+'.mat').read().splitlines():
            TrackID_st, TrackID_Face, proba = line.split(' ')
            if float(proba) > dic_trackID_st_to_speakingFace[int(TrackID_st)][1]: 
                dic_trackID_st_to_speakingFace[int(TrackID_st)] = [int(TrackID_Face), float(proba)]

        trackID_face_to_name = {}
        for s, t, name in NamedSpk.itertracks(label=True):
            if dic_trackID_st_to_speakingFace[t][0] != '': 
                trackID_face_to_name[dic_trackID_st_to_speakingFace[t][0]] = name

        namedFaces = Annotation(uri=videoID)
        for s, t, faceID in faces.itertracks(label=True):
            if t in faceID_to_name: 
                namedFaces[s, t] = trackID_face_to_name[t]

        # write person visible and speaking in a shot:
        for sshot, tshot, shot in shots.itertracks(label=True):
            NamedSpkShot = NamedSpk.crop(sshot)
            NamedFaceShot = namedFaces.crop(sshot)
            PersonShot = set(NamedSpkShot.labels()) & set(NamedFaceShot.labels())

            for p in (PersonShot & set(evidences.keys())):
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

                l_p_in_label.add(p)
                fout_label.write(videoID+' '+shot+' '+p+' '+str(conf)+'\n')
    fout_label.close()
    
    # select and write evidence
    fout_evidence = open(args['<output_evidence>'], 'w')
    for name in l_p_in_label:
        conf, videoID, shot = evidences[name]
        fout_evidence.write(name+' '+videoID+' '+shot+' image\n')
    fout_evidence.close()


