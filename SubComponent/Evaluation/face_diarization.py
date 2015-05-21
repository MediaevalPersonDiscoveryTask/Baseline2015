import sys
from mediaeval_util.repere import MESegParser, read_ref_facetrack_position, align_facetrack_ref

if __name__ == '__main__':


    diarization = sys.argv[1]
    facetrack = sys.argv[2]
    facetrackPosition = sys.argv[3]
    video_list = sys.argv[4]
    facePositionReferencePath = sys.argv[5]

    l_video = []
    for videoID in open(video_list).read().splitlines():
        l_video.append(videoID)


    for thr in ['0.2', '0.225', '0.25', '0.275', '0.3', '0.325', '0.35', '0.375']:
        print thr

        PtotCorpus = 0.0
        RtotCorpus = 0.0
        FtotCorpus = 0.0

        for videoID in l_video:
            facetracks = {}
            for line in open(facetrackPosition+'/'+videoID+'.facetrack').read().splitlines():
                frameID, faceID, xmin, ymin, w, h = map(int, line.split(' ')) 
                facetracks.setdefault(frameID, {})
                facetracks[frameID][faceID] = xmin, ymin, xmin+w, ymin+h
            ref_f = read_ref_facetrack_position(facePositionReferencePath, videoID, 0)
            facetrack_vs_ref = align_facetrack_ref(ref_f, facetracks)


            facetrack_seg, confs, timeToFrameID = MESegParser(facetrack+'/'+videoID+'.MESeg', videoID)
            face_dia, confs, timeToFrameID = MESegParser(diarization+'/'+videoID+'_'+thr+'.MESeg', videoID)

            clusters = {}
            persons = {}
            Ptot = 0.0
            Rtot = 0.0
            Ftot = 0.0

            for s, t, l in facetrack_seg.itertracks(label=True):
                if int(l) in facetrack_vs_ref:
                    clusters.setdefault(face_dia[s, t], []).append(facetrack_vs_ref[int(l)])
                    persons.setdefault(facetrack_vs_ref[int(l)], []).append(face_dia[s, t])

            for c in clusters:
                best_name = ''
                best_nb_name = 0
                for e in set(clusters[c]):
                    nb_name = clusters[c].count(e)
                    if nb_name>best_nb_name:
                        best_nb_name = nb_name
                        best_name = e
                P = float(clusters[c].count(best_name)) / float(len(clusters[c]))
                R = float(clusters[c].count(best_name)) / float(len(persons[best_name]))
                Ptot+=P
                Rtot+=R
                if P+R > 0.0:  Ftot+=(2*P*R)/(P+R)  
            Ptot /= float(len(clusters))
            Rtot /= float(len(clusters))            
            Ftot /= float(len(clusters))

            PtotCorpus += Ptot
            RtotCorpus += Rtot
            FtotCorpus += Ftot

        PtotCorpus /= float(len(l_video))
        RtotCorpus /= float(len(l_video))
        FtotCorpus /= float(len(l_video))

        print thr, PtotCorpus, RtotCorpus, FtotCorpus
