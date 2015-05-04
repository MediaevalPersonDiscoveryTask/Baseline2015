
import numpy as np
from pandas import read_table
from sklearn.isotonic import IsotonicRegression
from pyannote.core import Annotation, Segment
from pyannote.parser import MDTMParser
from pyannote.algorithms.tagging import ArgMaxDirectTagger
from pyannote.core import Scores
from munkres import Munkres

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

def MESegWriter(anno, scores, f, video):
    fout = open(f, 'w')
    for s, t, l in anno.itertracks(label=True):
        score = scores and str(scores[s, t, l]) or 'na'
        fout.write(video+' '+str(s.start)+' '+str(s.end)+' na na '+str(t)+' '+l+' '+score+'\n')
    fout.close()

def MESegParser(f, video):
    anno = Annotation(uri=video)
    scores = Scores()
    for line in open(f):
        v, startTime, endTime, startFrame, endFrame, t, l, conf = line[:-1].split(' ')
        s = Segment(start=float(startTime), end=float(endTime))
        anno[s, int(t)] = l
        if conf != 'na': scores[s, int(t), l] = float(conf)
    return anno, scores

def ShotSegParser(f, video):
    anno = Annotation(uri=video)
    nb_track=0
    for line in open(f):
        video, shotName, startTime, endTime, startFrame, endFrame = line[:-1].split(' ')
        segment = Segment(start=float(startTime), end=float(endTime))
        anno[segment, nb_track] = shotName
        nb_track+=1
    return anno

def parser_vtseg(f, video):
    anno = Annotation(uri=video)
    nb_track=0
    for line in open(f):
        FaceID, startTime, endTime, startFrame, endFrame = line[:-1].split(' ')
        segment = Segment(start=float(startTime), end=float(endTime))
        anno[segment, nb_track] = FaceID
        nb_track+=1
    return anno

def parser_atseg(f, video):
    anno = Annotation(uri=video)
    nb_track=0    
    for line in open(f):
        video, startTime, endTime, name = line[:-1].split(' ')
        segment = Segment(start=float(startTime), end=float(endTime))
        anno[segment, nb_track] = name
        nb_track+=1
    return anno

def cooc(seg1, seg2):
    start = max(seg1[0], seg2[0])
    end  = min(seg1[1], seg2[1]) 
    if start > end:
        return 0.0    
    return end-start

def align_st_ref(seg_st_path, ref_path, videoID):
    st_vs_ref = {}
    ref = parser_atseg(ref_path+'/'+videoID+'.atseg', videoID)
    seg_st = MDTMParser().read(seg_st_path+'/'+videoID+'.mdtm')(uri=videoID, modality="speaker")

    direct = ArgMaxDirectTagger()
    named_st = direct(ref, seg_st)

    track_to_st = {}
    for s, t, l in seg_st.itertracks(label=True):
        track_to_st[t] = l

    for s, t, name in named_st.itertracks(label=True):
        if 'st_' not in name:
            st_vs_ref[track_to_st[t]] = name
    return st_vs_ref

def read_ref_facetrack_position(f, tempo_margin):
    ref = {}
    for line in open(f).read().splitlines():
        startFrame, endFrame, frameAnnotated, name, position = line.split(' ')  
        if startFrame != '' and endFrame != '' and frameAnnotated != '':
            startFrame, endFrame, frameAnnotated = int(startFrame), int(endFrame), int(frameAnnotated)
            l_x = []
            l_y = []
            for point in position.split(';'):
                x, y = map(int, point.split(':'))
                l_x.append(x)
                l_y.append(y)
            x = (max(l_x) + min(l_x))/2
            y = (max(l_y) + min(l_y))/2
            ref.setdefault(frameAnnotated, {})
            ref[frameAnnotated][name] = [x, y, max(l_x)-min(l_x), max(startFrame,frameAnnotated-tempo_margin), min(endFrame,frameAnnotated+tempo_margin)]
    return ref

def align_facetrack_ref(ref_f, facetracks):
    d_align = {}
    for frameID in ref_f:
        l_faceID = set([])
        for name in ref_f[frameID]:
            x, y, size, start, end = ref_f[frameID][name]
            for i in range(start, end+1):
                if i in facetracks:
                    for faceID in facetracks[i]:
                        l_faceID.add(faceID)
        mat_size = max(len(ref_f[frameID]),len(l_faceID))

        if l_faceID != set([]):
            matrix = (mat_size,mat_size)
            matrix = np.zeros(matrix)
            matrix.fill(300)
            row = 0
            for name in sorted(ref_f[frameID]):
                col = 0
                x, y, size, start, end = ref_f[frameID][name]
                for faceID in sorted(l_faceID): 
                    tot_score = 0
                    for i in range(start, end+1):
                        score = size*2
                        if i in facetracks:
                            if faceID in facetracks[i]:
                                xmin, ymin, xmax, ymax = facetracks[i][faceID]
                                score = (abs(x-(xmin+xmax)/2) + abs(y-(ymin+ymax)/2))/2
                        tot_score += score
                    if start<=end:
                        matrix[row][col] = tot_score / (end-start+1)
                    col+=1
                row+=1

            m = Munkres()
            matrix_dist = matrix.copy()
            indexes = m.compute(matrix) 
            for row, col in indexes:
                if row < len(ref_f[frameID].keys()) and col < len(l_faceID):
                    name = sorted(ref_f[frameID].keys())[row]
                    faceID = sorted(l_faceID)[col]
                    x, y, size, start, end = ref_f[frameID][name]
                    if matrix_dist[row][col] < size:
                        d_align.setdefault(faceID, []).append(name)

    d_align_final = {}
    for faceID, l_name in d_align.items():
        if len(l_name) == 1:
            d_align_final[faceID] = l_name[0]
        else:
            d = {}
            for name in l_name:
                d.setdefault(name, 0)
                d[name] += 1
            best_occ = 0
            best_name = ''
            for name, nb_occ in d.items():
                if nb_occ>best_occ:
                    best_occ = nb_occ
                    best_name = name
            d_align_final[faceID] = name
    return d_align_final


class IDXHack(object):
    """

    Usage
    =====
    >>> from mediaeval_util.repere import IDXHack
    >>> frame2time = IDXHack(args['--idx'])
    >>> trueTime = frame2time(opencvFrame, opencvTime)

    """

    def __init__(self, idx=None):
        super(IDXHack, self).__init__()
        self.idx = idx

        if self.idx:

            # load .idx file using pandas
            df = read_table(
                self.idx, sep='\s+',
                names=['frame_number', 'frame_type', 'bytes', 'seconds']
            )
            x = np.array(df['frame_number'], dtype=np.float)
            y = np.array(df['seconds'], dtype=np.float)

            # train isotonic regression
            self.ir = IsotonicRegression(y_min=np.min(y), y_max=np.max(y))
            self.ir.fit(x, y)

            # frame number support
            self.xmin = np.min(x)
            self.xmax = np.max(x)

    def __call__(self, opencvFrame, opencvTime):

        if self.idx is None:
            return opencvTime

        return self.ir.transform([min(self.xmax,
                                      max(self.xmin, opencvFrame)
                                      )])[0]
