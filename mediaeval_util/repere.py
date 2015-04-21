
import numpy as np
from pandas import read_table
from sklearn.isotonic import IsotonicRegression
from pyannote.core import Annotation, Segment

def parser_vtseg(f, video):
    anno = Annotation(uri=video)
    for line in open(f):
        video, startTime, endTime, startFrame, endFrame, name = line[:-1].split(' ')
        segment = Segment(start=float(startTime), end=float(endTime))
        anno[segment] = name
    return anno

def parser_atseg(f, video):
    anno = Annotation(uri=video)
    for line in open(f):
        video, startTime, endTime, name = line[:-1].split(' ')
        segment = Segment(start=float(startTime), end=float(endTime))
        anno[segment] = name
    return anno


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
