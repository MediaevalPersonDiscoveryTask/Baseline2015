"""
Extract descriptor to find speaking face

Usage:
  extract_desc_speaking_face.py <videoFile> <flandmark> <SpeakingFaceDescriptor>
  extract_desc_speaking_face.py -h | --help
"""

from docopt import docopt
import numpy as np
import cv2, cv, math
from mediaeval_util.imageTools import OpticalFlow, calcul_hist
from mediaeval_util.repere import IDXHack

lk_params = dict( winSize  = (20, 20), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict( maxCorners = 100, qualityLevel = 0.01, minDistance = 4, blockSize = 3)  

def norm_roi(ROIymin, ROIymax, ROIxmin, ROIxmax, w, h):
    ROIymin = max(0, ROIymin)
    ROIymin = min(h, ROIymin)
    ROIymax = max(0, ROIymax)
    ROIymax = min(h, ROIymax)
    ROIxmin = max(0, ROIxmin)
    ROIxmin = min(w, ROIxmin)
    ROIxmax = max(0, ROIxmax)
    ROIxmax = min(w, ROIxmax)        

    if ROIymin==ROIymax:
        if ROIymin==0:
            ROIymax+=1
        elif ROIymax==h:
            ROIymin-=1
        else:
            ROIymax+=1

    if ROIxmin==ROIxmax:
        if ROIxmin==0:
            ROIxmax+=1
        elif ROIxmax==w:
            ROIxmin-=1
        else:
            ROIxmax+=1

    return ROIymin, ROIymax, ROIxmin, ROIxmax

def read_flandmark_file(f_flandmark):
    flandmark = {}
    for line in open(f_flandmark).read().splitlines():
        l = line    .split(' ')
        frame = int(l[0])
        head = int(l[1])
        x = int(l[2])
        y = int(l[3])
        w = int(l[4]) - x
        h = int(l[5]) - y
        mx1 = int(round(float(l[12]), 0))
        my1 = int(round(float(l[13]), 0))
        mx2 = int(round(float(l[14]), 0))
        my2 = int(round(float(l[15]), 0))
        if mx1 != -1:
            flandmark.setdefault(frame, {})
            flandmark[frame][head] = [x, y, w, h, mx1, my1, mx2, my2]
    return flandmark

if __name__ == "__main__": 
    # read arguments   
    args = docopt(__doc__)
    fout = open(args['<SpeakingFaceDescriptor>'], 'w')
    # read flandmark
    flandmark = read_flandmark_file(args['<flandmark>'])
    # open the video
    capture = cv2.VideoCapture(args['<videoFile>'])

    ret, frame_current = capture.read()
    frame_previous1 = frame_current.copy()
    video_w = capture.get(cv.CV_CAP_PROP_FRAME_WIDTH) 
    video_h = capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT) 
    frameID = capture.get(cv.CV_CAP_PROP_POS_FRAMES)  
    nb_frame = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT)-1)
    while (frameID<nb_frame):
        frame_previous2 = frame_previous1.copy()
        frame_previous1 = frame_current.copy()
        ret, frame_current = capture.read()
        frameID = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES))

        if frame_current.any() and frameID in flandmark and frameID-1 in flandmark and frameID-2 in flandmark :
            nb_face = len(flandmark[frameID])
            l_size = []
            for head, p in flandmark[frameID].items():
                x, y, w, h, mx1, my1, mx2, my2 = p
                l_size.append(w*h)

            # compute desc and write into the output file
            for head, p in flandmark[frameID].items():
                x, y, w, h, mx1, my1, mx2, my2 = p
                fout.write(str(frameID)+" "+str(head))

                # face centrality and size
                centre_face_x = x + w/2
                if nb_face == 2: 
                    center_window_x1 = video_w/3
                    center_window_x2 = 2*video_w/3
                    central = min(abs(centre_face_x-center_window_x1), abs(centre_face_x-center_window_x2))
                else:
                    center_window_x = video_w/2   
                    central = abs(centre_face_x-center_window_x)
                fout.write(" "+str(central/video_w))
                fout.write(" "+str(w*h/video_h))
                fout.write(" "+str(1 - (w*h / max(l_size))))

                # face mouvement
                if head in flandmark[frameID-1]:
                    x_p1, y_p1, w_p1, h_p1, mx1_p1, my1_p1, mx2_p1, my2_p1 = flandmark[frameID-1][head]
                    fout.write(" "+str(abs(mx1-mx1_p1)))
                    fout.write(" "+str(abs(my1-my1_p1)))
                    fout.write(" "+str(abs(mx2-mx2_p1)))
                    fout.write(" "+str(abs(my2-my2_p1))) 
                else:
                    fout.write(" -1 -1 -1 -1")
                if head in flandmark[frameID-2]:
                    x_p2, y_p2, w_p2, h_p2, mx1_p2, my1_p2, mx2_p2, my2_p2 = flandmark[frameID-2][head]
                    fout.write(" "+str(abs(mx1-mx1_p2)))
                    fout.write(" "+str(abs(my1-my1_p2)))
                    fout.write(" "+str(abs(mx2-mx2_p2)))
                    fout.write(" "+str(abs(my2-my2_p2))) 
                else:
                    fout.write(" -1 -1 -1 -1")

                # features based on OF
                centre_mouth_x = (mx2+mx1)/2
                centre_mouth_y = (my2+my1)/2
                h_mouth = (mx2-mx1)/3

                score1, l_move_x1, l_move_y1 = OpticalFlow(frame_current[centre_mouth_y-h_mouth:centre_mouth_y+h_mouth, mx1:mx2], frame_previous1[centre_mouth_y-h_mouth:centre_mouth_y+h_mouth, mx1:mx2], lk_params, feature_params)
                fout.write(" "+str(len(l_move_x1)))
                if l_move_x1 != []:
                    fout.write(" "+str(np.mean(np.array(l_move_x1))))
                    fout.write(" "+str(np.std(np.array(l_move_x1))))
                    fout.write(" "+str(np.mean(np.array(l_move_y1))))
                    fout.write(" "+str(np.std(np.array(l_move_y1))))
                else:
                    fout.write(" -1 -1 -1 -1")

                score2, l_move_x2, l_move_y2 = OpticalFlow(frame_current[centre_mouth_y-h_mouth:centre_mouth_y+h_mouth, mx1:mx2], frame_previous2[centre_mouth_y-h_mouth:centre_mouth_y+h_mouth, mx1:mx2], lk_params, feature_params)
                fout.write(" "+str(len(l_move_x2)))
                if l_move_x2 != []:
                    fout.write(" "+str(np.mean(np.array(l_move_x2))))
                    fout.write(" "+str(np.std(np.array(l_move_x2))))
                    fout.write(" "+str(np.mean(np.array(l_move_y2))))
                    fout.write(" "+str(np.std(np.array(l_move_y2))))
                else:
                    fout.write(" -1 -1 -1 -1")
                
                # features based on histogram diff
                m_x_center = (mx1+mx2)/2                        
                m_y_center = (my1+my2)/2
                mw = (mx2-mx1)/6
                mh = (mx2-mx1)/8

                if mw != 0.0 and mh != 0.0:
                    hist_current = calcul_hist(frame_current[m_y_center-mh:m_y_center+mh, m_x_center-mw:m_x_center+mw])
                    score, l_move_x, l_move_y = OpticalFlow(frame_current[centre_mouth_y-h_mouth:centre_mouth_y+h_mouth, mx1:mx2], frame_previous1[centre_mouth_y-h_mouth:centre_mouth_y+h_mouth, mx1:mx2], lk_params, feature_params)
                    move_x =0.0
                    move_y =0.0
                    if l_move_x != []:
                        move_x = int(round(np.mean(l_move_x),0))
                        move_y = int(round(np.mean(l_move_y),0))

                    ROIymin, ROIymax, ROIxmin, ROIxmax = norm_roi(m_y_center-mh+move_y, m_y_center+mh+move_y, m_x_center-mw+move_x, m_x_center+mw+move_x, video_w, video_h)
                    hist_previous1 = calcul_hist(frame_previous1[ROIymin:ROIymax, ROIxmin:ROIxmax])   
                    score, l_move_x, l_move_y = OpticalFlow(frame_current[centre_mouth_y-h_mouth:centre_mouth_y+h_mouth, mx1:mx2], frame_previous2[centre_mouth_y-h_mouth:centre_mouth_y+h_mouth, mx1:mx2], lk_params, feature_params)
                    move_x =0.0
                    move_y =0.0
                    if l_move_x != []:
                        move_x = int(round(np.mean(l_move_x),0))
                        move_y = int(round(np.mean(l_move_y),0))

                    ROIymin, ROIymax, ROIxmin, ROIxmax = norm_roi(m_y_center-mh+move_y, m_y_center+mh+move_y, m_x_center-mw+move_x, m_x_center+mw+move_x, video_w, video_h)
                    hist_previous2 = calcul_hist(frame_previous2[ROIymin:ROIymax, ROIxmin:ROIxmax])  

                    fout.write(" "+str(cv.CompareHist(hist_current, hist_previous1, cv.CV_COMP_CORREL)))
                    fout.write(" "+str(cv.CompareHist(hist_current, hist_previous2, cv.CV_COMP_CORREL)))
                else:
                    fout.write(" -1 -1")
                fout.write('\n')
    fout.close()
