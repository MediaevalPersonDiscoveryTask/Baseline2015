import cv2, cv
import numpy as np

def calcul_hist(src_np):  
    height, width, depth = src_np.shape
    src = cv.CreateImageHeader((width, height), cv.IPL_DEPTH_8U, 3)
    cv.SetData(src, src_np.tostring(), src_np.dtype.itemsize * 3 * width)
    hsv = cv.CreateImage((width, height), 8, 3)
    cv.CvtColor(src, hsv, cv.CV_BGR2HSV)
    h_plane = cv.CreateMat(height, width, cv.CV_8UC1)
    s_plane = cv.CreateMat(height, width, cv.CV_8UC1)
    cv.Split(hsv, h_plane, s_plane, None, None)
    hist = cv.CreateHist([30, 32], cv.CV_HIST_ARRAY, [[0, 180], [0, 255]], 1)    
    cv.CalcHist([cv.GetImage(i) for i in [h_plane, s_plane]], hist)
    return hist  
    
def prop_pts_find_by_optical_flow(img0, img1, lk_params, feature_params):
    # convert image to gray
    img0_tmp = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1_tmp = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # find point on the 2 images
    p0 = cv2.goodFeaturesToTrack(img0_tmp, **feature_params)    
    p1 = cv2.goodFeaturesToTrack(img1_tmp, **feature_params)    
    if p0 is None and p1 is None:
        return 1.0

    # find point on the other images    
    st1_0, st0_1 = [], []
    if p0 is not None:
        p0_1, st0_1, err0_1 = cv2.calcOpticalFlowPyrLK(img0_tmp, img1_tmp, p0, None, **lk_params)
        st0_1 = list(np.array(st0_1).T[0])
    if p1 is not None:
        p1_0, st1_0, err1_0 = cv2.calcOpticalFlowPyrLK(img1_tmp, img0_tmp, p1, None, **lk_params)
        st1_0 = list(np.array(st1_0).T[0])

    # return proportion of point retrieved
    return np.mean(st0_1 + st1_0)


def OpticalFlow(img0, img1, lk_params, feature_params):
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(img0, mask = None, **feature_params)    
    p1 = cv2.goodFeaturesToTrack(img1, mask = None, **feature_params)    
    if p0 is None or  p1 is None:
        return -1.0, [], []
    
    st1_0, st0_1 = [], []
    if p0 is not None:
        p0_1, st0_1, err0_1 = cv2.calcOpticalFlowPyrLK(img0_tmp, img1_tmp, p0, None, **lk_params)
        st0_1 = list(np.array(st0_1).T[0])
    if p1 is not None:
        p1_0, st1_0, err1_0 = cv2.calcOpticalFlowPyrLK(img1_tmp, img0_tmp, p1, None, **lk_params)
        st1_0 = list(np.array(st1_0).T[0])

    l_move_x = []
    l_move_y = []
    for pts0, pts1, s in itertools.izip(p0, p1_0, st1_0):
        if s[0] == 1:
            l_move_x.append(pts1[0][0]-pts0[0][0])
            l_move_y.append(pts1[0][1]-pts0[0][1])

    return np.mean(st0_1 + st1_0), l_move_x, l_move_y

def score_coverage_ROI(ROI1, ROI2):
    width_inter = min(ROI1[2], ROI2[2])-max(ROI1[0], ROI2[0])
    height_inter = min(ROI1[3], ROI2[3])-max(ROI1[1], ROI2[1])
    area_ROI1 = (ROI1[3]-ROI1[1]) * (ROI1[2]-ROI1[0])
    area_ROI2 = (ROI2[3]-ROI2[1]) * (ROI2[2]-ROI2[0])
    if width_inter < 0 or height_inter < 0:
        return 0.0
    area_intersection = width_inter * height_inter
    area_union = area_ROI1 + area_ROI2 - area_intersection
    return float(area_intersection) / area_union
