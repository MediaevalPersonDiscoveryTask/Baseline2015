"""
Compute HoG descriptor projected by LDML for each facetrack

Usage:
  facetrack_descriptor <video> <flandmark> <features_model> <ldml_matrix> <output_file> 
  facetrack_descriptor -h | --help
"""

from docopt import docopt
import json
import numpy as np
import cv2, cv
import os, sys, getopt
import math

class Point2D32f:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class CvRect:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

def sqdist(desc1, desc2):
    dim = min(desc1.shape[0], desc2.shape[0])
    dist = 0.0
    for i in range(dim):
        d = desc1[i] - desc2[i]
        dist += d*d
    return dist

def int_round_tab(t):    
    new_t = []
    for e in t:
        new_t.append(int(round(e)))
    return new_t

def read_face_landmarks_file(flandmark):
    dic_flm = {}
    for line in open(flandmark).read().splitlines():
        l = line.split(' ')
        frame = int(l[0])
        i_face =int(l[1])
        dic_flm.setdefault(frame, {})
        dic_flm[frame][i_face] = []
        #re-order the landmarks [eyes - nose - mouth]
        for i in [16, 17, 8, 9, 10, 11, 18, 19, 20, 21, 12, 13, 14, 15]:
            dic_flm[frame][i_face].append(float(l[i]))
    return dic_flm

def load_feature_model(filepath):
    model = []
    for line in open(filepath):
        model = map(float, line[:-1].split(' '))[2:]
    return model

def compute_center(pts):     # ok
    center = Point2D32f(0.0,0.0)
    for i in range(N):
        center.x += pts[i*2]
        center.y += pts[i*2+1]
    center.x /= N
    center.y /= N
    return center

def compute_affine_transformation(src, dst):
    src_center = compute_center(src)
    dst_center = compute_center(dst)

    X = cv.CreateMat(2, N, cv.CV_32FC1)
    Y = cv.CreateMat(2, N, cv.CV_32FC1)
    
    for i in range(N):
        cv.Set2D(Y, 0, i, src[i*2] - src_center.x)
        cv.Set2D(Y, 1, i, src[i*2+1] - src_center.y)
        cv.Set2D(X, 0, i, dst[i*2] - dst_center.x)
        cv.Set2D(X, 1, i, dst[i*2+1] - dst_center.y)    

    #Get transpose matrix
    Xt = cv.CreateMat(N, 2, cv.CV_32FC1)
    cv.Transpose(X, Xt)
    
    YXt = cv.CreateMat(2, 2, cv.CV_32FC1)
    cv.MatMul(Y, Xt, YXt)

    W = cv.CreateMat(2, 2, cv.CV_32FC1)
    Ut = cv.CreateMat(2, 2, cv.CV_32FC1)
    V = cv.CreateMat(2, 2, cv.CV_32FC1)    

    #The flags cause U to be returned transposed (does not work well without the transpose flags).
    cv.SVD(YXt, W, Ut, V, cv.CV_SVD_U_T) # A = U W V^T

    #Compute s = sum(sum( X.*(R*Y) )) / sum(sum( Y.^2 ));
    R = cv.CreateMat(2, 2, cv.CV_32FC1)
    cv.MatMul(V, Ut, R)

    RY = cv.CreateMat(2, N, cv.CV_32FC1)
    cv.MatMul(R, Y, RY)

    XRY = cv.CreateMat(2, N, cv.CV_32FC1)
    cv.Mul(X, RY, XRY, 1)

    YY = cv.CreateMat(2, N, cv.CV_32FC1)
    cv.Mul(Y, Y, YY, 1)

    #Compute scale, sum, angle
    scale = 0.0
    sum = 0.0
    for i in range(N):
        scale += cv.Get2D(XRY, 0, i)[0] + cv.Get2D(XRY, 1, i)[0]
        sum += cv.Get2D(YY, 0, i)[0] + cv.Get2D(YY, 1, i)[0]
    if sum != 0:
        scale = scale/sum
    else:
        scale = 1   
    angle = math.atan(cv.Get2D(R, 0, 1)[0]/cv.Get2D(R, 0, 0)[0]) * 180.0 / cv.CV_PI
   
    #Compute matrix
    warp_matrix = cv.CreateMat(2, 3, cv.CV_32FC1);
    cv.GetRotationMatrix2D((src_center.x, src_center.y), angle, scale, warp_matrix);

    return warp_matrix;

def compute_integral_hog(img):
    img_xsobel = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_32F, 1)
    img_ysobel = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_32F, 1)
    ksize = 1
    
    if(img.nChannels == 3)  :
        img_gray = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U, 1);
        cv.CvtColor(img, img_gray, cv.CV_BGR2GRAY);
        cv.EqualizeHist(img_gray,img_gray);
    
        cv.Sobel(img_gray, img_xsobel, 1, 0, ksize);
        cv.Sobel(img_gray, img_ysobel, 0, 1, ksize);

    else :
        cv.EqualizeHist(img, img);
        cv.Sobel(img, img_xsobel, 1, 0, ksize);
        cv.Sobel(img, img_ysobel, 0, 1, ksize);
    
    magnitude = cv.CreateMat(img.height, img.width, cv.CV_32F);
    angle = cv.CreateMat(img.height, img.width, cv.CV_32F);
 
    cv.CartToPolar(img_xsobel, img_ysobel, magnitude, angle, 1);    

    img_bins = []
    img_integrals = []
    
    for i in range(nbins):
        img_bins.append(cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_32F,1))
        cv.SetZero(img_bins[i])
        img_integrals.append(cv.CreateImage((img.width+1, img.height+1), cv.IPL_DEPTH_64F,1))

    dev = float(nbins) / 360
    for j in range(img.height):
        for i in range(img.width):     
            bin = int(float(cv.Get2D(angle, j, i)[0]) * dev)
            cv.Set2D(img_bins[bin], j, i, cv.Get2D(magnitude, j, i))

    for i in range(nbins):
        cv.Integral(img_bins[i], img_integrals[i], None, None);

    return img_integrals

def compute_hog_cell(hog, img_integrals, cell):
    hog_cell = []
    if( cell.x<0 or cell.y<0 or cell.x + cell.width>= img_integrals[0].width or cell.y + cell.height>=img_integrals[0].height) :    #case we are out of the image size
        for i in range(nbins):
            hog_cell.append(0)
    else :
        for i in range(nbins):
            h0 = cv.Get2D(img_integrals[i],cell.y, cell.x)[0]
            h1 = cv.Get2D(img_integrals[i],cell.y,cell.x + cell.width)[0]
            h2 = cv.Get2D(img_integrals[i],cell.y + cell.height,cell.x + cell.width)[0]
            h3 = cv.Get2D(img_integrals[i],cell.y + cell.height, cell.x)[0]
            hog_cell.append(float(h2 - h3 - h1 + h0))
    return hog_cell, hog

def compute_hog_block(hog,  img_integrals, block):
    nb = 0;
    sum = 0;
    cell = CvRect(0, 0, cell_sz, cell_sz);
    for i in range(block.x, block.x + block.width, cell_sz):
        for j in range(block.y, block.y + block.height, cell_sz):            
            if(i<0 or j<0 or i>=img_integrals[0].width or j>=img_integrals[0].height) :            #case we are out of the image size
                for h in range(nbins):
                    hog.append(0)
                    nb+=1
            else :
                cell.x = i
                cell.y = j
                hog_cell, hog = compute_hog_cell(hog, img_integrals, cell)     
                for h in range(nbins):
                    hog.append(hog_cell[h])
                    sum += hog[nb]*hog[nb]
                    nb+=1
    eps = 0.00000001
    for i in range(nb):
        hog[i] /= math.sqrt(sum + eps*eps);    
    return hog

def compute_hog_desc(hog, img_integrals, point):
    shift = block_sz * cell_sz / 2
    block = CvRect(point.x - shift, point.y - shift, block_sz*cell_sz, block_sz*cell_sz)
    hog = compute_hog_block(hog, img_integrals, block);
    return hog

def compute_aligned_face_descriptor(src, features, features_model, dim_HoG_pts):
    desc = []
    pts_dst = []
    pts = features[:]
    pts_mean = features_model[:]
    
    dst = cv.CreateImage(cv.GetSize(src), 8, 3);
    warp_matrix = compute_affine_transformation(pts, pts_mean);
    cv.WarpAffine(src, dst, warp_matrix, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS, (0, 0, 0, 0))   
    img_int = compute_integral_hog(dst)
    for j in range(N):        #compute new point
        hog = []
        a = cv.Get2D(warp_matrix, 0, 0)[0]
        b = cv.Get2D(warp_matrix, 0, 1)[0]
        c = cv.Get2D(warp_matrix, 0, 2)[0]
        d = cv.Get2D(warp_matrix, 1, 0)[0]
        e = cv.Get2D(warp_matrix, 1, 1)[0]
        f = cv.Get2D(warp_matrix, 1, 2)[0]
        pts_dst.append(a * pts[j*2] + b * pts[j*2+1] + c)
        pts_dst.append(d * pts[j*2] + e * pts[j*2+1] + f)
        pt = Point2D32f(int(pts_dst[j*2]),int(pts_dst[j*2+1]))        
        hog = compute_hog_desc(hog, img_int, pt);                             
        for k in range(dim_HoG_pts):
            desc.append(hog[k])
    return desc


if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
    # size of the descriptor before projection
    N = 7
    nbins = 8
    cell_sz = 7
    block_sz = 7
    dim_HoG_pts = nbins*cell_sz*block_sz
    dim_HoG_total = N*dim_HoG_pts
    # size of the descriptor after projection
    K = 100    
    # load projection matrix
    L = np.fromfile(args['<ldml_matrix>'], sep=' ')
    L = np.array(L)
    L = L.reshape(K, dim_HoG_total)  
    # open video
    capture = cv2.VideoCapture(args['<video>'])
    # load features model
    features_model = load_feature_model(args['<features_model>']); 
    # read position of facial landmark
    dic_flm = read_face_landmarks_file(args['<flandmark>'])
    last_frame_to_process = max(dic_flm)
    # compute descriptor
    c_frame = 0
    desc_facetrack = {}
    while (c_frame<last_frame_to_process):   
        ret, frame_tmp = capture.read()
        c_frame = capture.get(cv.CV_CAP_PROP_POS_FRAMES) 
        if ret and c_frame in dic_flm: 
            frame = cv.CreateImageHeader((frame_tmp.shape[1], frame_tmp.shape[0]), cv.IPL_DEPTH_8U, 3)
            cv.SetData(frame, frame_tmp.tostring(), frame_tmp.dtype.itemsize * 3 * frame_tmp.shape[1])            
            for i_face, flandmark in dic_flm[c_frame].items():
                if flandmark[0] != -1:
                    HoG_desc = compute_aligned_face_descriptor(frame, flandmark, features_model, dim_HoG_pts)
                    projected_desc = np.dot(np.array(HoG_desc, dtype='|S20').astype(np.float), L.T)
                    desc_facetrack.setdefault(i_face, []).append(projected_desc)
    # save the central projected descriptor for each facetrack
    fout = open(args['<output_file>'], 'w')
    for i_face, l_desc in sorted(desc_facetrack.items()):
        min_dist = +np.inf
        best_desc = []
        # find the more central descriptor
        for desc1 in l_desc:
            l_d = []
            for desc2 in l_desc:
                l_d.append(sqdist(desc1, desc2))
            d = np.mean(np.array(l_d))
            if d < min_dist:
                min_dist = d
                best_desc = desc1
        fout.write(str(i_face)+' '+str(min_dist))
        for e in best_desc:
            fout.write(' '+str(e))
        fout.write('\n')
    fout.close()
    capture.release() 
