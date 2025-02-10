import cv2, glob
import numpy as np
from utils import  find_sequence_chessboard_points, inv_svd



image_file = './img/PXL_20240823_140303256.jpg'
ptrn_size, scale_down = ((10,7)), False
P_chs_list, P_pxl_list,img_size = find_sequence_chessboard_points([image_file],ptrn_size,scale_down)
P_w = P_chs_list[0].T
P_pxl = P_pxl_list[0].T


# load camera calibration params:
cal_file_pref = './out/cal1'
mtx = np.load(cal_file_pref + '_mtx.npy')
dist = np.load(cal_file_pref + '_dist.npy')
img_size = np.load(cal_file_pref + '_imgsize.npy')
# refine mtx:
newcameramtx, roi =cv2.getOptimalNewCameraMatrix(mtx, dist, img_size[::-1], 1, img_size[::-1])



# find R and T using mtx and dist
ret, rvec, tvec=cv2.solvePnP(P_w.T,P_pxl.T,newcameramtx,dist)
R,_ = cv2.Rodrigues(rvec)
T = tvec
A = newcameramtx
Ainv = inv_svd(A)


# compute Z_c for all points over the plane:
# Z_c = <r_3^T, T> / <r_3^T, Ainv U>
# with U = [u,v,1]^T
# r_3 is the third column of R
U = np.row_stack((P_pxl,np.ones((1,P_pxl.shape[1]))))
r3 = R[:,2,None]
Z_c = r3.T.dot(T)/r3.T.dot(Ainv.dot(U))

# compute XYZ:
P_w_hat = R.T.dot(Z_c*Ainv.dot(U) - T)


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
for i in range(70):
    print(P_w_hat[:,i], P_w[:,i])
    a=3

a=3