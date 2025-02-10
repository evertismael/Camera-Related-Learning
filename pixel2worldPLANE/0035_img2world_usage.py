import numpy as np
import utils as uu
import cv2

# Step 1: Load a single image that contains the chess board. 
#         This Board defines the plane over which we can recover the XYZ coordinate.
#         We cannot recover XYZ of other points, ONLY works for points over the plane.
image_file = '../calibration/img/PXL_20240823_140303256.jpg'
ptrn_size, scale_down = ((10,7)), False
P_chs_list, P_pxl_list,img_size = uu.find_sequence_chessboard_points([image_file],ptrn_size,scale_down)
P_w = P_chs_list[0].T   # (3,Npoints)
P_pxl = P_pxl_list[0].T # (2, Npoints)

# Step 2: Load camera calibration matrices:
cal_file_pref = '../calibration/out/cal1'
_, dist, img_size, mtx_new, _ = uu.load_camera_calibration_matrices(cal_file_pref)

# Step 3: Using the image pixel and world points, compute R and T.
ret, rvec, T=cv2.solvePnP(P_w.T,P_pxl.T,mtx_new,dist)
R,_ = cv2.Rodrigues(rvec)
Ainv = uu.inv_svd(mtx_new)

# Step 4: with R,T,Ainv known we can map any point in the image 
#         to the corresponding point in the plane of the chessboard in world coords.
P_w_hat = uu.uv2XYZ(P_pxl, Ainv, R, T)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
for i in range(70):
    print(f'estimated: {P_w_hat[:,i]} vs given: {P_w[:,i]}')
    a=3