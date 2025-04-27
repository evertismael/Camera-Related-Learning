import cv2, glob
import numpy as np
from libs.utils import  find_chessboard_on_image_files, inv_svd, load_camera_calibration_matrices



image_file = './img/0_chess.png'
ptrn_size, scale_down = ((6,4)), False
ret_list, P_chs_list, P_pxl_list, img_size = find_chessboard_on_image_files([image_file],ptrn_size,scale_down)
assert ret_list[0]==True, 'Pattern not found in image. Check that pattern is correctly configured'
P_w = P_chs_list[0].T
P_pxl = P_pxl_list[0].T


# load camera calibration params:
cal_file_pref = './calibration/0000_default/'
_, dist, img_size, mtx_new, _ = load_camera_calibration_matrices(cal_file_pref)

# find R and T using mtx and dist
ret, rvec, tvec=cv2.solvePnP(P_w.T,P_pxl.T,mtx_new,dist)
R,_ = cv2.Rodrigues(rvec)
T = tvec
K = mtx_new
Kinv = inv_svd(K)


def img_to_plane(R_cp, p_c, P_pxl):
    # compute Z_c for all points over the plane:
    # Z_c = <r_3^T, T> / <r_3^T, Ainv U>
    # with U = [u,v,1]^T
    # r_3 is the third column of R
    U = np.row_stack((P_pxl,np.ones((1,P_pxl.shape[1]))))
    r3 = R_cp[:,2,None]
    Z_c = r3.T.dot(p_c)/r3.T.dot(Kinv.dot(U))

    # compute XYZ:
    P_w_hat = R_cp.T.dot(Z_c*Kinv.dot(U) - p_c)
    return P_w_hat

P_w_hat = img_to_plane(R_cp=R, p_c=T, P_pxl=P_pxl)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
for i in range(P_w.shape[1]):
    print(P_w_hat[:,i], P_w[:,i])