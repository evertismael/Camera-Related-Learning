import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from libs.show import make_figure, plot_coord_sys
from libs.utils import load_camera_calibration_matrices, find_chessboard_on_image_files, inv_svd
from libs.coordsys import make_transformation_matrix

def get_cursor_position(event,x,y,flags,param):
    global Pm_list, Pm
    if event == cv2.EVENT_LBUTTONDBLCLK:
        Pm_list.append((x,y))
    elif event== cv2.EVENT_MOUSEMOVE:
        Pm = (x,y)

def img_to_plane(R_cp, p_c, P_pxl, Kinv):
    # compute Z_c for all points over the plane:
    # Z_c = <r_3^T, T> / <r_3^T, Ainv U>
    # with U = [u,v,1]^T
    # r_3 is the third column of R
    U = np.vstack((P_pxl,np.ones((1,P_pxl.shape[1]))))
    r3 = R_cp[:,2,None]
    Z_c = r3.T.dot(p_c)/r3.T.dot(Kinv.dot(U))

    # compute XYZ:
    P_w_hat = R_cp.T.dot(Z_c*Kinv.dot(U) - p_c)
    return P_w_hat

Pm = (0,0) # current mouse position.
Pm_list = [] # pixel points.


cv2.namedWindow('image-plane')
cv2.setMouseCallback('image-plane', get_cursor_position)

x_lim, y_lim, z_lim = [-5,5], [0,15], [-5,5]
fig, ax = make_figure(x_lim, y_lim, z_lim)


def pixel2world(args):
    global Pm_list
    img_p = cv2.imread(args.img_world)


    # Extrinsic Calibration:
    ptrn_size = [int(a) for a in args.pattern.split(',')]
    ret_list, P_chs_list, P_pxl_list,img_size = find_chessboard_on_image_files([args.img_world],ptrn_size, False)
    P_p = P_chs_list[0].T   # (3, Npoints)
    P_pxl = P_pxl_list[0].T # (2, Npoints)
    _, dist, img_size, K, _ = load_camera_calibration_matrices(args.calib_dir)
    Kinv = inv_svd(K)

    # obtain H_cp: from chess-Plane to Camera frame
    ret, rvec_cp, p_c =cv2.solvePnP(P_p.T,P_pxl.T, K , dist)
    H_cp = make_transformation_matrix(rvec_cp, p_c)
    H_pc = np.linalg.pinv(H_cp)

    # camera orientation: from world to camera frame:
    H_cw = make_transformation_matrix(np.array(np.deg2rad([90,0,0])), np.zeros_like(p_c))
    H_wc = np.linalg.pinv(H_cw)

    # plot world and camera frames:
    plot_coord_sys(np.eye(4), scale=100, sys_name='W', ax=ax, alpha=.1)
    ax.set_xlabel('x[u]'), ax.set_ylabel('y[u]'), ax.set_zlabel('z[u]')
    plot_coord_sys(H_wc, scale=5, sys_name='Camera', ax=ax, alpha=1)
    
    # print chessboard points in 3D:
    P1_p = np.concat((P_p, np.ones((1,P_p.shape[1]))),axis=0)
    P1_w =  H_wc.dot(H_cp.dot(P1_p))
    ax.scatter(P1_w[0,:], P1_w[1,:], P1_w[2,:], s=1, alpha=0.3, c='k')
    m_hdlr = ax.scatter([],[], [], s=1, alpha=1, c='r')
    
    
    plt.draw()
    plt.pause(0.001)
    while True:
        img_pc = img_p.copy()

        # points to plane:
        if len(Pm_list)>0:
            Pm_pxl = np.array(Pm_list).T
            Pm_p_hat = img_to_plane(R_cp=H_cp[:3,:3], p_c=H_cp[:3,3,None], P_pxl=Pm_pxl, Kinv=Kinv)

            # compute XYZ:
            Pm1_p_hat = np.concat((Pm_p_hat, np.ones((1,Pm_p_hat.shape[1]))),axis=0)
            Pm1_w_hat =  H_wc.dot(H_cp.dot(Pm1_p_hat))
            m_hdlr.remove()
            m_hdlr = ax.scatter(Pm1_w_hat[0,:], Pm1_w_hat[1,:], Pm1_w_hat[2,:], s=10, alpha=1, c='r')

        # print Pm over list:
        for p in Pm_list:
            p = np.array(p).astype(int)
            cv2.circle(img_pc,p,1, (0,0,255),2)

        # print:
        plt.draw()
        plt.pause(0.001)
        cv2.imshow('image-plane',img_pc)
        key = cv2.waitKey(1)
        if key==ord('q'):
            break
        elif key==ord('r'):
            Pm_list = []
    cv2.destroyAllWindows()





if __name__ == '__main__':
    parse = argparse.ArgumentParser(prog='pixel2world', description='Interactive pixel to world')
    parse.add_argument('--calib_dir', type=str, default='./calibration/0000_default/')
    parse.add_argument('--img_world', type=str, default='./img/0_chess.png')
    parse.add_argument('--pattern', type=str, default='6,4')
    args = parse.parse_args()

    print(args)
    pixel2world(args)
    