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



Pm = (0,0) # current mouse position.
Pm_list = [] # pixel points.


cv2.namedWindow('image-plane')
cv2.setMouseCallback('image-plane', get_cursor_position)

x_lim, y_lim, z_lim = [-50,50], [-20,20], [-20,20]
ax = make_figure(x_lim, y_lim, z_lim)


def pixel2world(args):
    global Pm_list
    img_p = cv2.imread(args.img_world)


    # Extrinsic Calibration:
    ptrn_size = [int(a) for a in args.pattern.split(',')]
    ret_list, P_chs_list, P_pxl_list,img_size = find_chessboard_on_image_files([args.img_world],ptrn_size, False)
    P_c = P_chs_list[0].T   # (3, Npoints)
    P_pxl = P_pxl_list[0].T # (2, Npoints)
    _, dist, img_size, K, _ = load_camera_calibration_matrices(args.calib_dir)
    Kinv = inv_svd(K)
    ret, rvec, tvec =cv2.solvePnP(P_c.T,P_pxl.T, K , dist)
    #M_cam_ch = make_transformation_matrix(rvec, tvec)
    M_cam_w = make_transformation_matrix(np.array([[0.,0.,np.pi/2]]), np.array([[0.,0.,0.]]).T)
    
    plot_coord_sys(M_cam_w, scale=10, sys_name='Camera', ax=ax, alpha=1)
    
    # TODO: plot camera numpy and convert the image of the cursor over the image and mtplotlib



    while True:
        img_pc = img_p.copy()
                
        # print Pm over list:
        for p in Pm_list:
            p = np.array(p).astype(int)
            cv2.circle(img_pc,p,1, (0,0,255),2)

        
        # print:
        plt.show()
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
    