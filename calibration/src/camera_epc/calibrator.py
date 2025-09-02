import cv2, glob, argparse, os
import numpy as np
from datetime import datetime
from camera_epc.utils import find_chessboard_on_image_files



def calibrate_camera_to_world(args):
    
    # get all images in directory:
    chess_img_files = glob.glob(args.img_dir+'/*_img.png')


    # Step1: Collect all sequence of points for each chessboard, 
    #        as well as the ones in chess coords.
    print(chess_img_files)
    ret_list, P_chs_list, P_pxl_list,img_size = find_chessboard_on_image_files(chess_img_files, args.pattern_size)
    
    # select the pairs that are valid:
    P_chs_list = [P_chs for P_chs,rl in zip(P_chs_list,ret_list) if rl==True]
    P_pxl_list = [P_pxl for P_pxl,rl in zip(P_pxl_list,ret_list) if rl==True]

    if not P_chs_list or not P_pxl_list:
        print(np.array(ret_list))
        print('Chessboard was not found in none of the images')
        return
    
    # Step2: Compute the calibration, and return the matrices: (img_size(is reverted))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(P_chs_list,P_pxl_list,img_size[::-1],None,None)


    # -------------------------------------------------------------
    # Step3: show location of points on all image:
    for i in range(len(chess_img_files)):
        print(i)
        frame = cv2.imread(chess_img_files[i])
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        #frame = cv2.resize(frame,(640,480))
        cv2.drawChessboardCorners(frame,args.pattern_size,P_pxl_list[i],True)
        for p_idx, p in enumerate(P_pxl_list[i]):
            p = [int(a) for a in p]
            cv2.putText(frame,str(p_idx),p,cv2.FONT_HERSHEY_COMPLEX,.4,(0,0,0),1,1)
        
        frame_plot = cv2.resize(frame, fx=args.plot_scale, fy=args.plot_scale, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('example',frame_plot)
        cv2.waitKey(0)

    save_flag = input(f'save output matrices in {args.out_dir} [yes/no]:')
    if 'yes' in save_flag:
        calib_fname = args.out_dir + 'calib.npz'
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        np.savez(calib_fname, mtx=mtx, dist=dist, img_size=img_size)
        print(f'saved in {calib_fname}')

if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='CamCalib', description='Based on images of a given directory, it estimates the camera intrinsic parameters')
    parser.add_argument('-i', '--img_dir', type=str, default='./tests/calib_imgs_default/', help='directory of calibration images')
    parser.add_argument('-o', '--out_dir', type=str, default='./output/calib/'+datetime.now().strftime('%m%d-%H%M%S'), help='output directory to save intrinsic params')
    parser.add_argument('-s', '--plot_scale', type=float, default=1.0, help='in case the image is too big, it resize the image to a some factor (only for plotting output). It does not affect processing')
    parser.add_argument('-p','--pattern_size', type=str, default="[6,4]", help='pattern size, default [6,4]')
    args = parser.parse_args()
    args.pattern_size = list(map(int, args.pattern_size.strip('[]').split(',')))

    print(args)
    calibrate_camera_to_world(args)

