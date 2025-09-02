import cv2
import os
import time, argparse
from datetime import datetime
from camera_epc.utils import detect_chess_board_points


def collect_camera_calibration_images(args):
    '''
    It opens a camera view and tries to detect the points of a chessboard.
    When pressed 'c' it saves two images 
        1. *_img.png is the image to be used later on.
        2. *_chess.png is the same as *_img.png but with the chessboard points drawn over the image.
    Notice that the saving is refused if the chessboard is not detected.
    '''

    # init camera with single image buffer
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_BUFFERSIZE,1)
    a=cam.get(cv2.CAP_PROP_BUFFERSIZE) 
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # collect images:
    idx = 0
    while True:
        ret, frame = cam.read()
        frame_to_save = frame.copy()
        
        # try to detect the chessboard
        ptrn_size = ((6,4))
        ret, P_chs, P_pxl = detect_chess_board_points(frame, ptrn_size)

        # if detected draw:
        if ret==True:
            cv2.drawChessboardCorners(frame,ptrn_size,P_pxl,ret)
            for p_idx, p in enumerate(P_pxl):
                p = [int(a) for a in p]
                cv2.putText(frame,str(p_idx),p,cv2.FONT_HERSHEY_COMPLEX,.4,(0,0,0),1,1)

        cv2.imshow('camera',frame)


        pressed_key = cv2.waitKey(1)
        if pressed_key==ord('q'):
            break
        elif pressed_key==ord('c'):
            if ret==True:
                # check if directory exists
                if not os.path.exists(args.out_dir):
                    os.makedirs(args.out_dir)
                print('capturing image')
                filename = args.out_dir+'/'+str(idx)+'_img.png'
                cv2.imwrite(filename, frame_to_save)

                filename = args.out_dir+'/'+str(idx)+'_chess.png'
                cv2.imwrite(filename, frame)
                
                print('-------------------------------------------------')
                print(f'image {filename} saved')
                print('-------------------------------------------------')
                time.sleep(1)
                idx+=1
            else:
                print(f'Chess not in image - NOT SAVED')

    cam.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='CamImgCollect', description='RUN THIS in the Rpi. Collection of images for camera calibration. Intrinsic Params')
    parser.add_argument('-o', '--out_dir', type=str, default='./output/calib_imgs/'+datetime.now().strftime('%m%d-%H%M%S'), help='OutputDirectory')
    args = parser.parse_args()
    print("RUNT THIs in the RPI")
    print(args)
    collect_camera_calibration_images(args)
