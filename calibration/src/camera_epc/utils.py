import cv2
import numpy as np

def detect_chess_board_points(frame, ptrn_size):
    ''' Precompute Pw positions: By default cv2 locates the origin at the left-upper corner.
         format: (0,0,0),(1,0,0),(2,0,0)......(last,last,0)
         world coords is over the paper hence z=0 for all poins.
    '''
    P_chs = np.zeros((ptrn_size[0]*ptrn_size[1],3), np.float32)
    P_chs[:,:2] = np.mgrid[0:ptrn_size[0],0:ptrn_size[1]].T.reshape(-1,2)

    # convert to grey
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    ret, P_pxl_tmp = cv2.findChessboardCorners(frame_gray, ptrn_size,None)

    if ret==True:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        P_pxl = cv2.cornerSubPix(frame_gray,P_pxl_tmp,(10,10),(0,0),criteria)
        P_pxl = P_pxl.reshape(-1,2)
        
        # Notice that since each point is taken from different images, the origin of the world
        # coordinates (corner of chessboard with 0 index) might be different.
        # Ensure that the origin is the left most corner.
        if P_pxl[0,1] > P_pxl[-1,1]: # P0 to the right of Pf (INVERT ORDER)
            P_pxl = P_pxl[::-1,:]
    else:
        P_chs = []
        P_pxl = []
    return ret, P_chs, P_pxl



def find_chessboard_on_image_files(chess_img_files, ptrn_size):
    ''' Collects the pair of points in chess and pixel coords.
        Each element in the lists are the associated ones to each image.
         
             ptrn_size: This is very important!!! wrong param->unable to decode.
    '''
       
    # creating the lists (one sequence per image)
    P_chs_list = []
    P_pxl_list = []
    ret_list = []
    img_size = 0,0
    for img_file in chess_img_files:
        frame = cv2.imread(img_file)
        
        ret, P_chs, P_pxl = detect_chess_board_points(frame, ptrn_size)
        
        # append results
        ret_list.append(ret)
        P_chs_list.append(P_chs)
        P_pxl_list.append(P_pxl)
        img_size = frame.shape[:2]
    return ret_list, P_chs_list, P_pxl_list,img_size

