import cv2
import numpy as np
from utils import inv_svd
rng = np.random.default_rng()

#img = cv2.imread('./img/2Bfo4.png')
img = cv2.imread('./img/PXL_20240823_140332294.jpg')
imS = cv2.resize(img, (960, 540))
print(img.shape)

# get intersection points in pixel coords.
# 1. convert to gray:
imG = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)
print(imG.shape)
# 2. find chessboard corners:
pattern_squares = (10,7) # the pattern number of squares is TOO important!.
ret, corners = cv2.findChessboardCorners(imG,pattern_squares,None)
print(ret)
if ret==True:
    # 3. refine corners using cornerSubPixel:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    sb_corners = cv2.cornerSubPix(imG,corners, (11,11), (-1,-1), criteria)
    
    # just save the points for later use.
    Ppx = sb_corners.copy()
    cv2.drawChessboardCorners(imG,pattern_squares,sb_corners,ret)
    
    # print the order of detections:
    for i,Pp in enumerate(Ppx):
        Tp = tuple([int(a) for a in list(Pp[0])])
        cv2.putText(imG, str(i), Tp, cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)
    
    Ppx = Ppx.reshape(70,2)                                               

    # 4. Points in world coords: NOTICE: the scale is one block.
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    Pw = np.zeros((pattern_squares[0]*pattern_squares[1],3), np.float32)
    Pw[:,:2] = np.mgrid[0:pattern_squares[0],0:pattern_squares[1]].T.reshape(-1,2)
    # in the prev line, Pw are in world coords.
    

    # 5. With both Pw and Ppx we compute the calibration:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([Pw], [Ppx], imG.shape[::-1], None, None)
    # mtx: camera matrix (intrinsic)
    # dist: distortion coeefs:
    # rvecs: rotation vectors
    # tvecs: translation vectors

    # ---------------------------------------------------------------------------------
    # user camera matrix to refine camera matrix
    img_dist = imG.copy()
    h,  w = img_dist.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    img_undist = cv2.undistort(img_dist, mtx, dist, None, newcameramtx)

    # or using another approach:
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    img_undist2 = cv2.remap(img_dist, mapx, mapy, cv2.INTER_LINEAR)

    #cv2.imshow('dist', img_dist)
    #cv2.imshow('Undist', img_undist)
    #cv2.imshow('Undist2', img_undist2)
    #cv2.waitKey(1)

    # ---------------------------------------------------
    # ---------------------------------------------------
    # ---------------------------------------------------
    # tHIS IS NOT POSIBLE HERE SINCE THE ESTIMATION OF MTX WITH A SINGLE IMAGE IS VERY POOR;
    # THE NEXT APPROACH IS TO ESTIMATE  MTX  FROM SEVERAL IMAGES AND TO SAVE SOME OTHER
    # IMAGE TO COMPUTE NEW R AND T; AFTER THAT WE CAN DO A SIMILAR APPROACH DOWN HERE BUT THAT
    # WILL BE IMPLEMENTED IN OFFICIAL DEVELOPMENT OS THIS LIB; NOT IN EXPERIMENTAL SCRIPTS AS THIS ONE;
    # 6 Project back Pw to Ppx to see what the error is:
    Ppx_new, _ = cv2.projectPoints(Pw, rvecs[0], tvecs[0], mtx, dist)


    # projecting from world to pixel:
    R, _  = cv2.Rodrigues(rvecs[0])
    T = tvecs[0]
    
    # Ppx = mtx(R*Pw+T)
    RPw = np.dot(R,Pw.T)
    Ppx_my = np.dot(mtx,RPw+T)
    Ppx_my = Ppx_my/Ppx_my[2,:]


    # 7 Project back Ppx to Pw:
    # zc*Ppx = mtx(R*Pw+T)
    # Pw =  R_inv(zc*mtx_inv*Ppx - T)
    mtx_inv = inv_svd(mtx)
    zc = T[2]
    
    Ppx_ = np.concatenate((Ppx.copy(),np.ones((70,1))),axis=1)
    Ppx_ = Ppx_.T
    Pw_my = np.dot(R.T, zc*np.dot(mtx_inv,Ppx_) - T)



    print(np.dot(mtx_inv,mtx))


    # projecting from pixel to world:
    a = rng.normal(size=(9, 6)) + 1j*rng.normal(size=(9, 6))
    U, S, Vh = np.linalg.svd(a, full_matrices=True)
    #U.shape, S.shape, Vh.shape
    tmp1 = U[:, :6] * S
    tmp = np.dot(tmp1, Vh)
    np.allclose(a, np.dot(U[:, :6] * S, Vh))
    
    smat = np.zeros((9, 6), dtype=complex)
    smat[:6, :6] = np.diag(S**-1)
    ainv = np.dot(U, np.dot(smat, Vh))
    ch = np.matmul(a,ainv)
    np.allclose(ch, np.eye((6,6)))


    u,s,v=np.linalg.svd(MM.copy())
    tmp = np.dot(np.diag(s**-1),u.transpose())
    MMinv=np.dot(v.transpose(),tmp)
    check = np.matmul(MMinv,MM)

    Pw_my_all = []
    for i in range(Ppx.shape[0]):
        Ppxi = Ppx[i]
        Ppxi = np.concatenate((Ppxi,np.ones((1,1))),axis=1)
        Ppxi = Ppxi.transpose()
        Pwi = np.matmul(MMinv,Ppxi)
        Pwi = Pwi/Pwi[3]
        Pw_my_all.append(Pwi)

    a=2



cv2.imshow('chessboard', imS)
cv2.imshow('chessboard2', imG)
cv2.waitKey(1)
cv2.destroyAllWindows()
