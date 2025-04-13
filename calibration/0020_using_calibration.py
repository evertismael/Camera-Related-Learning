import cv2
import numpy as np


cal_file_pref = './out/0413-222615/'
mtx = np.load(cal_file_pref + '_mtx.npy')
dist = np.load(cal_file_pref + '_dist.npy')

print('-------------------------------------')
print('mtx')
print(mtx)
print('dist')
print(dist)


# case 1. compensate for distortion of the image:
image_file='./cal_imgs/0000_default/0_chess.png'

frame = cv2.imread(image_file)
frame_nodist = cv2.undistort(frame,mtx,dist,None,None)

cv2.imshow('image',frame)
cv2.imshow('undistort',frame_nodist)
cv2.waitKey(0)
cv2.destroyAllWindows()
