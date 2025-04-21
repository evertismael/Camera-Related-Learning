import numpy as np
import cv2


def make_transformation_matrix(rvec_1_2, tvec_1_2):
    R_1_2, _ = cv2.Rodrigues(rvec_1_2)
    H_1_2 = np.zeros((4,4))
    H_1_2[:3, :3] = R_1_2
    H_1_2[:3,  3] = tvec_1_2[:,0]
    H_1_2[3,3] = 1
    
    return H_1_2