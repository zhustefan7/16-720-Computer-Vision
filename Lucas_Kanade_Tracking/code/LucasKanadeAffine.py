import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy import linalg as la
import cv2
import scipy.ndimage
def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    p=np.zeros(6)
    threshold=2
    H,W=It.shape
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    while True:
        ##############################
        #warp it1 to it's perspective
        It1_warped=scipy.ndimage.affine_transform(It1,M, offset=0.0, output_shape=None, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
        #compute error image
        b=It-It1_warped
        #compute gradient
        gradient=np.gradient(It1_warped)
        gradient_x=gradient[1]
        gradient_y=gradient[0]


        H_final=np.zeros((6,6))
        Atb_final=np.zeros((6,1))
       	for i in range(H):
       		for j in range(W):
       			jacobian = np.array([[j,0,i,0,1,0],[0,j,0,i,0,1]])
       			# jacobian = np.array([[j,i,1,0,0,0],[0,0,0,j,i,1]])
       			curr_gradient = np.array([gradient_x[i,j],gradient_y[i][j]])
       			A = np.matmul(curr_gradient,jacobian)[np.newaxis,:]
       			At = np.transpose(A)
       			H_curr = np.matmul(At,A)
       			Atb_curr=At*b[i][j]

       			H_final=H_final+H_curr
       			Atb_final=Atb_final+Atb_curr
 
       	delta_p=np.matmul(la.pinv(H_final),Atb_final)

       	delta_p=np.reshape(delta_p,(6,))

        #update p
        p=p+delta_p
        M = np.array([[1.0+p[0], p[2], p[4]], [p[1], 1.0+p[3], p[5]]])
        # M = np.array([[1.0+p[0], p[1], p[2]], [p[3], 1.0+p[4], p[5]]])
        if (la.norm(delta_p) < threshold):
        	break

    return M
