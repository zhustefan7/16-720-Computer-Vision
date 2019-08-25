import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy import linalg as la
import cv2
import scipy.ndimage

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

  # put your implementation here
  threshold=2
  p=np.zeros(6)
  H,W=It.shape
  M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

	#compute gradient of the template image
  gradient=np.gradient(It)
  gradient_x=gradient[1]
  gradient_y=gradient[0]	
  
  #Evaluate the jacobian at (x;0)
  steepest_decent_imgs=[]
  H_final=np.zeros((6,6))
  for i in range(H):
 		for j in range(W):
 			jacobian = np.array([[j,0,i,0,1,0],[0,j,0,i,0,1]])
 			# jacobian = np.array([[j,i,1,0,0,0],[0,0,0,j,i,1]])
 			curr_gradient = np.array([gradient_x[i,j],gradient_y[i][j]])
 			A = np.matmul(curr_gradient,jacobian)[np.newaxis,:]
 			At = np.transpose(A)
 			H_curr = np.matmul(At,A)
 			H_final=H_final+H_curr


  while True:
    It1_warped=scipy.ndimage.affine_transform(It1,M,offset=0.0,output_shape=None,output=None,order=3,mode='constant',cval=0.0, prefilter=True)
 		#compute error image
    b=It1_warped-It

    #step 7 of the inverse compositional algorithm
    blah=np.zeros((6,1))
    for i in range(H):
   		for j in range(W):
   			jacobian = np.array([[j,0,i,0,1,0],[0,j,0,i,0,1]])
   			# jacobian = np.array([[j,i,1,0,0,0],[0,0,0,j,i,1]])
   			curr_gradient = np.array([gradient_x[i,j],gradient_y[i][j]])
   			A = np.matmul(curr_gradient,jacobian)[np.newaxis,:]
   			curr_blah=np.transpose(A)*b[i][j]
   			blah=blah+curr_blah



    delta_p=np.matmul(la.pinv(H_final),blah)
    # print(delta_p)
    delta_M = np.array([[1.0+delta_p[0][0], delta_p[2][0], delta_p[4][0]], [delta_p[1][0], 1.0+delta_p[3][0], delta_p[5][0]]])
    # print(delta_p[2][0])
    # print(delta_M.shape)
    delta_M=np.concatenate((delta_M,np.array([[0,0,1]])),axis=0)
    M=np.concatenate((M,np.array([[0,0,1]])),axis=0)
    M=np.matmul(M,la.pinv(delta_M))
    M=M[0:2,:]
    # print(delta_p)
  	#update p
    # p=p+delta_p

		#update warp matrix
		# M
    if (la.norm(delta_p) < threshold):
      break
  
  return M



