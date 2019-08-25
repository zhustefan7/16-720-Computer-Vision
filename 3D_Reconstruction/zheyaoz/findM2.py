'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
# from helper import*
import cv2
import numpy.linalg as la
from submission import*

def findM2(I1, I2, K1, K2, pts1, pts2, F):
	min_error=0

    #calculate essential matrix
	E=essentialMatrix(F, K1, K2)

    #calcuate camera matrix
	M2s=camera2(E)
	M1=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
	C1=np.matmul(K1,M1)
	positive_z_num = 0

	for i in range(4):
		M2 = M2s[:,:,i]
		C2=np.matmul(K2,M2)
		p,curr_error = triangulate(C1, pts1, C2, pts2)
		positive_z_num = np.sum((p[:,2]>0))
		print(positive_z_num)
		if positive_z_num == len(pts1):
			break



	M2 = np.matmul(la.inv(K2),C2)

	return M2, C2, p


if __name__ == "__main__":

		#load images
	I1 = cv2.imread('../data/im1.png')
	I2 = cv2.imread('../data/im2.png') 
	H,W,D = I1.shape

    #load intrinscis matri
	intrinsics=np.load('../data/intrinsics.npz')
	K1 = intrinsics['K1']
	K2 = intrinsics['K2']


	correspondence=np.load('../data/some_corresp.npz')
	pts1=correspondence['pts1']
	pts2=correspondence['pts2']


	intrinsics=np.load('../data/intrinsics.npz')
	K1=intrinsics['K1']
	K2=intrinsics['K2']

	#calculate Fundamenal Matrix
	H,W,D=I1.shape
	M=max(H,W)
	F=eightpoint(pts1,pts2,M)

	M2, C2, P = findM2(I1, I2, K1, K2, pts1, pts2, F)
	np.savez("../Files_to_submit/q3_3.npz" , M2=M2, C2=C2, P=P)



