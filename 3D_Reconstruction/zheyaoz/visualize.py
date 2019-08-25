'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
# from helper import*
import cv2
import numpy.linalg as la
from submission import*
from findM2 import*
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def visualize():


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

	#calculate F
	M = max(H,W)
	F = eightpoint(pts1,pts2,M)

	E=essentialMatrix(F, K1, K2)
    #calcuate camera matrix
	M2s=camera2(E)
	M1=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
	C1=np.matmul(K1,M1)

    #Calculate correspondant points 
	selected_pts = np.load('../data/templeCoords.npz')
	pts1 = np.concatenate((selected_pts['x1'],selected_pts['y1']),axis=1)
	pts2=[]

	for i in range(len(pts1)):
		x1,y1 = pts1[i,:]
		x2, y2 = epipolarCorrespondence(I1, I2, F, x1, y1)

		pts2.append([x2,y2])

	pts2=np.array(pts2)


    #calculating C2
	M2, C2, p = findM2(I1, I2, K1, K2, pts1, pts2, F)

	np.savez("../Files_to_submit/q4_2.npz" , M1=M1 ,M2=M2, C2=C2, P=p)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')	
	ax.scatter(p[:,0], p[:,1], p[:,2])
	plt.show()	

	return 
 





if __name__ == "__main__":
	visualize()
	


















