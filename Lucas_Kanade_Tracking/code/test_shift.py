import cv2
import scipy.ndimage
import numpy as np
from scipy.interpolate import RectBivariateSpline
def test_shift():
	# print('1')
	video_frames=np.load('../data/carseq.npy')
	frame=video_frames[:,:,0]
	print(frame.shape)
	# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	print(frame[239,310])

	nx, ny = frame.shape[1], frame.shape[0]
	X= np.arange(0, nx, 1)
	Y= np.arange(0, ny, 1)
	# X= np.meshgrid(np.arange(0, nx, 1))	
	# Y=np.meshgrid(np.arange(0, ny, 1))
	# print(X)
	# print(np.arange(0, nx, 1))
	interp_spline = RectBivariateSpline(Y, X, frame)

	rows=np.linspace(0,ny,num=ny+1)
	cols=np.linspace(0,nx,num=nx)
	# print(xx)
	a=interp_spline(rows,cols)
	print(a.shape)
	# a=interp_spline.ev([0:nx],[0:ny])



	print(a)
	print(frame)
	# cv2.imshow('frame',interp_spline)
	# cv2.waitKey(0)
	#row, col, z
	shift=[0,100]
	output_image=scipy.ndimage.shift(frame, shift, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
	# cv2.imshow('fs',output_image)
	# cv2.waitKey(0)
	pass



if __name__=='__main__':
	test_shift()