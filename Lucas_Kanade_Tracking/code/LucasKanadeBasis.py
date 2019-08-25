import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy import linalg as la
import cv2
import scipy.ndimage
import copy


def LucasKanadeBasis(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here



    # Put your implementation here
    p = np.zeros(2)
    threshold = 0.1
    x1 =rect[0,0]  
    y1 = rect[0,1]  
    x2 = rect[0,2]  
    y2 = rect[0,3]


    #interpolate the entire images
    nx, ny = It.shape[1], It.shape[0]
    X= np.arange(0, nx, 1)
    Y= np.arange(0, ny, 1)	

    It_interpolation = RectBivariateSpline(Y, X, It)
    It1_interpolation = RectBivariateSpline(Y, X, It1)

    cols=np.linspace(x1,x2,int(x2)-int(x1)+1)
    rows=np.linspace(y1,y2,int(y2)-int(y1)+1)

    It_rect=It_interpolation(rows,cols)

    #flatten B
    H,W,k=bases.shape
    B=np.reshape(bases,(H*W,k))
    BB_T=np.matmul(B,np.transpose(B))


    while True:
        H,W=It_rect.shape

        # print(It_rect.shape)

        ##############################
        #warp it1 first then cut out the rectangle
        shift=[-p[1],-p[0]]
        It1_warped=scipy.ndimage.shift(It1, shift, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
      
        gradient=np.gradient(It1_warped)
        gradient_x=gradient[1]
        gradient_y=gradient[0]
        #interpolate on the gradient
        gradient_x_interp=RectBivariateSpline(Y, X, gradient_x)
        gradient_y_interp=RectBivariateSpline(Y, X, gradient_y)

        it1_warped_iterpolation=RectBivariateSpline(Y, X, It1_warped)

        I_w=it1_warped_iterpolation(rows,cols)

        #compute error image
        b=It_rect-I_w

        #reshape error_image to 1xn
        b = np.reshape(b,(H*W,1))
        #compute gradient
        gradient_x=gradient_x_interp(rows,cols)
        gradient_y=gradient_y_interp(rows,cols)


        #reshaping gradient matrices to nx1
        gradient_x=np.reshape(gradient_x,(H*W,1))
        gradient_y=np.reshape(gradient_y,(H*W,1))

        #concatenate gradient matrices in x and y direction to nx2
        gradient=np.concatenate((gradient_x,gradient_y),axis=1)
        jacobian = np.array([[1,0],[0,1]])

        A=np.matmul(gradient,jacobian)
        # AT=np.transpose(A)

        A_new=A-np.matmul(BB_T,A)
        b_new=b-np.matmul(BB_T,b)
        AT_new=np.transpose(A_new) 

        # #computer Hessian
        H=np.matmul(AT_new,A_new)
        delta_p=np.matmul(np.matmul(la.inv(H),AT_new),b_new)
        #update p
        p[0]=p[0]+delta_p[0,0]
        p[1]=p[1]+delta_p[1,0]

        # print(delta_p)
        if (la.norm(delta_p) < threshold):
          break
    return p
    
