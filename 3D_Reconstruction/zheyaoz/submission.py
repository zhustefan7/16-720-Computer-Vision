import numpy as np
from helper import*
import cv2
import numpy.linalg as la
import math
import scipy.ndimage

"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation   
    U = []
    #scale the data
    pts1 = pts1/M
    pts2 = pts2/M
    #populate U matrix
    for i in range(len(pts1)):
        pt1 = pts1[i,:]
        pt2 = pts2[i,:]
        xl = pt1[0]
        yl = pt1[1]
        xr = pt2[0]
        yr = pt2[1]
        temp_row = [xl*xr,xl*yr, xl, yl*xr,yl*yr,yl,xr,yr,1]
        U.append(temp_row)
    U = np.array(U)
 
    #calculate f
    UtU = np.matmul(np.transpose(U),U)
    w,v = np.linalg.eigh(UtU)
    F = v[:,0]

    #refine f
    F = refineF(F, pts1, pts2)

    #rescale f
    t = 1/M
    T= np.array([[t,0,0],[0,t,0],[0,0,1]]) 
    F = np.matmul(np.matmul(np.transpose(T),F),T)

    # np.savez("../Files_to_submit/q2_1.npz" , F=F, M=M)
    

    return F



'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    indices=np.random.permutation(np.arange(len(pts1)))
    indices=indices[0:7]
    U = []
    Farray = []

    #scale the data
    pts1 = pts1/M
    pts2 = pts2/M

    #populate U matrix using 7 points
    for i in range(len(indices)):
        pt1 = pts1[indices[i],:]
        pt2 = pts2[indices[i],:]
        xl = pt1[0]
        yl = pt1[1]
        xr = pt2[0]
        yr = pt2[1]
        temp_row = [xl*xr,xl*yr, xl, yl*xr,yl*yr,yl,xr,yr,1]
        U.append(temp_row)

    U = np.array(U)
    UtU = np.matmul(np.transpose(U),U)
    w,v = np.linalg.eigh(UtU)
    F1 = v[:,0]
    F2 = v[:,1]
    F1 =  np.reshape(F1,(3,3))
    F2 =  np.reshape(F2,(3,3))

    fun = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)

    a0=fun(0)
    a1=2*(fun(1)-fun(-1))/3-(fun(2)-fun(-2))/12
    a2=0.5*fun(1)+0.5*fun(-1)-fun(0)
    a3 = 0.5*(fun(1)-fun(-1)-2*a1)

    coeff= [a0, a1, a2 , a3] 
    roots = np.roots(coeff)
    print('roots',roots)

    roots = np.real(roots)

    
    #rescale f
    t = 1/M
    T= np.array([[t,0,0],[0,t,0],[0,0,1]]) 

    for i in range(len(roots)):
        curr_F = roots[i]*F1 + (1- roots[i])*F2
        curr_F = np.matmul(np.matmul(np.transpose(T),curr_F),T)
        Farray.append(curr_F)

    np.savez("../Files_to_submit/q2_2_temp.npz" ,Farray, M, pts1, pts2)
    return Farray


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E=np.matmul(np.transpose(K2),F)
    E=np.matmul(E,K1)
    
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    P=[]
    error=0
    for i in range(len(pts1)):
        A=[]
        pt1 = pts1[i,:]
        pt2 = pts2[i,:]
        xl = pt1[0]
        yl = pt1[1]
        xr = pt2[0]
        yr = pt2[1]

        A.append(yl*C1[2,:]-C1[1,:])
        A.append(C1[0,:]-xl*C1[2,:])
        A.append(yr*C2[2,:]-C2[1,:])
        A.append(C2[0,:]-xr*C2[2,:])
        
        A=np.array(A)
        ATA=np.matmul(np.transpose(A),A)
        w,v = np.linalg.eigh(ATA)
        X=v[:,0]
        X=np.divide(X,X[3])
        # print(X.sha)

        projectet_l=np.matmul(C1,X)
        projectet_r=np.matmul(C2,X)

        projectet_l=np.divide(projectet_l,projectet_l[2])
        projectet_r=np.divide(projectet_r,projectet_r[2])


        error+=la.norm(projectet_l[0:2]-pt1)+la.norm(projectet_r[0:2]-pt2)

        X=X[0:3]
        P.append(X)
    P=np.array(P)


        
    return P,error



'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''



def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation

    #converting im1 and im2 to greyscale
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    #initialization
    H, W = im1.shape
    p1= np.array([[x1] ,[y1],[1]])
    min_dis = None
    min_x2 = None
    min_y2 = None
    #calcualte Epipolar line
    epline=np.matmul(F,p1)
    epline = np.divide(epline,epline[2][0])
    a=epline[0][0]
    b=epline[1][0]
    c=epline[2][0]

    #create reference patch
    patch_width = 3
    refer_patch = im1[y1-math.floor(patch_width/2):y1+math.ceil(patch_width/2),x1-math.floor(patch_width/2):x1+math.ceil(patch_width/2)]

    # print(refer_patch.shape)
    # for y2 in range(math.floor(patch_width/2),H-math.ceil(patch_width/2)):
    for y2 in range(y1-3,y1+3):
            x2=int(round((-b*y2-c)/a))
            if math.floor(patch_width/2) < x2 and x2 < W- math.ceil(patch_width/2):
                curr_patch = im2[y2-math.floor(patch_width/2):y2+math.ceil(patch_width/2),x2-math.floor(patch_width/2):x2+math.ceil(patch_width/2)]

                # Calculate distance
                # dis = la.norm(curr_patch - refer_patch)
                refer_patch = np.reshape(refer_patch, (patch_width**2, 1))
                curr_patch = np.reshape(curr_patch, (patch_width**2, 1))
                dis = scipy.spatial.distance.euclidean(curr_patch, refer_patch)
                # dis = scipy.spatial.distance.cityblock(curr_patch , refer_patch)
                dis = scipy.ndimage.gaussian_filter(dis,sigma = 50)
                if min_dis == None:
                    min_dis = dis
                    min_x2 = x2
                    min_y2 = y2
                else:
                    if dis < min_dis:
                        min_dis = dis
                        min_x2 = x2
                        min_y2 = y2

    return [min_x2,min_y2]




# if __name__ == "__main__":
#     correspondence=np.load('../data/some_corresp.npz')
#     pts1=correspondence['pts1']
#     pts2=correspondence['pts2']

#     I1 = cv2.imread('../data/im1.png')
#     I2 = cv2.imread('../data/im2.png')

#     H,W,D=I1.shape
#     M=max(H,W)
#     # F=eightpoint(pts1,pts2,M)
 
#     # q2_2 = np.load("../Files_to_submit/q2_2.npz")
#     # F=q2_2['arr_0']
#     # print('F', F)
#     # Farray = sevenpoint(pts1,pts2,M)
#     # for i in range(3):
#     #     F = refineF(Farray[i], pts1, pts2)
#     #     displayEpipolarF(I1, I2, F)
#     Farray = np.load("../Files_to_submit/q2_2_temp.npz")
#     Farray = Farray['arr_0']
#     F = refineF(Farray[2],pts1 ,pts2)
    # np.savez("../Files_to_submit/q2_2.npz" ,F, M, pts1, pts2)

    # print(F)




    # intrinsics=np.load('../data/intrinsics.npz')
    # K1=intrinsics['K1']
    # K2=intrinsics['K2']

    # E=essentialMatrix(F, K1, K2)
    # print('essentialMatrix', E)
    # M2s=camera2(E)
    # # print(M2s.shape)
    # # print(intrinsics.files)

    displayEpipolarF(I1, I2, F)
    # C1=np.matmul(K1,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
    # M2, C2, p = findM2(I1, I2, K1, K2, pts1, pts2, F)
    # p,error = triangulate(C1, pts1, C2, pts2)
    # print(p)
    # print('error' , error)
    # epipolarCorrespondence(I1, I2, F, pts1[1,:][0], pts2[1,:][1] )
    # epipolarMatchGUI(I1, I2, F)