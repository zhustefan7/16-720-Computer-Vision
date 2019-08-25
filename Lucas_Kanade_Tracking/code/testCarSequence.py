import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from numpy import linalg as la
from LucasKanade import*
import scipy.ndimage
import cv2
# write your script here, we recommend the above libraries for making your animation
video_frames=np.load('../data/carseq.npy')

def testCarSequence():
	original_rect=np.array([59,116,145,151])[np.newaxis,:]
	new_rect=original_rect
	rect=np.array([59,116,145,151])[np.newaxis,:]
	p0=np.zeros(2)
	p=np.zeros(2)
	for i in range(video_frames.shape[2]-1):
		It=video_frames[:,:,i]
		It1=video_frames[:,:,i+1]



		p=LucasKanade(It, It1, new_rect, p)

		#the rectangle of the current frame
		x1=new_rect[0,0]+p[0]
		y1=new_rect[0,1]+p[1]
		x2=new_rect[0,2]+p[0]
		y2=new_rect[0,3]+p[1]

		new_rect=np.array([x1,y1,x2,y2])[np.newaxis,:]
		#to populate the final output
		rect=np.concatenate((rect,new_rect),axis=0)

		#visualizing
		if i==1 or i==100 or i==200 or i==300 or i==400:
			It1_rgb=cv2.merge((It1,It1,It1))
			It1_rgb=It1_rgb*255
			cv2.rectangle(It1_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1)
			cv2.imwrite('../Writeup_images/LK_frame_%d.png'%i,It1_rgb)


	print(rect)
	np.save('carseqrects.npy',rect)
	return rect



def animate():

	rect=np.load('carseqrects.npy')
	fig=plt.figure()
	ims=[]
	for i in range(video_frames.shape[2]):
		curr_rect=rect[i,:]
		x1=curr_rect[0]
		y1=curr_rect[1]
		x2=curr_rect[2]
		y2=curr_rect[3]
		frame=video_frames[:,:,i]
		frame_rgb=cv2.merge((frame,frame,frame))
		cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1)
		im=plt.imshow(frame_rgb,animated=True) 
		ims.append([im])

	ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
	plt.show()




if __name__=='__main__':
	# testCarSequence()
	animate()

