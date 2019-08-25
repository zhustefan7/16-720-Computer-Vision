import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import*
import scipy.ndimage
import cv2
from numpy import linalg as la

# write your script here, we recommend the above libraries for making your animation
# write your script here, we recommend the above libraries for making your animation
video_frames=np.load('../data/carseq.npy')

def testCarSequenceWithTemplateCorrection():
	original_rect=np.array([59,116,145,151])[np.newaxis,:]
	new_rect=original_rect
	rect=np.array([59,116,145,151])[np.newaxis,:]
	p0=np.zeros(2)
	p=np.zeros(2)
	p_cumulative=p
	It_original=video_frames[:,:,0]
	e=2

	for i in range(video_frames.shape[2]-1):
		It=video_frames[:,:,i]
		It1=video_frames[:,:,i+1]

		p=LucasKanade(It, It1, new_rect, p)

		p_cumulative=p_cumulative+p


		p_star=LucasKanade(It_original, It1, original_rect, p_cumulative)

		delta_p_star=p_star-p_cumulative

		p_cumulative=p_cumulative+delta_p_star
		print(p_cumulative)

		if abs(la.norm(p_star-p_cumulative))<e:

			#update the rectangle of the current frame
			x1=new_rect[0,0]+p[0]+delta_p_star[0]
			y1=new_rect[0,1]+p[1]+delta_p_star[1]
			x2=new_rect[0,2]+p[0]+delta_p_star[0]
			y2=new_rect[0,3]+p[1]+delta_p_star[1]



		new_rect=np.array([x1,y1,x2,y2])[np.newaxis,:]
		#to populate the final output
		rect=np.concatenate((rect,new_rect),axis=0)


	np.save('carseqrects-wcrt.npy',rect)
	return rect



def animate(): 

	rect_wcrt=np.load('carseqrects-wcrt.npy')
	rect=np.load('carseqrects.npy')
	fig=plt.figure()
	ims=[]
	for i in range(video_frames.shape[2]):
		frame=video_frames[:,:,i]
		frame_rgb=cv2.merge((frame,frame,frame))

		#draw the rectangle with template correction
		curr_rect=rect_wcrt[i,:]
		x1=curr_rect[0]
		y1=curr_rect[1]
		x2=curr_rect[2]
		y2=curr_rect[3]
		cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 1)


		#draw the rectangle without template correction
		curr_rect=rect[i,:]
		x1=curr_rect[0]
		y1=curr_rect[1]
		x2=curr_rect[2]
		y2=curr_rect[3]
		cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1)


		if i==1 or i==100 or i==200 or i==300 or i==400:
			frame_rgb=frame_rgb*255
			cv2.imwrite('../Writeup_images/LK_correction_frame_%d.png'%i,frame_rgb)



		im=plt.imshow(frame_rgb,animated=True) 
		ims.append([im])

	ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
	plt.show()




if __name__=='__main__':
	# testCarSequenceWithTemplateCorrection()
	animate()

