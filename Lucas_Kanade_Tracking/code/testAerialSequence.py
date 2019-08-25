import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from SubtractDominantMotion import*

# write your script here, we recommend the above libraries for making your animation
video_frames=np.load('../data/aerialseq.npy')

def testAerialSequence():
	masks=[]
	for i in range(video_frames.shape[2]-1):
			It=video_frames[:,:,i]
			It1=video_frames[:,:,i+1]
			print(It.shape)
			mask=SubtractDominantMotion(It, It1)
			masks.append(mask)

			# if i%50==0:
			# 	plt.subplot(211)
			# 	plt.imshow(mask)
			# 	plt.show()


	np.save('invercompose_masks.npy',masks)
	return



def animate():

	masks=np.load('masks.npy')
	# masks=np.load('invercompose_masks.npy')

	fig=plt.figure()
	ims=[]
	for i in range(video_frames.shape[2]-1):
		mask=masks[i]
		frame=video_frames[:,:,i]
		# print(frame)
		frame_rgb=cv2.merge((frame,frame,frame))
		frame_rgb[mask==0]=(0,0,255)

		im=plt.imshow(frame_rgb,animated=True) 
		ims.append([im])

		frame_rgb=frame_rgb*255
		if i==30 or i==60 or i==90 or i==120:
			cv2.imwrite('../Writeup_images/LK_arial%d.png'%i,frame_rgb)

	ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
	plt.show()




if __name__=='__main__':
	testAerialSequence()
	animate()
