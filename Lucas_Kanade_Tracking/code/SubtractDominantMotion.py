import numpy as np
from LucasKanadeAffine import*
from InverseCompositionAffine import*
import matplotlib.pyplot as plt
import scipy.ndimage


video_frames=np.load('../data/carseq.npy')
image1=video_frames[:,:,400]
image2=video_frames[:,:,401]
def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    threshold=0.3
    mask = np.ones(image1.shape, dtype=bool)
    # m = InverseCompositionAffine(image1 , image2)
    m = LucasKanadeAffine(image1, image2)
    image1_warped=scipy.ndimage.affine_transform(image1,-m, offset=0.0, output_shape=None, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
    diff=abs(image2-image1_warped)
    mask[diff>threshold]=1
    mask[diff<=threshold]=0

    mask=scipy.ndimage.morphology.binary_erosion(mask)

    return mask


if __name__=="__main__":
	SubtractDominantMotion(image1,image2)