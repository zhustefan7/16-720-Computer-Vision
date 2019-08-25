import numpy as np
import matplotlib.pyplot as plt
import os
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.patches as mpatches

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    # print(type(image))
    #denoise the image
    # image = skimage.restoration.denoise_tv_chambolle(image, weight=0.1, eps=0.0002, n_iter_max=200, multichannel=False)
    # image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    image = skimage.filters.gaussian(image, sigma=1.0)
    #convert the image to greyscale
    image = skimage.color.rgb2gray(image)

    # #threshold the image
    thresh = skimage.filters.threshold_minimum(image)
    bw = (image < thresh)
    # print(bw)

  
    #apply morphology
    morphed = skimage.morphology.binary_opening(bw, selem=None, out=None)
   

    #label regions
    label_image = skimage.measure.label(bw)
    image_label_overlay = skimage.color.label2rgb(label_image, image=bw)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)
    
    counter = 0
    area_list=[]
    for region in skimage.measure.regionprops(label_image):
    	area_list.append(region.area)

    mean = np.mean(area_list)
    std = np.std(area_list)


    for region in skimage.measure.regionprops(label_image):
    	if abs(region.area -mean) < 4*std and region.area > mean/2:
    		# counter+=1
    		minr, minc, maxr, maxc = region.bbox
    		rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False, edgecolor='green', linewidth=2)
    		ax.add_patch(rect)

    		bboxes.append((minr, minc, maxr, maxc))
    # print(counter)

    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.savefig('../image_submission/find_letter_04_deep.jpeg')
    # plt.show()

    bw =1.0-1.0*bw

    


    return bboxes, bw




if __name__ == "__main__":

    im1 = skimage.img_as_float(skimage.io.imread('../images/04_deep.jpg'))
    bboxes, bw = findLetters(im1)
    # print(type(bw))


    # plt.gray()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(bw)
    # plt.gray()
    # plt.show()
