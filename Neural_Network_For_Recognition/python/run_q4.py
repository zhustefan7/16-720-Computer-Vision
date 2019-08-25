import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import pickle
import string

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    # print(img)
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    print('\n')
    print(img)
    # plt.imshow(bw)
    # plt.show()

    # plt.imshow(bw)
    # for bbox in bboxes:
    #     minr, minc, maxr, maxc = bbox
    #     rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                             fill=False, edgecolor='red', linewidth=2)
    #     plt.gca().add_patch(rect)
    # plt.show()



    # find the rows using..RANSAC, counting, clustering, etc.
    bboxes = sorted(bboxes, key = lambda l: (l[0]+l[2])/2)
    # minr_i, minc_i, maxr_i, maxc_i = bboxes[0]
    sorted_bboxes = []
    # last_y = (minr_i +maxr_i)/2
    y_list = []
    row_num = 0
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        curr_y = (minr + maxr)/2
        y_list.append(curr_y)
        mean = np.mean(y_list)


        # print('mean',mean)
        # print('std', std)
        # print('curr_y',curr_y)

        if curr_y - mean > maxr-minr:
            row_num +=1 
            y_list = []

        
        # print('row_num', row_num)
        sorted_bboxes.append(( minr, minc, maxr, maxc , row_num))



        # sorted_bboxes.append((minr, minc, maxr, maxc, (minc+maxc)/2))
        # sorted_bboxes=sorted(sorted_bboxes, key =lambda l :l[4])
        sorted_bboxes=sorted(sorted_bboxes, key =lambda l :(l[1]+l[3]/2))
        sorted_bboxes=sorted(sorted_bboxes, key =lambda l : l[4])


    # for bbox in sorted_bboxes:
    #     (y1, x1, y2, x2, row_num) = bbox
    #     print('row_num', x1 ,y1)

    char_list = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    saved_params = pickle.load(open('q3_weights.pickle','rb'))
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    chars = []
    predicted_letters = []
    for bbox in sorted_bboxes:
        minr, minc, maxr, maxc ,row_num = bbox
        width = maxc - minc
        height  = maxr - minr
        # box_width = max(width,height)
        # print(bw.shape)
        #crop out the character
        char = bw[minr:minr+height , minc:minc + width]
        char = np.pad(char, (30,30), 'constant', constant_values = (1,1))
        #remove stuff on the border
        # print(char.shape)
        # plt.imshow(char)
        # plt.show()
        char = np.transpose(char)
        char = skimage.transform.resize(char, (32, 32))
      

        
        char = np.reshape(char,(1,1024))
        h1 = forward(char ,saved_params,'layer1')
        probs = forward(h1 , saved_params, 'output', activation=softmax)
        indices = np.argmax(probs,axis=1)
        predicted_letter = char_list[indices]
        predicted_letters.append((predicted_letter[0],row_num))
        # print(predicted_letter)

    
    #Print the letters to different lines 
    pre_row_num = 0 
    for i in range(len(predicted_letters)):
        predicted_letter,curr_row_num = predicted_letters[i]
        if curr_row_num > pre_row_num:
            pre_row_num = curr_row_num
            print('\n')
            print(predicted_letter, end =" ")
        else:
            print(predicted_letter, end =" ")







    # print(predicted_letters)
    #     chars.append(char)
    # chars=np.array(chars)
    # chars=np.reshape(chars,(chars.shape[0],chars.shape[2]))
    # print(chars.shape)

    # load the weights
    # run the crops through your neural network and print them out
    # char_list = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    # print(char_list.shape)

    # saved_params = pickle.load(open('q3_weights.pickle','rb'))

    # for i in range(len(chars)):
    # print(chars[i].shape)
    # print(chars.shape)
    # h1 = forward(chars ,saved_params,'layer1')
    # probs = forward(h1 , saved_params, 'output', activation=softmax)
    # print(probs)
    # print(probs.shape)
    # print(probs[0,0,:])
    # indices = np.argmax(probs,axis=1)
    # print(indices.shape)
    # print(indices)

    # for i in range(probs.shape[0]):
    #     pred_char = probs[i].argmax()
    #     print(char_list[pred_char])

    # print(char_list)
    # predicted_letters = char_list[indices]

    # print(predicted_letters)



    
