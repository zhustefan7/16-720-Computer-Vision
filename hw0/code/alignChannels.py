import numpy as np
import matplotlib
from scipy.misc import toimage	
from PIL import Image

def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""
    (Row,Col)=red.shape
    search_range=60
    ref=500
    # g_col_shift=0
    # b_col_shift=0
    # g_row_shift=0
    # b_row_shift=0

    temp=50000000
    for i in range(search_range):
    	euc_diff_col=find_euclidean_col(red[ref,:],green[ref-search_range//2+i,:])
    	if euc_diff_col <= temp:
    		temp=euc_diff_col
    		g_col_shift=-search_range//2+i


    temp=50000000

    for i in range(search_range):
    	euc_diff_col=find_euclidean_col(green[ref,:],blue[ref-search_range//2+i,:])
    	# print(euc_diff_col)
    	if euc_diff_col <= temp:
    		print(temp)
    		temp=euc_diff_col
    		b_col_shift=-search_range//2+i

    total_col_shift=g_col_shift+b_col_shift


 
    temp=50000000

    for i in range(search_range):
    	euc_diff_row=find_euclidean_col(red[:,ref],green[:,ref-search_range//2+i])
    	if euc_diff_row <= temp:
    		temp=euc_diff_row
    		g_row_shift=-search_range//2+i


    temp=50000000

    for i in range(search_range):
    	euc_diff_row=find_euclidean_col(green[:,ref],blue[:,ref-search_range//2+i])
    	if euc_diff_row <= temp:
    		temp=euc_diff_row
    		b_row_shift=-search_range//2+i

    total_row_shift=g_row_shift+b_row_shift
    print(g_row_shift)
    print(b_row_shift)



    total_row_shift=0
    g_row_shift=0
    b_row_shift=0



    rgbArray=np.zeros((Row-total_row_shift,Col-total_col_shift,3),'uint8')
    rgbArray[..., 0] = red[0:Row-total_row_shift,0:Col-total_col_shift]
    rgbArray[..., 1] = green[g_row_shift:Row-b_row_shift,g_col_shift:Col-b_col_shift]
    rgbArray[..., 2] = blue[total_row_shift:Row,total_col_shift:Col]
    rgbResult = Image.fromarray(rgbArray)

    print(g_col_shift)
    print(b_col_shift)






    return rgbResult




def find_euclidean_col(v1,v2):
	result=0
	for i in range(len(v1)):
		curr_diff=(int(v1[i])-int(v2[i]))**2
		result+=curr_diff
	return result



