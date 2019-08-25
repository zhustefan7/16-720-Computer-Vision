import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import cv2
import util
import random
# from PIL import Image
from matplotlib import pyplot as plt
import visual_words
import visual_recog


#K 100 Alpha 200: Accuracy=0.114


def main():
	# visual_words.compute_dictionary(num_workers=4)
	# removeTmpFiles('../data/training_temp_data')
	# visual_words.process_kmeans()
	# visual_recog.build_recognition_system(num_workers=4)
	conf,accuracy=visual_recog.evaluate_recognition_system(num_workers=6)
	return 

def removeTmpFiles(path):
    if path.split("/")[-1] == '.DS_Store':
        os.remove(path)
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            removeTmpFiles(path + "/" + filename)





def create_test_data_small():
	test_data_small_image_path=['aquarium/sun_auxtusgjytgqmpis.jpg','kitchen/sun_atmnmiboxsxtnduv.jpg',
	'waterfall/sun_bhczjtvnuhhqeaxg.jpg','desert/sun_bthdhekortklqyvg.jpg','windmill/sun_blqipqrmlmccloro.jpg',
	'aquarium/sun_amwolbznqszsutww.jpg','waterfall/sun_axhoqjcnfanrwjut.jpg','laundromat/sun_afenfgcjranevxdd.jpg',
	 'kitchen/sun_aszqvzzkxjwjehht.jpg','windmill/sun_bnjvizuvdryjlxky.jpg','park/sun_aiqzpealjtmdbulg.jpg']


	create_test_data_small_labels=[0,4,6,2,7,0,6,5,4,7,1]
	np.savez('../data/test_data_small',files=test_data_small_image_path,labels=create_test_data_small_labels)




if __name__ == "__main__":

	# result=np.load('trial_result/K_200_Alpha_200.npz')
	# print(result['arr_0'])

	main()
	# create_test_data_small()
	# test_data_small=np.load('test_data_small.npz')
	# print(test_data_small['/files'])



# create_test_data_small()
# main()