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

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    if len(image.shape) == 2:
        image = np.tile(image[:, np.newaxis], (1, 1, 3))

    if image.shape[2] == 4:
        image = image[:,:,0:3]

    image = skimage.color.rgb2lab(image)
    
    scales = [1,2,4,8,8*np.sqrt(2)]
    for i in range(len(scales)):
        for c in range(3):
            #img = skimage.transform.resize(image, (int(ss[0]/scales[i]),int(ss[1]/scales[i])),anti_aliasing=True)
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i])
            if i == 0 and c == 0:
                imgs = img[:,:,np.newaxis]
            else:
                imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_laplace(image[:,:,c],sigma=scales[i])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[0,1])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[1,0])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)

    return imgs

def get_visual_words(image,dictionary):

    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    # dictionary=np.load(dictionary.npy)
    # ----- TODO -----

    (rows,cols)=(image.shape[0],image.shape[1])
    image_response=extract_filter_responses(image)
    image_response=np.reshape(image_response,(rows*cols,image_response.shape[2]))


    distances=scipy.spatial.distance.cdist(image_response, dictionary, metric='euclidean')

    #get the minimum indices
    Indices=np.argmin(distances,axis=1)[:,np.newaxis]
    #map the corresponding dictionary entires to the wordmap$
    wordmap=np.reshape(Indices,(rows,cols))



    return wordmap







def compute_dictionary_one_image(args):
    # print('entered')
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''
   # ----- TODO -----

    i, alpha, image_path, temp_data_path= args


    curr_img=imageio.imread(image_path)

    (rows,cols,d)=curr_img.shape
    filter_response=extract_filter_responses(curr_img)
    filter_response=np.reshape(filter_response,(rows*cols,filter_response.shape[2])) #reshaping the filter response into a 2D row vector
    random_index=np.random.permutation(rows*cols)[0:alpha]   #to generate randomized index
    # sampled_image_index=random_index[0:alpha]
    # print('sampled_image_index',sampled_image_index)
    filter_response=filter_response[random_index]            #index into the filter_response matrix
    np.save(temp_data_path+'/Image%s.npy'%i,filter_response)
    return

def compute_dictionary(num_workers=4):

    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel
    
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''

    # ----- TODO -----
    K=200
    alpha=200
    # image_paths=['../data/aquarium','../data/desert','../data/highway','../data/kitchen','../data/laundromat','../data/park','../data/waterfall','../data/windmill']
    # temp_data_paths=['../data/temp_data/aquarium_temp','../data/temp_data/desert_temp','../data/temp_data/highway_temp','../data/temp_data/kitchen_temp','../data/temp_data/laundromat_temp','../data/temp_data/park_temp','../data/temp_data/waterfall_temp','../data/temp_data/windmill_temp']
    train_data=np.load('../data/train_data.npz')['files']
    train_data_size=len(train_data)
    process_list=[]
    temp_data_path='../data/training_temp_data'
    temp_data_list=os.listdir(temp_data_path)
    filter_responses=[]   #the list that stores the filter response of all training images


    for i in range(train_data_size):        #loop through all image directory and temp_data storage direcotyr
        # print(curr_size)
        image_path='../data/'+train_data[i]       

        process_list.append((i,alpha,image_path,temp_data_path))  #create the list for the subprocessor
            # compute_dictionary_one_image((j,alpha,image_path,temp_data_path))

        # next line works, but with [something,something,...] as an argument
    pool = multiprocessing.Pool(processes=num_workers)
    pool.map(compute_dictionary_one_image, process_list)
    pool.close()
    pool.join()

    #Remove DS_Store files
    removeTmpFiles(temp_data_path)

    #load the temp files and append filter responses to the list 
    for i in range(len(temp_data_list)):
        filter_responses.append(np.load(temp_data_path+'/'+temp_data_list[i]))

    all_filter_response=np.concatenate(filter_responses)
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(all_filter_response)
    dictionary = kmeans.cluster_centers_
    np.save('dictionary.npy',dictionary)

    return

def process_kmeans():
    K=200
    temp_data_path='../data/training_temp_data'
    temp_data_list=os.listdir(temp_data_path)

    # curr_data_list=os.listdir(temp_data_paths[0])
    filter_responses=[]
    for i in range(len(temp_data_list)):
        filter_responses.append(np.load(temp_data_path+'/'+temp_data_list[i]))

    all_filter_response=np.concatenate(filter_responses)
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(all_filter_response)
    dictionary = kmeans.cluster_centers_
    np.save('dictionary.npy',dictionary)


def removeTmpFiles(path):
    if path.split("/")[-1] == '.DS_Store':
        os.remove(path)
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            removeTmpFiles(path + "/" + filename)


if __name__ == "__main__":


    image_path='../data/aquarium/sun_aztvjgubyrgvirup.jpg'
    # image=imageio.imread(image_path)
    image=cv2.imread(image_path)
    print(image.shape)
    response_map=extract_filter_responses(image)
    # dictionary = np.load("dictionary.npy")
    # wordmap=get_visual_words(image,dictionary)
    util.display_filter_responses(response_map)

    # util.save_wordmap(wordmap,'labelme_aacpgupgzvdjapw_wordmap.jpeg')


