import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import cv2
import matplotlib.pyplot as plt
import multiprocessing


def build_recognition_system(num_workers=4):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''



    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    # ----- TODO -----
    SPM_layer_num=3
    (K,F)=dictionary.shape
    features=[]
    train_images=train_data['files']
    labels=train_data['labels']
    output_labels=[]
    process_list=[]

    for i in range(len(train_images)):
        process_list.append(('../data/'+train_images[i],dictionary,SPM_layer_num,K))
        # feature=get_image_feature('../data/'+train_images[i],dictionary,SPM_layer_num,K)
        # features.append(feature)
        output_labels.append(labels[i])


    pool = multiprocessing.Pool(processes=num_workers)
    features=pool.map(get_image_feature,process_list)
    pool.close()
    pool.join()


    np.savez('trained_system.npz',dictionary,features,output_labels,SPM_layer_num)
    return






def evaluate_recognition_system(num_workers=6):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''


    test_data = np.load("../data/test_data.npz")
    # test_data=np.load("../data/test_data_small.npz")
    trained_system = np.load("trained_system.npz")
    # ----- TODO -----
    dictionary=trained_system['arr_0']
    features=trained_system['arr_1']     #histogram of all training images
    train_data_labels=trained_system['arr_2']
    SPM_layer_num=trained_system['arr_3']
    K=dictionary.shape[0]
    # print(SPM_layer_num)
    test_images=test_data['files']
    # print(test_images)
    test_data_labels=test_data['labels']
    conf=np.zeros((8,8))
    process_list1=[]
    process_list2=[]





    # print('train image size',len(train_data_labels))

    # collecting multiprocess arguments for get_image_feature
    for i in range(len(test_images)):
        image_path='../data/'+test_images[i] 
        process_list1.append((image_path,dictionary,SPM_layer_num,K))


    pool1 = multiprocessing.Pool(processes=num_workers)
    #a list of all histograms of the testing data 
    histograms=pool1.map(get_image_feature,process_list1)   
    pool1.close()
    pool1.join()

    #collecting multiprocess arguments for distance_to_set
    for i in range(len(test_images)):
        process_list2.append((histograms[i],features))


    pool2 = multiprocessing.Pool(processes=num_workers)
    #a list of all similarity vectors of all test images 
    sims=pool2.map(distance_to_set,process_list2)   
    pool2.close()
    pool2.join()

    np.savez('historgrams & sims',histograms,sims)

    for i in range(len(test_images)):
        indx=np.argmax(sims[i])
        # print(len(sims[i]))
        guessed_label=train_data_labels[indx]
        test_image_label=test_data_labels[i]
        if test_image_label== 3 and guessed_label==7:
            print('Test label 3, guess label 7', test_images[i])

        elif test_image_label== 3 and guessed_label==2:
            print('Test label 3, guess label 2', test_images[i])

        elif test_image_label== 2 and guessed_label==7:
            print('Test label 2, guess label 7', test_images[i])


        conf[test_image_label,guessed_label]+=1 



#slower version
    # for i in range(len(test_images)):
    #     image_path='../data/'+test_images[i] 
    #     print('current Image',image_path)
    #     #histogram of the current testing image
    #     word_hist=get_image_feature((image_path,dictionary,SPM_layer_num,K))
    #     args=(word_hist,features)
    #     sim=distance_to_set(args)
    #     # the index of the closet histogram in traning image
    #     indx=np.argmax(sim)            
    #     guessed_label=train_data_labels[indx]
    #     test_image_label=test_data_labels[i]
    #     conf[test_image_label,guessed_label]+=1 

    accuracy=np.trace(conf)/np.sum(conf)


    print('conf',conf)
    print('accuracy',accuracy)
    
    np.savez('trial_result/K_200_Alpha_200.npz',conf,accuracy)

    return conf,accuracy










def get_image_feature(args):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K)
    '''


    file_path,dictionary,layer_num,K=args

    # ----- TODO -----
    image=imageio.imread(file_path)
    wordmap=visual_words.get_visual_words(image,dictionary)
    feature=get_feature_from_wordmap_SPM(wordmap,layer_num,K)

    return feature



def distance_to_set(args):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)



    '''
    word_hist,histograms=args

    # ----- TODO -----
    k=len(word_hist)
    n,k=histograms.shape
    sim=np.empty(n)
    for i in range(n):
        min_list=np.fmin(word_hist,histograms[i,:])
        similarity=sum(min_list)
        sim[i]=similarity

    return sim




def get_feature_from_wordmap(wordmap,dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    # ----- TODO -----

    (rows,cols)=wordmap.shape
    wordmap=np.reshape(wordmap,(rows*cols,1))
    hist,bin_edges=np.histogram(wordmap, bins=dict_size,normed=True)
    return hist




def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers    
    if layer_num==3 then it would have layer 0, layer 1, layer 2
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    # print('get_feature_from_wordmap_SPM')
    # ----- TODO -----
    histograms=[]
    sub_arrays=[]
    (rows,cols)=wordmap.shape
    # print(rows,cols,d)

    #Loop through each layer
    for i in range(layer_num-1,-1,-1):
        # print(i)
        nrows=math.ceil(rows/(2**(i)))
        ncols=math.ceil(cols/(2**(i)))

        for row in range(0,rows,nrows):
            for col in range(0,cols,ncols):
                block=wordmap[row:row+nrows,col:col+ncols]
                sub_arrays.append((block,i))


    #construct histogram all all the blocks 
    # print(sub_arrays)
    for element in sub_arrays:
        block,i=element
        if i ==0 or i==1:
            block=block*1/4
        else:
            block=block*1/2
  
        hist=get_feature_from_wordmap(block,dict_size)
        histograms.append(hist)

    

    #concatenate all histograms
    hist_all=np.concatenate(histograms)
    # print(hist_all.shape)

    return hist_all

    



# if __name__ == "__main__":

    # image_path=os.listdir('../data/aquarium')
    # image_path='../data/aquarium'
    # temp_data_path='../data/temp_data/aquarium_temp'
    # args=10,100,image_path,temp_data_path
    # compute_dictionary_one_image(args)
    # compute_dictionary(num_workers=4)
    # process_kmeans()

    # image_path='../data/laundromat/sun_aakuktqwgbgavllp.jpg'
    # image=cv2.imread(image_path)
    # dict_size=200
    # dictionary=np.load('dictionary.npy')
    # wordmap=visual_words.get_visual_words(image,dictionary)

    # # print(wordmap.shape)
    # all_hist=get_feature_from_wordmap_SPM(wordmap,3,dict_size)
    # print(all_hist.shape)
    # hist=get_feature_from_wordmap(wordmap,100)
    # build_recognition_system()
    # evaluate_recognition_system(num_workers=4)
    # print(hist)





    

