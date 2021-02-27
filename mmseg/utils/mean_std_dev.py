/*
Script for calculating mean and std deviation of the training image dataset
*/

import numpy as np
from PIL import Image
import os
import glob

src_dir = "C:/Users/Xyedj/Desktop/datasets/wires/PLDU/pldu dataset/train/aug_data/0.0_0" #path to dir containing training images
mean_dataset=0
std_dataset=0
len_dataset= len(os.listdir(src_dir))
print(len_dataset)

means_list=[]
std_list=[]

# i=0; debugging

for imgfile in glob.iglob((os.path.join(src_dir, "*.jpg"))):
    # load image
    # if i==10:  #for debugging purposes
    #     break
    image = Image.open(imgfile)
    pixels = np.asarray(image)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # calculate per-channel means and standard deviations for each image
    means= pixels.mean(axis=(0,1), dtype='float64')
    std= pixels.std(axis=(0,1), dtype='float64')

    #append in the list for all the dataset
    means_list.append(means)
    std_list.append(std)


    print('Means: %s' % means)
    # print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))
    print('Std devs: %s' % std)

    # per-channel centering of pixels.
    # pixels -= means #this step is not needed as it will be done automatically in mmsegmentation setup

    # confirm it had the desired effect
    # means = pixels.mean(axis=(0,1), dtype='float64')
    # print('Means: %s' % means)
    # print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))

    # i= i+1 debugging

#Averaging results across the whole dataset
mean_arr=np.array(means_list)
std_arr= np.array(std_list)


mean_dataset = np.sum(mean_arr, 0)/len_dataset
std_dataset= np.sum(std_arr,0)/len_dataset

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)}) #restrict to 3 decimal places

print(mean_dataset)
print(std_dataset)
