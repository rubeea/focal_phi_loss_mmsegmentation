/*
# Script to convert the validation ground truth of pldu in .mat format to .png image format
*/

import scipy.io
import os
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt

src_dir = "C:/Users/Xyedj/Desktop/datasets/wires/PLDU/pldu_original_dataset/test_gt"

for matfile in glob.iglob((os.path.join(src_dir, "*.mat"))): #.mat file in test data
    filename= matfile.split("\\")[1].split(".")[0]  # extract filename without .mat extension
    filename= src_dir+"/"+filename+".png" # add the .png extension
    print(filename)
    mat = scipy.io.loadmat(matfile)
    mat.items()

    # Print the data
    print(mat["groundTruth"][0][0][0][0][0])
    data = np.array(mat["groundTruth"][0][0][0][0][0])
    # print(data)
    im = Image.fromarray(data*255)
    im.save(filename)
    # plt.imshow(data)
