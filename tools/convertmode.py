import glob
from PIL import Image
import os
import numpy as np

# src_class="pldu"
src_dir = "C:/Users/Xyedj/Desktop/datasets/wires/PLDU/train/aug_gt/0.0_0"
dst_dir = "C:/Users/Xyedj/Desktop/datasets/wires/PLDU/train/aug_gt/pmode/"

def convert_mode(): #convert data to 'P' mode
    for imgfile in glob.iglob((os.path.join(src_dir, "*.png"))):
        filename = imgfile.split("\\")[1]  # extract filename with .png extension
        final_file_name= dst_dir+filename
        pimg = Image.open(imgfile)
        for pixel in pimg.getdata():
            if (pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255):  # white pixels
                pixel=[1,1,1]
        # print(pimg.format)
        # print(pimg.size)
        # print(pimg.mode)
        palette = [[0, 0, 0], [255, 255, 255]]
        pimg = pimg.convert("P")
        pimg.putpalette(np.array(palette, dtype=np.uint8))
        # print(pimg.mode)
        pimg=pimg.save(final_file_name)

convert_mode()
