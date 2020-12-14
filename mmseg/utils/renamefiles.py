import os
import glob
import shutil

src_dir = "C:/Users/Xyedj/Desktop/datasets/wires/PLDU/train/aug_gt_scale_1.5/315.0_1"
dst_dir= "C:/Users/Xyedj/Desktop/datasets/wires/PLDU/images/aug_gt_scale_1.5/315.0_1"

def copy_data():
    j=1
    # copy jpg files
    for imgfile in glob.iglob((os.path.join(src_dir, "*.png"))):
        jpgfile = imgfile.split("\\")[1]  # extract filename with .jpg extension
        print("---Copying-----:" + jpgfile)
        shutil.copy(imgfile, dst_dir) #copy .jpg image files to project mooring
        j=j+1

def rename_files():
    suffix_letter="_47"
    os.chdir(dst_dir)
    i=1
    newname=""
    for dst_imgfile in os.listdir():
        filename= dst_imgfile.split(".")[0]
        ext_file= dst_imgfile.split(".")[1]
        if(ext_file.casefold()=="jpg"):
            newname=filename+suffix_letter+".jpg"
        elif(ext_file.casefold()=="jpeg"):
            newname = filename+suffix_letter+ ".jpeg"
        elif(ext_file.casefold()=="png"):
            newname = filename+suffix_letter + ".png"
        elif (ext_file.casefold() == "gif"):
            newname = filename+suffix_letter + ".gif"
        else:
            "Invalid extension"
        #print(dst_imgfile)
        #print(newname)
        os.rename(dst_imgfile, newname)   #rename the newly copied file to a different name
        i=i+1
        #print("Renamed file. Value of i" + str(i))

    #decrement counter for one last increment at end
    i=i-1
    print("Renamed "+str(i)+" files")

copy_data()
rename_files()
