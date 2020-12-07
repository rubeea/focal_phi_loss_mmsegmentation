import argparse
import os
import os.path as osp
import tempfile
import zipfile

import cv2
import mmcv
from PIL import Image  
import PIL  

TRAIN_LEN= 363 #80% of the total dataset (453 images)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PLDU dataset to mmsegmentation format')
    parser.add_argument(
        'images_path', help='the images part of PLDU dataset')
    parser.add_argument(
        'labels_path', help='the labels part of PLDU dataset')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    images_path = args.images_path
    labels_path = args.labels_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'pldu')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        print('Extracting images.zip...')
        zip_file = zipfile.ZipFile(images_path)
        zip_file.extractall(tmp_dir)

        print('Generating training and validation datasets...')
        now_dir = osp.join(tmp_dir, 'images')
        for img_name in sorted(os.listdir(now_dir))[:TRAIN_LEN]:
            img = Image.open(osp.join(now_dir, img_name))
            img= img.save(osp.join(out_dir, 'img_dir', 'train',
                    osp.splitext(img_name)[0] +'.jpg'))
            
         
        for img_name in sorted(os.listdir(now_dir))[TRAIN_LEN:]:
            img =Image.open(osp.join(now_dir, img_name))
            img= img.save(osp.join(out_dir, 'img_dir', 'val',
                    osp.splitext(img_name)[0] + '.jpg'))

        print('Extracting gt.zip...')
        zip_file = zipfile.ZipFile(labels_path)
        zip_file.extractall(tmp_dir)

        print('Generating annotations for training and validation datasets...')
        
        if osp.exists(now_dir):
            now_dir = osp.join(tmp_dir, 'gt')
            for img_name in sorted(os.listdir(now_dir))[:TRAIN_LEN]:
                img = Image.open(osp.join(now_dir, img_name))
                img=img.save(osp.join(out_dir, 'ann_dir', 'train',
                             osp.splitext(img_name)[0] + '.png'))
            
            for img_name in sorted (os.listdir(now_dir))[TRAIN_LEN:]:
                img = Image.open(osp.join(now_dir, img_name))
                # The annotation img should be divided by 128, because some of
                # the annotation imgs are not standard. We should set a
                # threshold to convert the nonstandard annotation imgs. The
                # value divided by 128 is equivalent to '1 if value >= 128
                # else 0'
                img= img.save(osp.join(out_dir, 'ann_dir', 'val',
                             osp.splitext(img_name)[0] + '.png'))
        
        
        print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    main()
