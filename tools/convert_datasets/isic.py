import argparse
import os
import os.path as osp
import tempfile
import zipfile

import cv2
import mmcv
from PIL import Image
import PIL


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PLDU dataset to mmsegmentation format')
    parser.add_argument(
        'train_images_path', help='the train images part of ISIC dataset')
    parser.add_argument(
        'train_labels_path', help='the train labels part of ISIC dataset')
    parser.add_argument(
        'val_images_path', help='the val images part of ISIC dataset')
    parser.add_argument(
        'val_labels_path', help='the val labels part of ISIC dataset')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    train_images_path = args.train_images_path
    train_labels_path = args.train_labels_path
    val_images_path = args.val_images_path
    val_labels_path = args.val_labels_path

    if args.out_dir is None:
        out_dir = osp.join('/content/pldu_mmsegmentation/data', 'pldu')
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
        print('Extracting train images.zip...')
        zip_file = zipfile.ZipFile(train_images_path)
        zip_file.extractall(tmp_dir)

        print('Generating image training dataset...')
        now_dir = osp.join(tmp_dir, 'train_imgs')
        for img_name in sorted(os.listdir(now_dir)):
            img = Image.open(osp.join(now_dir, img_name))
            img = img.save(osp.join(out_dir, 'img_dir', 'train',
                                    osp.splitext(img_name)[0] + '.jpg'))

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        print('Extracting val images.zip...')
        zip_file = zipfile.ZipFile(val_images_path)
        zip_file.extractall(tmp_dir)

        print('Generating image training dataset...')
        now_dir = osp.join(tmp_dir, 'val_imgs')
        for img_name in sorted(os.listdir(now_dir)):
            img = Image.open(osp.join(now_dir, img_name))
            img = img.save(osp.join(out_dir, 'img_dir', 'val',
                                    osp.splitext(img_name)[0] + '.jpg'))

        print('Extracting train_gt.zip...')
        zip_file = zipfile.ZipFile(train_labels_path)
        zip_file.extractall(tmp_dir)

        print('Generating annotations for training dataset...')

        if osp.exists(now_dir):
            now_dir = osp.join(tmp_dir, 'train_gt')
            for img_name in sorted(os.listdir(now_dir)):
                img = Image.open(osp.join(now_dir, img_name))
                img = img.save(osp.join(out_dir, 'ann_dir', 'train',
                                        osp.splitext(img_name)[0] + '.png'))

        print('Extracting val_gt.zip...')
        zip_file = zipfile.ZipFile(val_labels_path)
        zip_file.extractall(tmp_dir)

        print('Generating annotations for validation dataset...')

        if osp.exists(now_dir):
            now_dir = osp.join(tmp_dir, 'val_gt')
            for img_name in sorted(os.listdir(now_dir)):
                img = Image.open(osp.join(now_dir, img_name))
                img = img.save(osp.join(out_dir, 'ann_dir', 'val',
                                        osp.splitext(img_name)[0] + '.png'))

        print('Removing the temporary files...')


print('Done!')

if __name__ == '__main__':
    main()
