import argparse
import os
import os.path as osp
import tempfile
import zipfile

import cv2
import mmcv

TRAIN_LEN= 363 #80% of the total dataset (453 images)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PLDU dataset to mmsegmentation format')
    parser.add_argument(
        'training_path', help='the training part of PLDU dataset')
    parser.add_argument(
        'testing_path', help='the testing part of PLDU dataset')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    training_path = args.training_path
    testing_path = args.testing_path
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
        print('Extracting training.zip...')
        zip_file = zipfile.ZipFile(training_path)
        zip_file.extractall(tmp_dir)

        print('Generating training dataset...')
        now_dir = osp.join(tmp_dir, 'images')
        for img_name in os.listdir(now_dir)[:TRAIN_LEN]:
            img = mmcv.imread(osp.join(now_dir, img_name))
            mmcv.imwrite(
                img,
                osp.join(
                    out_dir, 'img_dir', 'train',
                    osp.splitext(img_name)[0] +'.jpg'))

        now_dir = osp.join(tmp_dir, 'gt')
        for img_name in os.listdir(now_dir)[:TRAIN_LEN]:
            img = mmcv.imread(osp.join(now_dir, img_name))
            mmcv.imwrite(
                img[:, :, 0] // 128,
                osp.join(out_dir, 'ann_dir', 'train',
                         osp.splitext(img_name)[0] + '.png'))

        print('Extracting test.zip...')
        zip_file = zipfile.ZipFile(testing_path)
        zip_file.extractall(tmp_dir)

        print('Generating validation dataset...')
        now_dir = osp.join(tmp_dir, 'images')
        for img_name in os.listdir(now_dir)[TRAIN_LEN:]:
            img = mmcv.imread(osp.join(now_dir, img_name))
            mmcv.imwrite(
                img,
                osp.join(
                    out_dir, 'img_dir', 'val',
                    osp.splitext(img_name)[0] + '.jpg'))

        now_dir = osp.join(tmp_dir, 'gt')
        if osp.exists(now_dir):
            for img_name in os.listdir(now_dir)[TRAIN_LEN:]:
                img = mmcv.imread(osp.join(now_dir, img_name))
                # The annotation img should be divided by 128, because some of
                # the annotation imgs are not standard. We should set a
                # threshold to convert the nonstandard annotation imgs. The
                # value divided by 128 is equivalent to '1 if value >= 128
                # else 0'
                mmcv.imwrite(
                    img[:, :, 0] // 128,
                    osp.join(out_dir, 'ann_dir', 'val',
                             osp.splitext(img_name)[0] + '.png'))

        
        print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    main()
